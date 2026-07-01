"""
STAGE 1 — longer-window corridor dynamics (resolve the boundary-pinned phase-kick response).

The clean energy-conserving phase-kick test (corridor_phasetest.py) showed structure-vs-void
propagation (node/bridge >> void) but was BOUNDARY-PINNED at +1200 steps (response still growing),
so we could not tell delayed corridor routing from slow whole-structure relaxation / global
collective mode / unresolved chaotic divergence. This re-runs the SAME clean phase kick with a
MUCH longer continuation, resolving the full response-time curve, on the top-3 finalists PLUS a
no-corridor negative control.

Energy-conserving phase kick psi -> psi*exp(i*theta*bump): identical perturbation at node / bridge
/ void, zero energy injected anywhere -> a fair comparison of where a phase twist propagates.

Per the interpretation rules:
  resolved bridge-SELECTIVE response (bridge>node, bridge>void, resolved lag, bounded/recovers)
      -> PROMISING_CORRIDOR_ROUTING_CANDIDATE
  structure>void but node~=bridge (no selectivity)        -> COUPLED_STRUCTURE_NONSELECTIVE
  response decays/disappears                              -> TRANSIENT_SCOUT_ARTIFACT
  grows without resolving / no recovery                   -> UNRESOLVED_LONG_TIMESCALE_RESPONSE
A boundary-pinned response is NOT called causal routing.

CAUTION: JAX scout-level; not proof; CuPy still required for any promotion.
WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/corridor_longwindow.py
"""
import os, sys, json, glob, csv, time
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td

L, dt = 10.0, 0.005
order = physics.SWEEP_PARAM_ORDER
BASE_SEED = 20260619
SETTLE = 800
LONG_STEPS, LONG_NSNAP = 4000, 100      # ~5x the previous window; snapshot every 40 steps
THETA = 0.4


def multiseed_ic(N, seed, K=6):
    rng = np.random.default_rng(seed)
    x = np.linspace(-L/2, L/2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L/12.0; psi = np.zeros((N, N, N), np.complex128)
    for _ in range(K):
        cx, cy, cz = rng.uniform(-L/2, L/2, 3)
        psi += np.exp(-((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/(2*w**2))
    noise = 0.01*(rng.standard_normal((N, N, N))+1j*rng.standard_normal((N, N, N)))
    return (psi+noise).astype(np.complex128)


def _bump(N, c, r):
    G = np.meshgrid(*([np.arange(N)]*3), indexing="ij")
    d2 = sum(np.minimum((G[a]-c[a]) % N, (c[a]-G[a]) % N).astype(float)**2 for a in range(3))
    return np.exp(-d2/(2*(r/1.5)**2)), d2


def _phase_kick(psi, c, N, r, theta):
    bump, _ = _bump(N, c, r)
    return (psi*np.exp(1j*theta*bump)).astype(np.complex128)


def _region_E(psi, mask):
    return float(np.sum(np.abs(psi[mask])**2))


def _curve_metrics(mdev, step_per_snap):
    """Resolve the response-time curve: peak, time-to-peak, recovery, growth, boundary-pinned."""
    T = len(mdev); peak = float(mdev.max()); ip = int(np.argmax(mdev))
    boundary = ip >= T-2
    recovery = float(mdev[-1]/(peak+1e-30))                 # <0.8 = decayed back; ~1 = saturated/grew
    # late growth: slope over last quarter (normalised by peak)
    q = max(2, T//4); late = (mdev[-1]-mdev[-q])/(q*step_per_snap)/(peak+1e-30)
    return {"peak": peak, "time_to_peak_steps": ip*step_per_snap, "peak_frac": ip/(T-1),
            "recovery_ratio": recovery, "late_slope": float(late), "boundary_pinned": boundary}


def run_config(par, N, kinds):
    """kinds: subset of {node,bridge,void,bridge_neg}. Returns per-kick curve metrics + curves."""
    pv = [par[k] for k in order]; dx = L/N
    snaps, fin = td.capture_trajectory(pv, multiseed_ic(N, BASE_SEED), N, L, dt, SETTLE, 20)
    if not fin:
        return {"status": "settle_nonfinite"}
    psi0 = snaps[-1]; nodes = td.detect_nodes(psi0, dx)
    if len(nodes) < 2:
        return {"status": "too_few_nodes", "n_nodes": len(nodes)}
    nodes = sorted(nodes, key=lambda n: -n["E"]); cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes])**(1/3))))
    geo = td.geometry_fields(psi0, par, dx); best, bp = -1, (0, 1)
    for i in range(len(cents)):
        for j in range(i+1, len(cents)):
            c = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
            if c > best:
                best, bp = c, (i, j)
    disp = cents[bp[1]]-cents[bp[0]]; disp = disp-N*np.round(disp/N)
    bridge_pt = np.round((cents[bp[0]]+0.5*disp)) % N
    rho = np.abs(psi0)**2; far = np.ones((N, N, N), bool)
    for c in cents:
        _, d2 = _bump(N, c, 2*node_r); far &= d2 > (2*node_r)**2
    void = np.array(np.unravel_index(np.argmin(np.where(far, rho, rho.max()+1)), rho.shape))
    omasks = [(_bump(N, c, node_r)[1] <= node_r*node_r) for c in cents[1:]]
    spc = LONG_STEPS//LONG_NSNAP

    def cont(p0):
        s, f = td.capture_trajectory(pv, p0, N, L, dt, LONG_STEPS, LONG_NSNAP); return s, f
    s_ctrl, f0 = cont(psi0)
    if not f0:
        return {"status": "control_nonfinite"}
    T = s_ctrl.shape[0]
    ctrlE = np.array([[_region_E(s_ctrl[t], m) for t in range(T)] for m in omasks])
    locs = {"node": cents[0], "bridge": bridge_pt, "void": void, "bridge_neg": bridge_pt}
    sign = {"node": 1, "bridge": 1, "void": 1, "bridge_neg": -1}
    out = {"status": "ok", "best_bridge_conductance": float(best), "n_nodes": len(nodes),
           "node_r": int(node_r), "steps_per_snap": spc}
    for nm in kinds:
        s_b, fb = cont(_phase_kick(psi0, locs[nm], N, node_r, sign[nm]*THETA))
        if not fb:
            out[nm] = {"finite": False}; continue
        bE = np.array([[_region_E(s_b[t], m) for t in range(T)] for m in omasks])
        mdev = (np.abs(bE-ctrlE)/(ctrlE[:, :1]+1e-30)).mean(0)
        m = _curve_metrics(mdev, spc); m["finite"] = True
        m["curve"] = [float(x) for x in mdev[::5]]            # downsampled curve for inspection
        out[nm] = m
    # ratios + recovery summary at peak
    if all(out.get(k, {}).get("finite") for k in ("node", "bridge", "void")):
        vp = out["void"]["peak"]+1e-30
        out["ratio_node_vs_void"] = out["node"]["peak"]/vp
        out["ratio_bridge_vs_void"] = out["bridge"]["peak"]/vp
        out["selectivity_bridge_vs_node"] = out["bridge"]["peak"]/(out["node"]["peak"]+1e-30)
    return out


def classify(r):
    if r.get("status") != "ok":
        return "NO_RESULT_" + r.get("status", "unknown").upper()
    if not all(r.get(k, {}).get("finite") for k in ("node", "bridge", "void")):
        return "NONFINITE_BRANCH"
    bvd = r["ratio_bridge_vs_void"]; nvd = r["ratio_node_vs_void"]; sel = r["selectivity_bridge_vs_node"]
    bridge = r["bridge"]
    resolved = (not bridge["boundary_pinned"]) and bridge["peak_frac"] < 0.9
    recovers = bridge["recovery_ratio"] < 0.8
    structure_over_void = max(bvd, nvd) > 1.5
    # decayed to nothing?
    if bridge["peak"] < 1e-3 and r["node"]["peak"] < 1e-3:
        return "TRANSIENT_SCOUT_ARTIFACT"
    if structure_over_void and resolved and bvd > 1.5 and sel > 1.3 and (recovers or r["n_nodes"] >= 2):
        return "PROMISING_CORRIDOR_ROUTING_CANDIDATE"
    if structure_over_void and (not resolved or bridge["recovery_ratio"] > 0.9):
        return "UNRESOLVED_LONG_TIMESCALE_RESPONSE"
    if structure_over_void:
        return "COUPLED_STRUCTURE_NONSELECTIVE"
    return "TRANSIENT_SCOUT_ARTIFACT"


def load_negative_control():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "BRIDGE_HUNT_2026*")))[-1]
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    def F(r, k):
        try: return float(r[k])
        except: return float("nan")
    nc = [r for r in rows if r["klass"] == "no_corridor_stable_nodes" and 2 <= F(r, "nodes") <= 8
          and F(r, "max_cond") < 0.05 and 0.5 <= F(r, "er") <= 2.0]
    nc.sort(key=lambda r: F(r, "max_cond"))
    r = nc[0]
    return {k: F(r, k) for k in order}, r["gen"], F(r, "max_cond")


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "BRIDGE_HUNT_2026*")))[-1]
    fz = json.load(open(os.path.join(d, "frozen_finalists.json")))
    print(f"STAGE 1 longer-window phase-kick ({LONG_STEPS} steps) — top 3 finalists + no-corridor control\n")
    report = []
    for fr in fz["finalists"][:3]:
        par = {k: float(fr["params"][k]) for k in order}
        label = f"gen{fr['generation']}_{fr['config_hash']}"
        t0 = time.time(); r = run_config(par, 48, ("node", "bridge", "void", "bridge_neg"))
        r["label"] = label; r["role"] = "finalist"; r["verdict"] = classify(r)
        _print(label, r, time.time()-t0); report.append(r)
    # negative control
    ncpar, ncgen, nccond = load_negative_control()
    t0 = time.time(); r = run_config(ncpar, 48, ("node", "void"))
    r["label"] = f"NEGCTRL_gen{ncgen}_cond{nccond:.3f}"; r["role"] = "no_corridor_control"
    r["verdict"] = "control"
    _print(r["label"], r, time.time()-t0, ctrl=True); report.append(r)
    od = os.path.join(d, "corridor_longwindow.json")
    json.dump(report, open(od, "w"), indent=2, default=float)
    print("=== STAGE 1 SUMMARY ===")
    for r in report:
        if r["role"] == "finalist":
            print(f"  {r['label']:28} -> {r['verdict']}")
        else:
            nk = r.get("node", {}); print(f"  {r['label']:28} (control) node/void="
                  f"{(nk.get('peak',0)/(r.get('void',{}).get('peak',1e-30)+1e-30)):.2f} "
                  f"[if >>1, structure-vs-void is NOT bridge-specific]")
    print(f"\nwrote {od}")


def _print(label, r, secs, ctrl=False):
    if r.get("status") != "ok":
        print(f"[{label}] {r.get('status')}  ({secs:.0f}s)\n"); return
    print(f"[{label}] bridge_cond={r['best_bridge_conductance']:.3f} nNodes={r['n_nodes']}")
    for nm in ("node", "bridge", "void", "bridge_neg"):
        x = r.get(nm)
        if x and x.get("finite"):
            print(f"   {nm:11} peak={x['peak']:.4f} t2peak={x['time_to_peak_steps']}st "
                  f"peakfrac={x['peak_frac']:.2f} recovery={x['recovery_ratio']:.2f} "
                  f"{'BOUNDARY-PINNED' if x['boundary_pinned'] else 'RESOLVED'}")
    if not ctrl and "ratio_bridge_vs_void" in r:
        print(f"   -> node/void={r['ratio_node_vs_void']:.2f} bridge/void={r['ratio_bridge_vs_void']:.2f} "
              f"bridge/node(selectivity)={r['selectivity_bridge_vs_node']:.2f}")
        print(f"   -> VERDICT: {r['verdict']}")
    print(f"   ({secs:.0f}s)\n")


if __name__ == "__main__":
    main()
