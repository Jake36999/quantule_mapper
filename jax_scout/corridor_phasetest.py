"""
PHASE-kick corridor test — the clean causal routing probe.

Both density-kick variants are confounded: un-matched amplitude kicks inject more energy in the
dense bridge than in a sparse void; energy-matched amplitude kicks force a huge eps in the void
(creating a new blob). A PHASE kick psi -> psi*exp(i*theta*bump) conserves |psi| EXACTLY at
every voxel -> zero energy injected anywhere -> node/bridge/void all receive the IDENTICAL
perturbation, no matching needed. It perturbs the phase/current field, which is exactly the FMIA
information-routing channel. Clean comparison: does a phase twist at the bridge propagate to the
OTHER nodes more than the same twist in a void?

POSITIVE (causal phase-current routing, JAX-tier) = bridge response > void response, resolved
lag, bounded + recovers. NEGATIVE = bridge ~ void (a phase twist anywhere propagates the same)
=> no selective corridor routing.

CAUTION: JAX scout-level; not proof; CuPy still required for any promotion.
WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/corridor_phasetest.py
"""
import os, sys, json, glob, time
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
SETTLE, N_CONT = 800, 120
THETA = 0.4                      # phase-twist amplitude (radians), identical at every location


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


def test(par, N=48):
    pv = [par[k] for k in order]; dx = L/N
    snaps, fin = td.capture_trajectory(pv, multiseed_ic(N, BASE_SEED), N, L, dt, SETTLE, 20)
    if not fin:
        return {"status": "settle_nonfinite"}
    psi0 = snaps[-1]; nodes = td.detect_nodes(psi0, dx)
    if len(nodes) < 2:
        return {"status": "too_few_nodes"}
    nodes = sorted(nodes, key=lambda n: -n["E"])
    cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
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
    others = cents[1:]; omasks = [(_bump(N, c, node_r)[1] <= node_r*node_r) for c in others]

    def cont(p0):
        s, f = td.capture_trajectory(pv, p0, N, L, dt, N_CONT*10, N_CONT); return s, f
    s_ctrl, f0 = cont(psi0)
    if not f0:
        return {"status": "control_nonfinite"}
    T = s_ctrl.shape[0]
    ctrlE = np.array([[_region_E(s_ctrl[t], m) for t in range(T)] for m in omasks])
    res = {"status": "ok", "best_bridge_conductance": float(best), "theta": THETA}
    kicks = {"node": cents[0], "bridge": bridge_pt, "void": void}
    for nm, c in kicks.items():
        s_b, fb = cont(_phase_kick(psi0, c, N, node_r, THETA))
        if not fb:
            res[nm] = {"finite": False}; continue
        bE = np.array([[_region_E(s_b[t], m) for t in range(T)] for m in omasks])
        dev = np.abs(bE-ctrlE)/(ctrlE[:, :1]+1e-30); mdev = dev.mean(0)
        peak = float(mdev.max()); lag = int(np.argmax(mdev))
        res[nm] = {"finite": True, "peak_resp": peak, "lag_steps": lag*10,
                   "recovery_ratio": float(mdev[-1]/(peak+1e-30)), "boundary_pinned": lag >= T-2}
    nk, bk, vk = res.get("node", {}), res.get("bridge", {}), res.get("void", {})
    if nk.get("finite") and bk.get("finite") and vk.get("finite"):
        vp = vk["peak_resp"]+1e-30
        res["ratio_node_vs_void"] = nk["peak_resp"]/vp
        res["ratio_bridge_vs_void"] = bk["peak_resp"]/vp
        res["phase_routing_positive"] = bool(res["ratio_bridge_vs_void"] > 1.5
                                             and not bk["boundary_pinned"] and bk["recovery_ratio"] < 0.85)
    return res


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "BRIDGE_HUNT_2026*")))[-1]
    fz = json.load(open(os.path.join(d, "frozen_finalists.json")))
    out = []
    print(f"PHASE-KICK corridor test (energy-conserving, clean control) theta={THETA}rad — top 3\n")
    for fr in fz["finalists"][:3]:
        par = {k: float(fr["params"][k]) for k in order}
        label = f"gen{fr['generation']}_{fr['config_hash']}"
        t0 = time.time(); r = test(par); r["label"] = label
        print(f"[{label}] bridge_cond={r.get('best_bridge_conductance',0):.3f}")
        for nm in ("node", "bridge", "void"):
            x = r.get(nm, {})
            if x.get("finite"):
                print(f"   {nm:7} peak={x['peak_resp']:.4f} lag={x['lag_steps']}steps "
                      f"recovery={x['recovery_ratio']:.2f} {'BOUNDARY' if x['boundary_pinned'] else 'resolved'}")
        print(f"   -> node/void={r.get('ratio_node_vs_void',float('nan')):.2f} "
              f"bridge/void={r.get('ratio_bridge_vs_void',float('nan')):.2f} "
              f"phase_routing_positive={r.get('phase_routing_positive')}  ({time.time()-t0:.0f}s)\n")
        out.append(r)
    od = os.path.join(d, "corridor_phasetest.json")
    json.dump(out, open(od, "w"), indent=2, default=float)
    npos = sum(1 for r in out if r.get("phase_routing_positive"))
    print(f"wrote {od}")
    print(f"\n=== {npos}/3 show energy-conserving phase-current routing along the bridge ===")
    print("bridge/void ~1 => a phase twist propagates the same everywhere -> NO selective corridor "
          "routing. bridge/void >>1 (resolved+recovery) => clean causal phase-current routing.")


if __name__ == "__main__":
    main()
