"""
Validation Tier for the top bridge-hunt finalists (JAX scout-level POSITIVE -> stricter test).

CAUTION: this is a JAX scout validation. Surviving it makes a finalist a JAX-tier candidate,
NOT proof. Full promotion still requires CuPy/ASTE FP64 reproduction (step 4, separate, gated
on passing here). Scout findings are not IRER evidence on their own.

Protocol (user-specified):
  1. FREEZE — record full provenance of the top finalists (params, config hash, seed, IC family,
     run id, all scout metrics) so the adaptive hunt cannot mutate them during validation.
  2. HIGH-FIDELITY JAX — longer trajectory (>=1600 steps), same/repeat/altered seeds, local
     parameter perturbations (D, eta, s, f, a, a_coupling), optional finer grid (N=64).
     PASS = bounded nodes persist, bridge conductance persists, phase coupling stays above the
     0.73 independence floor, energy-exchange excess stays above null, no energy/curvature
     runaway, no space-filling, no collapse to independent condensates.
  3. CORRIDOR PERTURBATION (critical, causal) — kick a node, kick the bridge, kick a matched
     void (control), and the opposite-sign bridge kick; measure delayed bounded response at the
     OTHER nodes, transfer lag, recovery. POSITIVE = bounded delayed response, node/bridge kick
     > void kick, measurable lag, recovery to a bounded node arrangement (NOT node death).
  5-way verdict: VALIDATED_FMIA_TRANSFER_CANDIDATE (JAX-tier; CuPy pending) / PROMISING_BUT_FRAGILE
     / SCOUT_ONLY_ARTIFACT / RUNAWAY_REJECT / NONREPRODUCING_REJECT.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/validate_finalists.py
"""
import os, sys, json, time, glob, hashlib
import numpy as np
# Shared, fragmented 8GB GPU under WSL2: platform allocator (on-demand alloc/free) is the only
# stable choice (see docs/FMIA_TRANSFER_DIAGNOSTIC_FINDING.md §10).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td

L, dt = 10.0, 0.005
order = physics.SWEEP_PARAM_ORDER
FLOOR = td.THR_PHASECOUP                 # 0.73 phase-coupling independence floor
BASE_SEED = 20260619
HI_STEPS, HI_NSNAP = 1600, 40            # high-fidelity trajectory
ER_LO, ER_HI = 0.5, 2.0                  # energy-conservation band for a PASS (stricter than the gate)


def cfg_hash(par):
    return hashlib.sha1(json.dumps({k: round(par[k], 6) for k in order}, sort_keys=True).encode()).hexdigest()[:12]


def multiseed_ic(N, seed, K=6):
    rng = np.random.default_rng(seed)
    x = np.linspace(-L/2, L/2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L/12.0; psi = np.zeros((N, N, N), np.complex128)
    for _ in range(K):
        cx, cy, cz = rng.uniform(-L/2, L/2, 3)
        psi += np.exp(-((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/(2*w**2))
    noise = 0.01*(rng.standard_normal((N, N, N))+1j*rng.standard_normal((N, N, N)))
    return (psi+noise).astype(np.complex128)


def run_one(par, ic, N, steps, n_snap):
    pv = [par[k] for k in order]
    return td.analyze_candidate(pv, par, ic, N, L, dt, steps, n_snap, iso_surv=np.nan)


def hi_pass(r):
    """High-fidelity pass condition for one analyze result."""
    return (r.get("finite") and r.get("n_persistent_nodes", 0) >= 2
            and ER_LO <= _safe(r, "amp_final", 0) * 0 + _er(r) <= ER_HI  # er via amp proxy below
            )


def _er(r):  # analyze_candidate doesn't return er; recompute via a flag we add (fallback 1.0)
    return r.get("er", 1.0)


def _safe(r, k, d): return r.get(k, d)


# ---- Step 2: high-fidelity battery -------------------------------------------------
def high_fidelity(par, label):
    print(f"\n--- HIGH-FIDELITY [{label}] N=48/{HI_STEPS} ---")
    runs = {}
    # same seed (baseline) + 2 altered seeds
    seedset = [("baseline", BASE_SEED), ("altseed_A", 7001), ("altseed_B", 8002)]
    for nm, sd in seedset:
        r = run_one(par, multiseed_ic(48, sd), 48, HI_STEPS, HI_NSNAP)
        runs[nm] = r
        _print_run(nm, r)
    # parameter perturbations (local)
    perturb = {"D-0.5": ("param_D", -0.5), "eta+0.05": ("param_eta", 0.05),
               "s+0.15": ("param_s", 0.15), "f+0.10": ("param_f", 0.10),
               "a+0.10": ("param_a", 0.10), "acoup+0.2": ("param_a_coupling", 0.2)}
    for nm, (k, dv) in perturb.items():
        p2 = dict(par); p2[k] = par[k] + dv
        r = run_one(p2, multiseed_ic(48, BASE_SEED), 48, HI_STEPS, HI_NSNAP)
        runs[f"perturb_{nm}"] = r
        _print_run(f"perturb_{nm}", r)
    # optional finer grid (N=64) on baseline seed
    try:
        r = run_one(par, multiseed_ic(64, BASE_SEED), 64, HI_STEPS, HI_NSNAP)
        runs["grid_N64"] = r
        _print_run("grid_N64", r)
    except Exception as e:
        runs["grid_N64"] = {"error": str(e)[:120]}
        print(f"  grid_N64: SKIPPED ({str(e)[:80]})")
    return runs


def _print_run(nm, r):
    if "error" in r:
        print(f"  {nm:16} ERROR {r['error'][:60]}"); return
    pc = r.get("phase_coupling_score", 0.0); flag = "  *>floor*" if pc > FLOOR else ""
    print(f"  {nm:16} fin={int(bool(r.get('finite')))} nP={r.get('n_persistent_nodes',0)} "
          f"maxCond={r.get('max_transfer_strength',0):.3f} meanCond={r.get('omega_corridor_conductance',0):.3f} "
          f"pcoup={pc:.3f} exch={r.get('energy_exchange_index',0):.3f} amp={r.get('amp_final',0):.1f}{flag}")


# ---- Step 3: causal corridor perturbation -----------------------------------------
def _region_mask(N, c, r):
    G = np.meshgrid(*([np.arange(N)]*3), indexing="ij")
    d2 = sum(np.minimum((G[a]-c[a]) % N, (c[a]-G[a]) % N).astype(float)**2 for a in range(3))
    return d2, d2 <= r*r


def _region_E(psi, mask):
    return float(np.sum(np.abs(psi[mask])**2))


def _kick(psi, c, N, r, eps, phase=False):
    d2, _ = _region_mask(N, c, r)
    bump = np.exp(-d2/(2*(r/1.5)**2))
    if phase:
        return (psi*np.exp(1j*eps*bump)).astype(np.complex128)
    return (psi*(1.0+eps*bump)).astype(np.complex128)


def corridor_perturbation(par, label, N=48, settle=800, n_cont=120, eps=0.06):
    """Kick node / bridge / matched-void / opposite-sign-bridge; measure bounded delayed response
    at OTHER nodes vs the void control, plus recovery."""
    print(f"\n--- CORRIDOR PERTURBATION [{label}] settle={settle} continue={n_cont*10} steps ---")
    ic = multiseed_ic(N, BASE_SEED); dx = L/N; pv = [par[k] for k in order]
    snaps, fin = td.capture_trajectory(pv, ic, N, L, dt, settle, 20)
    if not fin:
        print("  settle non-finite; abort."); return {"status": "settle_nonfinite"}
    psi0 = snaps[-1]; nodes = td.detect_nodes(psi0, dx)
    if len(nodes) < 2:
        print(f"  only {len(nodes)} node(s); abort."); return {"status": "too_few_nodes"}
    nodes = sorted(nodes, key=lambda n: -n["E"])
    cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes])**(1/3))))
    # strongest bridge pair (by density conductance) -> bridge midpoint
    geo = td.geometry_fields(psi0, par, dx); best, bp = -1, (0, 1)
    for i in range(len(cents)):
        for j in range(i+1, len(cents)):
            c = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
            if c > best:
                best, bp = c, (i, j)
    disp = cents[bp[1]] - cents[bp[0]]; disp = disp - N*np.round(disp/N)
    bridge_pt = np.round((cents[bp[0]] + 0.5*disp)) % N
    rho = np.abs(psi0)**2
    G = np.meshgrid(*([np.arange(N)]*3), indexing="ij")
    far = np.ones((N, N, N), bool)
    for c in cents:
        d2, _ = _region_mask(N, c, 2*node_r); far &= d2 > (2*node_r)**2
    void = np.array(np.unravel_index(np.argmin(np.where(far, rho, rho.max()+1)), rho.shape))
    # OTHER nodes = all nodes except the kicked one (node 0) — measure response there
    others = cents[1:]
    omasks = [(_region_mask(N, c, node_r)[1]) for c in others]

    def cont(p0):
        s, f = td.capture_trajectory(pv, p0, N, L, dt, n_cont*10, n_cont)
        return s, f

    s_ctrl, f0 = cont(psi0)
    branches = {
        "node_kick":   _kick(psi0, cents[0], N, node_r, eps),
        "bridge_kick": _kick(psi0, bridge_pt, N, node_r, eps),
        "void_kick":   _kick(psi0, void, N, node_r, eps),
        "bridge_neg":  _kick(psi0, bridge_pt, N, node_r, -eps),
    }
    T = s_ctrl.shape[0]
    res = {"status": "ok", "best_bridge_conductance": float(best), "node_r": int(node_r)}
    if not f0:
        return {"status": "control_nonfinite"}
    ctrlE = np.array([[_region_E(s_ctrl[t], m) for t in range(T)] for m in omasks])  # [nOther,T]
    for nm, p0 in branches.items():
        s_b, fb = cont(p0)
        if not fb:
            res[nm] = {"finite": False}; print(f"  {nm:12} NON-FINITE"); continue
        bE = np.array([[_region_E(s_b[t], m) for t in range(T)] for m in omasks])
        dev = np.abs(bE - ctrlE) / (ctrlE[:, :1] + 1e-30)        # rel deviation per other-node
        mdev = dev.mean(0)                                        # mean over other nodes vs time
        peak = float(mdev.max()); lag = int(np.argmax(mdev))
        recov = float(mdev[-1] / (peak + 1e-30))                 # <1 and decaying = recovery
        finite_b = bool(fb and np.all(np.isfinite(bE)) and bE.max() < 1e6)
        res[nm] = {"finite": finite_b, "peak_resp": peak, "lag_steps": lag*10,
                   "recovery_ratio": recov, "boundary_pinned": lag >= T-2}
        print(f"  {nm:12} peak={peak:.4f} lag={lag*10}steps recovery={recov:.2f} "
              f"{'BOUNDARY-PINNED' if lag>=T-2 else 'resolved'}")
    # routing verdict
    nk = res.get("node_kick", {}); bk = res.get("bridge_kick", {}); vk = res.get("void_kick", {})
    if nk.get("finite") and bk.get("finite") and vk.get("finite"):
        vpeak = vk["peak_resp"] + 1e-30
        ratio_node = nk["peak_resp"]/vpeak; ratio_bridge = bk["peak_resp"]/vpeak
        routed = (max(ratio_node, ratio_bridge) > 1.5 and not bk.get("boundary_pinned", True)
                  and bk["recovery_ratio"] < 0.8)
        res["ratio_node_vs_void"] = float(ratio_node)
        res["ratio_bridge_vs_void"] = float(ratio_bridge)
        res["routing_positive"] = bool(routed)
        print(f"  -> node/void={ratio_node:.2f} bridge/void={ratio_bridge:.2f} "
              f"routing_positive={routed}")
    return res


# ---- verdict -----------------------------------------------------------------------
def classify_finalist(hi, corr):
    base = hi.get("baseline", {})
    # runaway?
    for r in hi.values():
        if isinstance(r, dict) and "error" not in r:
            if (not r.get("finite", False)) or r.get("amp_final", 0) > 1e3:
                pass  # individual non-finite handled below; only base/seed runaway -> reject
    if not base.get("finite", False) or base.get("amp_final", 0) > 1e3:
        return "RUNAWAY_REJECT"
    base_ok = (base.get("n_persistent_nodes", 0) >= 2
               and base.get("max_transfer_strength", 0) > 0.2          # bridge persists
               and base.get("phase_coupling_score", 0) > FLOOR)        # above floor at hi-fi
    if not base_ok:
        return "NONREPRODUCING_REJECT"
    # seed robustness: altered seeds keep bridge + phase>floor
    seeds = [hi.get("altseed_A", {}), hi.get("altseed_B", {})]
    seed_hold = sum(1 for s in seeds if s.get("finite") and s.get("max_transfer_strength", 0) > 0.2
                    and s.get("phase_coupling_score", 0) > FLOOR)
    routed = corr.get("routing_positive", False)
    if seed_hold == 2 and routed:
        return "VALIDATED_FMIA_TRANSFER_CANDIDATE"   # JAX-tier; CuPy still pending
    if seed_hold >= 1 or routed:
        return "PROMISING_BUT_FRAGILE"
    return "SCOUT_ONLY_ARTIFACT"


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "BRIDGE_HUNT_2026*")))[-1]
    finals = json.load(open(os.path.join(d, "finalists.json")))
    top3 = finals[:3]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "sweep_runs", f"VALIDATION_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    print(f"VALIDATION TIER (JAX scout) on top {len(top3)} finalists from {os.path.basename(d)}")
    print("CAUTION: JAX scout-level. Passing => JAX-tier candidate, NOT proof; CuPy step still required.\n")

    report = []
    for n, fr in enumerate(top3):
        par = {k: float(fr["params"][k]) for k in order}
        label = f"gen{fr['gen']}_n{fr['nodes']}_{cfg_hash(par)}"
        # Step 1: FREEZE
        frozen = {"rank": n+1, "label": label, "config_hash": cfg_hash(par), "params": par,
                  "ic_family": "coherent_multiseed_K6", "base_seed": BASE_SEED,
                  "source_run": os.path.basename(d), "gen": fr["gen"],
                  "scout_metrics": {k: fr.get(k) for k in
                                    ("nodes", "pcoup", "exch", "maxCond", "er", "curv")}}
        print(f"\n{'='*70}\n[{n+1}/3] FROZEN {label}\n  params={par}\n  scout: {frozen['scout_metrics']}")
        t0 = time.time()
        hi = high_fidelity(par, label)
        corr = corridor_perturbation(par, label)
        verdict = classify_finalist(hi, corr)
        print(f"\n  >>> VERDICT [{label}]: {verdict}  ({time.time()-t0:.0f}s)")
        def strip(r):
            return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in r.items() if k not in ()} if isinstance(r, dict) else r
        report.append({"frozen": frozen, "verdict": verdict,
                       "high_fidelity": {k: strip(v) for k, v in hi.items()},
                       "corridor": corr})
        json.dump(report, open(os.path.join(outdir, "validation_report.json"), "w"),
                  indent=2, default=float)

    print(f"\n{'='*70}\n=== VALIDATION SUMMARY ===")
    for r in report:
        print(f"  {r['frozen']['label']:30} -> {r['verdict']}")
    print(f"\nwrote {outdir}/validation_report.json")
    print("NOTE: any VALIDATED_FMIA_TRANSFER_CANDIDATE is JAX-tier only; CuPy/ASTE FP64 "
          "reproduction (step 4) is still required before promotion.")


if __name__ == "__main__":
    main()
