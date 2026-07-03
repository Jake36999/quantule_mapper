"""
Energy-MATCHED corridor-perturbation re-test for the top finalists.

The first validation's corridor test (validate_finalists.py) found bridge-kick responses
10-14x larger than a void-kick, BUT the kicks were not energy-matched: a multiplicative kick
in the dense bridge injects more energy than the same kick in a sparse void, so part of that
ratio is injection, not routing. This re-test injects EQUAL energy at every location
(closed-form: for psi -> psi*(1+eps*b), dE = eps^2*sum(rho*b^2) + eps*2*sum(rho*b); solve the
quadratic for eps to hit a common target dE). A clean routing-positive then means: equal energy
injected at the bridge propagates to the OTHER nodes more than equal energy injected in a void,
with a resolved (non-boundary) lag and recovery -- i.e. the corridor actually routes.

CAUTION: JAX scout-level, de-confounding one test only. Not proof; CuPy still required for any
promotion. The top-3 already FAILED high-fidelity (transfer metrics are a finite-window
transient); this isolates whether the *causal routing* along the (robust) bridges is real.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/corridor_retest.py
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
SETTLE, N_CONT = 800, 120          # continue 1200 steps
EPS_REF = 0.06                      # reference kick amplitude (defines the common target dE)


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


def _eps_for_target(rho, bump, target):
    """Solve eps^2*A + eps*B - target = 0 for the kick that injects energy `target` here."""
    A = float(np.sum(rho*bump**2)); B = 2.0*float(np.sum(rho*bump))
    if A <= 0:
        return 0.0
    return float((-B + np.sqrt(max(B*B + 4*A*target, 0.0)))/(2*A))


def _kick_matched(psi, c, N, r, target, sign=1.0):
    bump, _ = _bump(N, c, r); rho = np.abs(psi)**2
    eps = sign*_eps_for_target(rho, bump, abs(target))
    return (psi*(1.0+eps*bump)).astype(np.complex128), eps


def _region_E(psi, mask):
    return float(np.sum(np.abs(psi[mask])**2))


def retest(par, label, N=48):
    pv = [par[k] for k in order]; dx = L/N
    snaps, fin = td.capture_trajectory(pv, multiseed_ic(N, BASE_SEED), N, L, dt, SETTLE, 20)
    if not fin:
        return {"status": "settle_nonfinite"}
    psi0 = snaps[-1]; nodes = td.detect_nodes(psi0, dx)
    if len(nodes) < 2:
        return {"status": "too_few_nodes", "n": len(nodes)}
    nodes = sorted(nodes, key=lambda n: -n["E"])
    cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes])**(1/3))))
    # strongest-conductance bridge pair
    geo = td.geometry_fields(psi0, par, dx); best, bp = -1, (0, 1)
    for i in range(len(cents)):
        for j in range(i+1, len(cents)):
            c = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
            if c > best:
                best, bp = c, (i, j)
    disp = cents[bp[1]]-cents[bp[0]]; disp = disp-N*np.round(disp/N)
    bridge_pt = np.round((cents[bp[0]]+0.5*disp)) % N
    rho = np.abs(psi0)**2
    far = np.ones((N, N, N), bool)
    for c in cents:
        _, d2 = _bump(N, c, 2*node_r); far &= d2 > (2*node_r)**2
    void = np.array(np.unravel_index(np.argmin(np.where(far, rho, rho.max()+1)), rho.shape))
    others = cents[1:]
    omasks = [(_bump(N, c, node_r)[1] <= node_r*node_r) for c in others]

    # common target dE = energy a reference node kick injects
    bn, _ = _bump(N, cents[0], node_r)
    target = EPS_REF*EPS_REF*float(np.sum(rho*bn**2)) + EPS_REF*2*float(np.sum(rho*bn))

    def cont(p0):
        s, f = td.capture_trajectory(pv, p0, N, L, dt, N_CONT*10, N_CONT); return s, f
    s_ctrl, f0 = cont(psi0)
    if not f0:
        return {"status": "control_nonfinite"}
    T = s_ctrl.shape[0]
    ctrlE = np.array([[_region_E(s_ctrl[t], m) for t in range(T)] for m in omasks])

    kicks = {}
    for nm, c, sign in [("node", cents[0], 1.0), ("bridge", bridge_pt, 1.0),
                        ("void", void, 1.0), ("bridge_neg", bridge_pt, -1.0)]:
        pk, eps = _kick_matched(psi0, c, N, node_r, target, sign)
        kicks[nm] = (pk, eps)
    res = {"status": "ok", "best_bridge_conductance": float(best), "target_dE": target,
           "node_r": int(node_r)}
    for nm, (pk, eps) in kicks.items():
        s_b, fb = cont(pk)
        if not fb:
            res[nm] = {"finite": False, "eps": eps}; continue
        bE = np.array([[_region_E(s_b[t], m) for t in range(T)] for m in omasks])
        dev = np.abs(bE-ctrlE)/(ctrlE[:, :1]+1e-30); mdev = dev.mean(0)
        peak = float(mdev.max()); lag = int(np.argmax(mdev))
        res[nm] = {"finite": True, "eps": float(eps), "peak_resp": peak,
                   "lag_steps": lag*10, "recovery_ratio": float(mdev[-1]/(peak+1e-30)),
                   "boundary_pinned": lag >= T-2}
    nk, bk, vk = res.get("node", {}), res.get("bridge", {}), res.get("void", {})
    if nk.get("finite") and bk.get("finite") and vk.get("finite"):
        vp = vk["peak_resp"]+1e-30
        res["ratio_node_vs_void"] = nk["peak_resp"]/vp
        res["ratio_bridge_vs_void"] = bk["peak_resp"]/vp
        res["routing_positive_energymatched"] = bool(
            res["ratio_bridge_vs_void"] > 1.5 and not bk["boundary_pinned"]
            and bk["recovery_ratio"] < 0.85)
    return res


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "BRIDGE_HUNT_2026*")))[-1]
    fz = json.load(open(os.path.join(d, "frozen_finalists.json")))
    top3 = fz["finalists"][:3]
    out = []
    print("ENERGY-MATCHED corridor re-test (de-confounds injection vs routing) — top 3 finalists")
    print("CAUTION: JAX scout-level; isolates causal routing along the robust bridges.\n")
    for fr in top3:
        par = {k: float(fr["params"][k]) for k in order}
        label = f"gen{fr['generation']}_{fr['config_hash']}"
        t0 = time.time(); r = retest(par, label); r["label"] = label
        print(f"[{label}] bridge_cond={r.get('best_bridge_conductance',0):.3f} "
              f"target_dE={r.get('target_dE',0):.3g}")
        for nm in ("node", "bridge", "void", "bridge_neg"):
            x = r.get(nm, {})
            if x.get("finite"):
                print(f"   {nm:11} eps={x['eps']:+.4f} peak={x['peak_resp']:.4f} "
                      f"lag={x['lag_steps']}steps recovery={x['recovery_ratio']:.2f} "
                      f"{'BOUNDARY' if x['boundary_pinned'] else 'resolved'}")
        print(f"   -> node/void={r.get('ratio_node_vs_void',float('nan')):.2f} "
              f"bridge/void={r.get('ratio_bridge_vs_void',float('nan')):.2f} "
              f"routing_positive_energymatched={r.get('routing_positive_energymatched')}  ({time.time()-t0:.0f}s)\n")
        out.append(r)
    od = os.path.join(d, "corridor_retest_energymatched.json")
    json.dump(out, open(od, "w"), indent=2, default=float)
    print(f"wrote {od}")
    print("\nINTERPRETATION: bridge/void ~1 (energy-matched) => earlier 10-14x was INJECTION, not "
          "routing -> causal routing NOT supported. bridge/void >>1 with resolved lag + recovery "
          "=> routing survives energy-matching -> causal routing along the bridge is real (JAX-tier).")


if __name__ == "__main__":
    main()
