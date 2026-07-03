"""
DE-CONFOUND — does conservative anisotropy create BRIDGE-SELECTIVE ROUTING (the real wire
signature), independent of the confounded global_mode-drop metric?

Stage A's global_mode drop was confounded (fixed-point regression + no-bridge control also
dropped). The clean test of a 'wire' is the energy-conserving PHASE-KICK routing test: does a
phase twist at the BRIDGE propagate to the other nodes MORE than the same twist in a VOID, with a
resolved (non-boundary-pinned) lag? And does turning anisotropy ON (lam=0.1) make routing MORE
bridge-selective than lam=0? Run on gen29/gen6 (the 2 strong-bridge 'reproductions').

If anisotropy raises bridge/void selectivity + resolves the response -> real bridge->wire effect
(de-confounded), proceed to Stage B. If bridge ~ void under anisotropy too -> the global_mode drop
was generic, not routing.

CAUTION: JAX scout. conservative dispersive aniso only. Not proof.
WSL2 jax venv:  python jax_scout/afield_aniso_deconfound.py
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
from jax_scout._legacy.afield_anisotropic import capture_aniso
from jax_scout.afield_current_coupled import multiseed_ic, L, order

BASE_SEED = 20260619
SETTLE, CONT, CSNAP = 800, 200, 50    # continue 2000 steps, 50 snaps
THETA = 0.4


def _bump_d2(N, c):
    G = np.meshgrid(*([np.arange(N)]*3), indexing="ij")
    return sum(np.minimum((G[a]-c[a]) % N, (c[a]-G[a]) % N).astype(float)**2 for a in range(3))


def _phase_kick(psi, c, N, r):
    return (psi*np.exp(1j*THETA*np.exp(-_bump_d2(N, c)/(2*(r/1.5)**2)))).astype(np.complex128)


def routing(par, g, kap, cA, lam, N=48):
    dx = L/N
    s0, fin = capture_aniso(par, multiseed_ic(N, BASE_SEED), g, lam, N, SETTLE, 20, kappa=kap, c_A=cA, q_source="stress")
    if not fin:
        return {"status": "settle_nonfinite"}
    psi0 = s0[-1]; nodes = td.detect_nodes(psi0, dx)
    if len(nodes) < 2:
        return {"status": "too_few_nodes"}
    nodes = sorted(nodes, key=lambda n: -n["E"]); cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes])**(1/3))))
    geo = td.geometry_fields(psi0, par, dx); best, bp = -1, (0, 1)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            c = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
            if c > best:
                best, bp = c, (i, j)
    disp = (cents[bp[1]]-cents[bp[0]]).astype(float); disp = disp - N*np.round(disp/N)
    bpt = np.round(cents[bp[0]]+0.5*disp).astype(int) % N
    rho = np.abs(psi0)**2; far = np.ones((N, N, N), bool)
    for c in cents:
        far &= _bump_d2(N, c) > (2*node_r)**2
    void = np.array(np.unravel_index(np.argmin(np.where(far, rho, rho.max()+1)), rho.shape))
    omasks = [(_bump_d2(N, c) <= node_r*node_r) for c in cents[1:]]

    def cont(p0):
        s, f = capture_aniso(par, p0, g, lam, N, CONT, CSNAP, kappa=kap, c_A=cA, q_source="stress"); return s, f
    sc, f0 = cont(psi0)
    if not f0:
        return {"status": "control_nonfinite"}
    T = sc.shape[0]; ctrlE = np.array([[float(np.sum(np.abs(sc[t][m])**2)) for t in range(T)] for m in omasks])

    def resp(loc):
        sb, fb = cont(_phase_kick(psi0, loc, N, node_r))
        if not fb:
            return None
        bE = np.array([[float(np.sum(np.abs(sb[t][m])**2)) for t in range(T)] for m in omasks])
        mdev = (np.abs(bE-ctrlE)/(ctrlE[:, :1]+1e-30)).mean(0)
        return {"peak": float(mdev.max()), "lag": int(np.argmax(mdev)), "boundary": int(np.argmax(mdev)) >= T-2}
    nk = resp(cents[0]); bk = resp(bpt); vk = resp(void)
    out = {"status": "ok", "bridge_cond": float(best), "n_nodes": len(nodes)}
    if nk and bk and vk:
        vp = vk["peak"]+1e-30
        out.update({"node_void": nk["peak"]/vp, "bridge_void": bk["peak"]/vp,
                    "bridge_resolved": (not bk["boundary"]), "bridge_lag_frac": bk["lag"]/(T-1)})
    return out


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "AF_BRIDGE_HUNT_2026*")))[-1]
    sb = json.load(open(os.path.join(d, "afield_aniso_strongbridge.json")))
    repro = [p for p in sb["panel"] if p.get("reproduced")]
    # need full params + A-params: re-pull from CSV by label gen number
    import csv
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    def F(r, k):
        try: return float(r[k])
        except: return float("nan")
    targets = []
    for p in repro:
        gen = p["label"].split("_")[0].replace("gen", "")
        r = next((r for r in rows if r["gen"] == gen and abs(F(r, "bridge")-p["bridge0"]) < 0.01), None)
        if r:
            targets.append(({k: F(r, k) for k in order}, F(r, "gamma_A"), F(r, "kappa"), F(r, "c_A"), p["label"]))
    print(f"DE-CONFOUND: phase-kick bridge-selectivity under anisotropy (lam=0 vs 0.1) on {len(targets)} reproduced configs\n")
    report = []
    for par, g, kap, cA, label in targets:
        print(f"[{label}]")
        row = {"label": label}
        for lam in (0.0, 0.1):
            t0 = time.time(); r = routing(par, g, kap, cA, lam)
            row[f"lam{lam}"] = r
            if r.get("status") == "ok" and "bridge_void" in r:
                print(f"   lam={lam}: bridge/void={r['bridge_void']:.2f} node/void={r['node_void']:.2f} "
                      f"bridge_resolved={r['bridge_resolved']} lag_frac={r['bridge_lag_frac']:.2f} "
                      f"(cond={r['bridge_cond']:.2f})  ({time.time()-t0:.0f}s)")
            else:
                print(f"   lam={lam}: {r.get('status','no_ratio')}  ({time.time()-t0:.0f}s)")
        # de-confound verdict per config
        r0 = row.get("lam0.0", {}); r1 = row.get("lam0.1", {})
        if "bridge_void" in r0 and "bridge_void" in r1:
            improved = (r1["bridge_void"] > 1.5 and r1["bridge_void"] > r0["bridge_void"]*1.2 and r1["bridge_resolved"])
            row["anisotropy_improves_routing"] = bool(improved)
            print(f"   -> anisotropy improves bridge-selective routing: {improved}\n")
        report.append(row)
    json.dump(report, open(os.path.join(d, "afield_aniso_deconfound.json"), "w"), indent=2, default=float)
    nimp = sum(1 for r in report if r.get("anisotropy_improves_routing"))
    print(f"=== {nimp}/{len(report)} show anisotropy IMPROVING bridge-selective routing ===")
    print("If >=1-2: de-confounded REAL bridge->wire effect -> Stage B tensor branch justified. "
          "If 0 (bridge~void under anisotropy too): global_mode drop was generic, NOT routing -> "
          "Stage B on weaker grounds (record the confound).")


if __name__ == "__main__":
    main()
