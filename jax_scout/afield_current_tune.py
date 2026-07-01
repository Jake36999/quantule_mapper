"""
STAGE 1 (bounded) — focused current-coupled A tuning on the gamma_A=0 finalists (calibration
probes, NOT proof targets).

Maps whether the current-coupled A-field has a CLEAN operating regime between too-weak (no effect)
and too-strong (bridge saturates, exchange collapses, energy grows = distortion, not wires).
Sweeps gamma_A x kappa(source) x c_A(speed) on gen18. The headline is NOT aggregate phase
coupling but whether the coupling STRUCTURE shifts from global web -> pairwise/corridor:
global_mode_fraction (down), pairwise_fraction (up), node_bridge_selectivity (up), A localized on
the bridge (not saturating the web).

DISTORTION HARD-REJECT (a "rise" via any of these does NOT count):
  energy runaway (er outside [0.5,2]), curvature runaway (>1), bridge conductance saturating
  (>0.9 via space-filling), energy-exchange collapse (<0.01), A-energy ballooning.

Phase A: broad cheap map on gen18 (phase coupling @800 & @1600, exch, bridge, er, curv, A-energy,
         A-bridge-localization + guards). Phase B: for the best CLEAN combo (if any), the
         global_mode response-matrix UNDER A-ON vs gamma_A=0. If no clean regime ->
         CURRENT_A_V1_CALIBRATION_NO_RESCUE_ON_GAMMA0_FINALISTS (NOT a theory failure).

CAUTION: JAX scout, ACTIVE branch, default-off elsewhere, contract-stamped, segregated from
gamma_A=0 rankings. Not proof.
WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/afield_current_tune.py
"""
import os, sys, json, glob, time
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td, geometry_diag as gd
from jax_scout.afield_current_coupled import capture_cc, analyze, multiseed_ic, L, dt, order, FLOOR

BASE_SEED = 20260619
STEPS, NSNAP = 1600, 40
GAMMAS = [0.1, 0.2]
KAPPAS = [1.0, 4.0, 16.0]
C_AS = [0.5, 1.0, 2.0]
# distortion guards
ER_LO, ER_HI, CURV_MAX, BRIDGE_SAT, EXCH_MIN = 0.5, 2.0, 1.0, 0.9, 0.01


def _bump_d2(N, c):
    G = np.meshgrid(*([np.arange(N)]*3), indexing="ij")
    return sum(np.minimum((G[a]-c[a]) % N, (c[a]-G[a]) % N).astype(float)**2 for a in range(3))


def a_bridge_localization(Asq, psi_final, par, N):
    dx = L/N; nodes = td.detect_nodes(psi_final, dx)
    if len(nodes) < 2:
        return float("nan")
    nodes = sorted(nodes, key=lambda n: -n["E"]); cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    geo = td.geometry_fields(psi_final, par, dx); best, bp = -1, (0, 1)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            c = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
            if c > best:
                best, bp = c, (i, j)
    disp = (cents[bp[1]]-cents[bp[0]]).astype(float); disp = disp - N*np.round(disp/N)
    bpt = np.round(cents[bp[0]]+0.5*disp).astype(int) % N
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes])**(1/3))))
    mask = _bump_d2(N, bpt) <= node_r*node_r
    A_in = float(Asq[mask].sum()); A_tot = float(Asq.sum())+1e-30
    vol = mask.sum()/Asq.size
    return (A_in/A_tot)/(vol+1e-30)


def cheap_eval(par, g, kap, cA, N=48):
    ic = multiseed_ic(N, BASE_SEED)
    snaps, Asnaps, fin = capture_cc(par, ic, g, N, STEPS, NSNAP, kappa=kap, c_A=cA)
    if not fin:
        return {"finite": False, "klass": "A_CURRENT_RUNAWAY_REJECT"}
    er = float(np.sum(np.abs(snaps[-1])**2)/(np.sum(np.abs(snaps[0])**2)+1e-30))
    curv = gd.curvature_max_only(snaps[-1], par, L/N)
    a_full = analyze(snaps, par, N)
    a_half = analyze(snaps[:NSNAP//2+1], par, N)
    A_E = float(np.sum(Asnaps[-1])); A_loc = a_bridge_localization(Asnaps[-1], snaps[-1], par, N)
    distort = (er > ER_HI or er < ER_LO or curv > CURV_MAX or a_full["max_cond"] > BRIDGE_SAT
               or a_full["energy_exchange_index"] < EXCH_MIN or A_E > 1e3)
    return {"finite": True, "gamma_A": g, "kappa": kap, "c_A": cA, "er": er, "curv": float(curv),
            "pcoup_800": a_half["phase_coupling_score"], "pcoup_1600": a_full["phase_coupling_score"],
            "exch": a_full["energy_exchange_index"], "bridge": a_full["max_cond"],
            "nP": a_full["n_persistent_nodes"], "A_energy": A_E, "A_bridge_loc": A_loc,
            "distort": bool(distort)}


# ---- global_mode response matrix UNDER A-on (Phase B) ----
def _phase_kick(psi, c, N, r, theta=0.4):
    b = np.exp(-_bump_d2(N, c)/(2*(r/1.5)**2)); return (psi*np.exp(1j*theta*b)).astype(np.complex128)


def global_mode_under_A(par, g, kap, cA, N=48, settle=800, cont=2000, csnap=50):
    ic = multiseed_ic(N, BASE_SEED); dx = L/N
    s0, _, fin = capture_cc(par, ic, g, N, settle, 20, kappa=kap, c_A=cA)
    if not fin:
        return {"status": "settle_nonfinite"}
    psi0 = s0[-1]; nodes = td.detect_nodes(psi0, dx)
    if len(nodes) < 2:
        return {"status": "too_few_nodes"}
    nodes = sorted(nodes, key=lambda n: -n["E"]); nn = len(nodes)
    cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes])**(1/3))))
    masks = [(_bump_d2(N, c) <= node_r*node_r) for c in cents]

    def cont_cc(p0):
        s, _, f = capture_cc(par, p0, g, N, cont, csnap, kappa=kap, c_A=cA); return s, f
    sc, f0 = cont_cc(psi0)
    if not f0:
        return {"status": "control_nonfinite"}
    T = sc.shape[0]; ctrlE = np.array([[float(np.sum(np.abs(sc[t][m])**2)) for t in range(T)] for m in masks])
    M = np.full((nn, nn), np.nan)
    for i in range(nn):
        sb, fb = cont_cc(_phase_kick(psi0, cents[i], N, node_r))
        if not fb:
            continue
        bE = np.array([[float(np.sum(np.abs(sb[t][m])**2)) for t in range(T)] for m in masks])
        peaks = (np.abs(bE-ctrlE)/(ctrlE[:, :1]+1e-30)).max(1)
        for j in range(nn):
            if j != i:
                M[i, j] = peaks[j]
    Mf = np.nan_to_num(M, nan=0.0)
    if not np.any(Mf):
        return {"status": "no_response", "n_nodes": nn}
    sv = np.linalg.svd(Mf, compute_uv=False); tot = float(np.sum(sv**2))
    gmf = float(sv[0]**2/tot) if tot > 0 else float("nan")
    return {"status": "ok", "n_nodes": nn, "global_mode_fraction": gmf,
            "pairwise_fraction": 1.0-gmf, "structure_response_gain": float(np.mean(Mf[Mf > 0])) if np.any(Mf > 0) else 0.0}


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "BRIDGE_HUNT_2026*")))[-1]
    fz = json.load(open(os.path.join(d, "frozen_finalists.json")))
    g18 = fz["finalists"][0]; par = {k: float(g18["params"][k]) for k in order}
    label = f"gen{g18['generation']}_{g18['config_hash']}"
    print(f"STAGE 1 current-A tuning on {label} (calibration probe)")
    print(f"grid gamma_A {GAMMAS} x kappa {KAPPAS} x c_A {C_AS}; guards er[{ER_LO},{ER_HI}] curv<{CURV_MAX} "
          f"bridge<{BRIDGE_SAT} exch>{EXCH_MIN}\n")

    # baseline gamma_A=0
    base = cheap_eval(par, 0.0, 1.0, 1.0)
    print(f"  BASELINE gA=0: pcoup@1600={base['pcoup_1600']:.3f} exch={base['exch']:.3f} "
          f"bridge={base['bridge']:.3f} er={base['er']:.2f}\n")
    rows = [base]
    for g in GAMMAS:
        for kap in KAPPAS:
            for cA in C_AS:
                t0 = time.time(); r = cheap_eval(par, g, kap, cA); rows.append(r)
                if r["finite"]:
                    clean = (not r["distort"]) and r["pcoup_1600"] > base["pcoup_1600"] + 0.02
                    print(f"  gA={g} kap={kap:<4} cA={cA}: pc@800={r['pcoup_800']:.3f} pc@1600={r['pcoup_1600']:.3f} "
                          f"exch={r['exch']:.3f} bridge={r['bridge']:.3f} er={r['er']:.2f} curv={r['curv']:.2f} "
                          f"A_E={r['A_energy']:.2g} A_loc={r['A_bridge_loc']:.2f} "
                          f"{'DISTORT' if r['distort'] else ('CLEAN-RISE' if clean else 'flat/down')}  ({time.time()-t0:.0f}s)")
                else:
                    print(f"  gA={g} kap={kap} cA={cA}: NON-FINITE/REJECT  ({time.time()-t0:.0f}s)")
    # pick best CLEAN combo (non-distort, biggest pcoup_1600 rise over baseline)
    clean = [r for r in rows if r.get("finite") and not r.get("distort") and r.get("gamma_A", 0) > 0
             and r["pcoup_1600"] > base["pcoup_1600"] + 0.02]
    clean.sort(key=lambda r: -r["pcoup_1600"])
    out = {"label": label, "baseline": base, "grid": rows}
    if clean:
        best = clean[0]
        print(f"\nBest CLEAN combo: gA={best['gamma_A']} kap={best['kappa']} cA={best['c_A']} "
              f"pcoup@1600={best['pcoup_1600']:.3f} (baseline {base['pcoup_1600']:.3f})")
        print("Phase B: global_mode_fraction UNDER A-on vs gamma_A=0 ...")
        gm0 = global_mode_under_A(par, 0.0, 1.0, 1.0)
        gmA = global_mode_under_A(par, best["gamma_A"], best["kappa"], best["c_A"])
        out["phaseB"] = {"gamma0": gm0, "best": gmA, "combo": best}
        print(f"  gamma_A=0:   global_mode_fraction={gm0.get('global_mode_fraction')}")
        print(f"  best combo:  global_mode_fraction={gmA.get('global_mode_fraction')}")
        if gm0.get("global_mode_fraction") and gmA.get("global_mode_fraction"):
            drop = gm0["global_mode_fraction"] - gmA["global_mode_fraction"]
            print(f"  -> global_mode_fraction change = {-drop:+.3f} "
                  f"({'WEB->WIRES shift' if drop > 0.05 else 'no structural shift'})")
        out["verdict"] = ("A_CURRENT_STRUCTURE_SHIFT" if (gm0.get("global_mode_fraction", 1) -
                          gmA.get("global_mode_fraction", 1)) > 0.05 else "A_CURRENT_PARTIAL_NO_STRUCTURE_SHIFT")
    else:
        out["verdict"] = "CURRENT_A_V1_CALIBRATION_NO_RESCUE_ON_GAMMA0_FINALISTS"
        print("\nNo clean (non-distorting) phase-coupling rise in the tested grid.")
        print("VERDICT: CURRENT_A_V1_CALIBRATION_NO_RESCUE_ON_GAMMA0_FINALISTS "
              "(NOT a theory failure; old finalists likely the wrong substrate -> Stage 2 A-coupled hunt).")
    od = os.path.join(d, "afield_current_tune.json")
    json.dump(out, open(od, "w"), indent=2, default=float)
    print(f"\nwrote {od}")


if __name__ == "__main__":
    main()
