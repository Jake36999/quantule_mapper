"""
CORRECTED bond test for the feb56dc7 4-node steady state. Phase A's local phase/amplitude kicks only
proved each node heals its OWN local perturbation (nodes never moved -> 'RESTORING' was a weak-test
artifact). With nodes ~56 vox apart and ZERO inter-node corridors, the honest question is INTERACTION
vs PASSIVE COEXISTENCE. Two decisive SPATIAL tests from the steady state:

  (A) DISPLACE one node toward the group centroid (~12 vox): does it get re-attracted (merge),
      repelled (separation restored), or stay put (independent)?
  (B) REMOVE one node (zero its region): do the other 3 reconfigure, does a 4th REGROW (4-node state
      is a genuine attractor of the dynamics), or do they sit unchanged (passive)?

Classes: INTERACTING_ATTRACTIVE / INTERACTING_REPULSIVE / FOUR_NODE_ATTRACTOR / PASSIVE_COEXISTENCE /
FRAGILE. WSL2 jax venv: python jax_scout/feb_bond_test.py
"""
import os, sys, json, glob, time
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import transfer_diag as td
from jax_scout import core_characterize as cc

N, L_, order = cc.N, 10.0, cc.order
PAR = {"param_D": 2.7329, "param_eta": 0.0704, "param_rho_vac": 1.1866, "param_omega0": 0.0,
       "param_a_coupling": 2.3098, "param_s": 0.0129, "param_f": -0.4861, "param_a": 0.4802}
T, NSNAP = 2500, 25


def cents_of(psi, dx):
    return [(np.round(n["centroid"]).astype(int) % N) for n in sorted(td.detect_nodes(psi, dx), key=lambda n: -n["E"])]


def mindist(a, b):
    d = (np.array(a, float) - np.array(b, float)); d -= N * np.round(d / N); return float(np.linalg.norm(d))


def ball(c, r):
    return cc._bump(c) <= r * r


def main():
    dx = L_ / N
    fr = np.load(sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "SUBSTRATE_HUNT_*", "feb56dc7_bound_state", "frames.npz")))[-1])
    psi_s = fr["psi"][-1].astype(np.complex128)            # steady 4-node state (~T=6000)
    c0 = cents_of(psi_s, dx)
    if len(c0) < 4:
        print(f"only {len(c0)} nodes in steady state; abort"); return
    group = np.mean([np.array(c) for c in c0[1:]], 0)      # centroid of nodes 1..3
    r = 4
    print(f"steady: {len(c0)} nodes; node0={c0[0].tolist()} group_centroid={group.round(1).tolist()} "
          f"orig dist(node0,group)={mindist(c0[0], group):.1f}")
    report = {"steady_n_nodes": len(c0), "params": PAR, "tests": {}}

    # (A) displace node0 toward group centroid by ~12 vox
    disp = (group - np.array(c0[0], float)); disp -= N * np.round(disp / N); disp = (disp / (np.linalg.norm(disp) + 1e-9) * 12).round().astype(int)
    m = ball(c0[0], r + 1); patch = psi_s * m; rest = psi_s * (~m)
    psi_disp = (rest + np.roll(patch, tuple(disp), axis=(0, 1, 2))).astype(np.complex128)
    d_orig = mindist(c0[0], group)
    snaps, _ = cc.capture(PAR, psi_disp, T, NSNAP)
    serA = []
    for t in range(snaps.shape[0]):
        if not np.all(np.isfinite(np.abs(snaps[t]))):
            break
        cs = cents_of(snaps[t], dx)
        # node nearest the group is the displaced one; track its dist to group
        dmin = min((mindist(c, group) for c in cs), default=np.nan)
        serA.append({"t_step": t * (T // NSNAP), "n_nodes": len(cs), "min_dist_to_group": dmin})
    dA = serA[-1]["min_dist_to_group"]
    # displaced to ~ d_orig-12; if returns to ~d_orig -> repelled; if ->0 -> attracted/merged; if stays -> independent
    if serA[-1]["n_nodes"] < 4: clsA = "MERGED_OR_LOST"
    elif dA > d_orig - 4: clsA = "REPELLED_restores_separation"
    elif dA < 6: clsA = "ATTRACTED_merged"
    else: clsA = "STAYED_independent"
    report["tests"]["displace"] = {"disp_vox": disp.tolist(), "orig_dist": d_orig, "final_dist": dA,
                                   "klass": clsA, "series": serA}
    print(f"(A) displace: dist(node0,group) {d_orig:.1f} -> kicked ~{d_orig-12:.0f} -> final {dA:.1f} | nodes->{serA[-1]['n_nodes']} -> {clsA}")

    # (B) remove node0 (zero its region)
    psi_rm = (psi_s * (~ball(c0[0], r + 1))).astype(np.complex128)
    snaps, _ = cc.capture(PAR, psi_rm, T, NSNAP)
    serB = []; others0 = c0[1:]
    for t in range(snaps.shape[0]):
        if not np.all(np.isfinite(np.abs(snaps[t]))):
            break
        cs = cents_of(snaps[t], dx)
        shift = np.mean([min((mindist(c, o) for c in cs), default=0.0) for o in others0])  # how far others moved
        serB.append({"t_step": t * (T // NSNAP), "n_nodes": len(cs), "others_drift": float(shift)})
    nfB = serB[-1]["n_nodes"]; driftB = serB[-1]["others_drift"]
    if nfB >= 4: clsB = "FOUR_NODE_REGREW_attractor"
    elif nfB == 3 and driftB < 5: clsB = "PASSIVE_others_unchanged"
    elif nfB == 3: clsB = "RECONFIGURED_others_moved"
    else: clsB = "COLLAPSED_or_grew"
    report["tests"]["remove"] = {"final_n_nodes": nfB, "others_drift": driftB, "klass": clsB, "series": serB}
    print(f"(B) remove node0: nodes 4->3 then -> {nfB} (others_drift {driftB:.1f}) -> {clsB}")

    # overall honest verdict
    interacting = ("REPELLED" in clsA) or ("ATTRACTED" in clsA) or (clsB == "FOUR_NODE_REGREW_attractor") or (clsB == "RECONFIGURED_others_moved")
    verdict = ("FOUR_NODE_ATTRACTOR" if clsB == "FOUR_NODE_REGREW_attractor" else
               "INTERACTING_REPULSIVE" if "REPELLED" in clsA else
               "INTERACTING_ATTRACTIVE" if "ATTRACTED" in clsA else
               "PASSIVE_COEXISTENCE" if (clsA == "STAYED_independent" and clsB.startswith("PASSIVE")) else
               "MIXED_SEE_TESTS")
    report["bond_verdict_corrected"] = verdict
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "SUBSTRATE_HUNT_*", "feb56dc7_bound_state")))[-1]
    json.dump(report, open(os.path.join(d, "feb_bond_test.json"), "w"), indent=2, default=float)
    print(f"\nCORRECTED BOND VERDICT: {verdict}\nwrote {os.path.join(d,'feb_bond_test.json')}")


if __name__ == "__main__":
    main()
