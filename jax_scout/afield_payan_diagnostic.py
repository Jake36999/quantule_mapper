"""
PASSIVE Payan diagnostic (NO solver coupling) — does Payan alignment correlate with bridge stability?

Stage 1 of docs/PAYAN_PHASE_ALIGNMENT_RFC.md, per docs/PAYAN_DERIVATION_PASSIVE_DIAGNOSTIC.md
(PAYAN_DIAGNOSTIC_DEFINED). Payan state = "spin along axis" = axial circulation/winding of grad-phi
around a node (angular deficit). Computed PASSIVELY from the existing psi=sqrt(rho)e^{i phi}:
  v = J/rho   (J=rho grad-phi = Im(conj psi grad psi), from transfer_diag.geometry_fields)
  omega = curl(v)                                  (vorticity = density of topological charge)
  s_k = sum_{ball around node k} (omega . a_ij)    (axial Payan spin along the bridge axis a_ij)
  A_ij = (s_i s_j)/(|s_i||s_j|)  in [-1,1]         (Payan alignment: +1 co-handed, -1 frustrated)
  R_ij = | sum_corridor rho e^{i phi} | / sum rho   (corridor phase-coherence / phase-lock)

FALSIFIABLE TEST on the existing 576-eval substrate population (re-settle to 800; no new long run):
do STABLE substrates (accepted) have higher Payan alignment / coherence than UNSTABLE ones
(energy_drift that formed structure at settle then blew up)? Must BEAT the density-preserved
phase-randomized control (same rho, random phi) -- if random phase reproduces the separation, the
signal is not Payan.

Verdict: PASSIVE_PAYAN_ALIGNMENT_CORRELATES_WITH_STABILITY (-> PAYAN_COUPLING_RFC_READY) or
PASSIVE_PAYAN_ALIGNMENT_NO_SIGNAL (-> PAYAN_COUPLING_NOT_JUSTIFIED). Earned claim only:
"geometry-only routing failed -> Payan is the next justified hypothesis" (NOT proven).

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/afield_payan_diagnostic.py --n 25
"""
import os, sys, csv, json, glob, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td
from jax_scout import afield_current_coupled as cc
from jax_scout.afield_current_coupled import multiseed_ic, L

order = physics.SWEEP_PARAM_ORDER
SEED = 20260619
SETTLE = 800
RHO_FLOOR = 1e-6


def _curl(vx, vy, vz, dx):
    gx = td._grad(vx, dx); gy = td._grad(vy, dx); gz = td._grad(vz, dx)
    # _grad -> (d/dx, d/dy, d/dz)
    wx = gz[1] - gy[2]   # dvz/dy - dvy/dz
    wy = gx[2] - gz[0]   # dvx/dz - dvz/dx
    wz = gy[0] - gx[1]   # dvy/dx - dvx/dz
    return wx, wy, wz


def _ball(N, c, r):
    G = np.meshgrid(*([np.arange(N)] * 3), indexing="ij")
    d2 = sum(np.minimum((G[a] - c[a]) % N, (c[a] - G[a]) % N).astype(float) ** 2 for a in range(3))
    return d2 <= r * r


def payan_observables(psi, par, N):
    """Returns dict with bridge-pair Payan alignment A, corridor coherence R, bridge conductance,
    node spins, n_nodes -- or None if <2 nodes."""
    dx = L / N
    nodes = td.detect_nodes(psi, dx)
    if len(nodes) < 2:
        return None
    nodes = sorted(nodes, key=lambda n: -n["E"])
    geo = td.geometry_fields(psi, par, dx)
    rho = geo["rho"]; Jx, Jy, Jz = geo["J"]
    rs = np.maximum(rho, RHO_FLOOR)
    wx, wy, wz = _curl(Jx / rs, Jy / rs, Jz / rs, dx)
    # strongest-conductance pair = the bridge
    best, bp = -1.0, (0, 1)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            c = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
            if c > best:
                best, bp = c, (i, j)
    ci = np.round(nodes[bp[0]]["centroid"]).astype(int) % N
    cj = np.round(nodes[bp[1]]["centroid"]).astype(int) % N
    disp = (cj - ci).astype(float); disp -= N * np.round(disp / N)
    ax = disp / (np.linalg.norm(disp) + 1e-30)
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes]) ** (1 / 3))))
    def spin(c):
        m = _ball(N, c, node_r)
        return float(np.sum(wx[m] * ax[0] + wy[m] * ax[1] + wz[m] * ax[2]))
    s_i, s_j = spin(ci), spin(cj)
    A = float((s_i * s_j) / (abs(s_i) * abs(s_j) + 1e-30))   # ~ sign product in {-1,+1} (chirality match)
    # corridor coherence (density-weighted Kuramoto order parameter along the bridge, node interiors excluded)
    phi = np.angle(psi)
    p_line, _ = td._sample_line(phi, nodes[bp[0]]["centroid"], nodes[bp[1]]["centroid"], N, 48)
    r_line, _ = td._sample_line(rho, nodes[bp[0]]["centroid"], nodes[bp[1]]["centroid"], N, 48)
    w = r_line[6:-6]; p = p_line[6:-6]
    R = float(np.abs(np.sum(w * np.exp(1j * p)) / (np.sum(w) + 1e-30)))
    return {"A": A, "R": R, "aligned": int(A > 0), "spin_mag": float(abs(s_i) + abs(s_j)),
            "bridge_cond": float(best), "s_i": s_i, "s_j": s_j, "n_nodes": len(nodes)}


def phase_randomized(psi, rng):
    """Density-preserved phase-randomized control: same rho, i.i.d. uniform phase."""
    rho = np.abs(psi) ** 2
    return (np.sqrt(rho) * np.exp(1j * rng.uniform(0, 2 * np.pi, psi.shape))).astype(np.complex128)


def settle(par, g, kap, cA, N):
    s0, _, fin = cc.capture_cc(par, multiseed_ic(N, SEED), g, N, SETTLE, 20, kappa=kap, c_A=cA)
    return s0[-1] if fin else None


def F(r, k):
    try: return float(r[k])
    except: return float("nan")


def select_population(d, n_each):
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    stable = sorted([r for r in rows if r["reject"] == "" and 2 <= F(r, "n_s") <= 8],
                    key=lambda r: -F(r, "bridge_s"))
    # unstable = formed structure at settle (er_s bounded, >=2 nodes) but drifted out by horizon
    unstable = sorted([r for r in rows if r["reject"] == "energy_drift"
                       and 0.5 <= F(r, "er_s") <= 2.0 and F(r, "n_s") >= 2
                       and not (0.5 <= F(r, "er_e") <= 2.0)],
                      key=lambda r: -F(r, "bridge_s"))
    seen = set();
    def take(rowset):
        out = []
        for r in rowset:
            h = r.get("hash", "")
            if h in seen: continue
            seen.add(h)
            out.append(({k: F(r, k) for k in order}, F(r, "gamma_A"), F(r, "kappa"), F(r, "c_A"),
                        r.get("hash", "?"), F(r, "bridge_s")))
            if len(out) >= n_each: break
        return out
    return take(stable), take(unstable)


def auc(pos, neg):
    """P(pos > neg): >0.5 means 'pos' (stable) tends higher. 0.5 = no separation."""
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="configs per class (stable / unstable)")
    args = ap.parse_args()
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "SUBSTRATE_HUNT_2026*")))[-1]
    stable, unstable = select_population(d, args.n)
    rng = np.random.default_rng(SEED)
    print(f"=== PASSIVE PAYAN DIAGNOSTIC (no coupling) — {len(stable)} stable vs {len(unstable)} unstable ===")
    print("Payan alignment A in [-1,1] (axial spin co-handedness), corridor coherence R; "
          "control = density-preserved phase-randomized\n")
    rec = {"stable": [], "unstable": []}
    for klass, pop in (("stable", stable), ("unstable", unstable)):
        for par, g, kap, cA, h, br in pop:
            t0 = time.time()
            psi = settle(par, g, kap, cA, 48)
            if psi is None:
                print(f"  [{klass} {h}] settle_nonfinite  ({time.time()-t0:.0f}s)", flush=True); continue
            ob = payan_observables(psi, par, 48)
            if ob is None:
                print(f"  [{klass} {h}] <2 nodes  ({time.time()-t0:.0f}s)", flush=True); continue
            ctrl = payan_observables(phase_randomized(psi, rng), par, 48)
            row = {"hash": h, "bridge_s_hunt": br, **ob,
                   "aligned_ctrl": (ctrl["aligned"] if ctrl else None),
                   "spin_mag_ctrl": (ctrl["spin_mag"] if ctrl else None), "R_ctrl": (ctrl["R"] if ctrl else None)}
            rec[klass].append(row)
            print(f"  [{klass} {h}] aligned={ob['aligned']} A={ob['A']:+.2f} |s|={ob['spin_mag']:.2e} "
                  f"R={ob['R']:.3f} cond={ob['bridge_cond']:.3f} | ctrl aligned={row['aligned_ctrl']}  ({time.time()-t0:.0f}s)", flush=True)
    # aggregate: do STABLE bridges have ALIGNED (co-handed) Payan spins more often than UNSTABLE,
    # and does the density-preserved phase-randomized control NOT reproduce that gap (~50% both)?
    def vals(klass, key): return [r[key] for r in rec[klass] if r.get(key) is not None]
    def mean(klass, key):
        v = vals(klass, key); return float(np.mean(v)) if v else float("nan")
    fa_s, fa_u = mean("stable", "aligned"), mean("unstable", "aligned")
    fa_s_c, fa_u_c = mean("stable", "aligned_ctrl"), mean("unstable", "aligned_ctrl")
    gap, gap_ctrl = fa_s - fa_u, fa_s_c - fa_u_c
    print(f"\n--- Payan alignment fraction (aligned = co-handed axial spins on the bridge pair) ---")
    print(f"  STABLE   aligned: {fa_s:.2f}  (control {fa_s_c:.2f})   mean|s|={mean('stable','spin_mag'):.2e}")
    print(f"  UNSTABLE aligned: {fa_u:.2f}  (control {fa_u_c:.2f})   mean|s|={mean('unstable','spin_mag'):.2e}")
    print(f"  gap (stable - unstable): real {gap:+.2f} | control {gap_ctrl:+.2f}")
    print(f"  corridor coherence R (saturates ~1 for smooth fields; reported, not gated): "
          f"stable {mean('stable','R'):.3f} unstable {mean('unstable','R'):.3f}")
    # signal = stable substantially more aligned than unstable, NOT reproduced by the phase-random control
    signal = (abs(gap) >= 0.25) and (abs(gap) > abs(gap_ctrl) + 0.15)
    verdict = "PASSIVE_PAYAN_ALIGNMENT_CORRELATES_WITH_STABILITY" if signal else "PASSIVE_PAYAN_ALIGNMENT_NO_SIGNAL"
    out = {"n_stable": len(rec["stable"]), "n_unstable": len(rec["unstable"]),
           "frac_aligned_stable": fa_s, "frac_aligned_unstable": fa_u,
           "frac_aligned_stable_ctrl": fa_s_c, "frac_aligned_unstable_ctrl": fa_u_c,
           "gap": gap, "gap_ctrl": gap_ctrl,
           "mean_spin_stable": mean("stable", "spin_mag"), "mean_spin_unstable": mean("unstable", "spin_mag"),
           "mean_R_stable": mean("stable", "R"), "mean_R_unstable": mean("unstable", "R"),
           "verdict": verdict, "records": rec}
    od = os.path.join(d, "payan_passive_diagnostic.json")
    json.dump(out, open(od, "w"), indent=2, default=float)
    print(f"\nVERDICT: {verdict}")
    print("  -> PAYAN_COUPLING_RFC_READY" if signal else
          "  -> PAYAN_COUPLING_NOT_JUSTIFIED (passive Payan alignment does not predict stability here)")
    print(f"wrote {od}")
    print("NOTE: passive correlation only; NOT proof of Payan. Coupling remains unbuilt.")


if __name__ == "__main__":
    main()
