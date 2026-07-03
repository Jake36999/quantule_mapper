"""
Independence-floor control for the phase-coupling channel of transfer_diag (v2).

The surviving signal after null-referencing is phase_coupling_score (~0.66-0.68 on the
gen15 configs). Before trusting it we need the metric's NOISE FLOOR under GUARANTEED
independence: feed the SAME pairwise pipeline two node-phase residual series that come
from DIFFERENT, independent simulations (true coupling = 0 by construction). If the
in-situ within-candidate coupling is well above this cross-candidate floor, the signal
is genuine pairwise coupling; if not, it is pipeline bias.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/transfer_null_control.py
"""
import os, sys, glob, csv
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td
from jax_scout.run_transfer_diag import intact_ic, N, L, dt, STEPS, N_SNAP, order


def prepped_nodes(pvec, par, ic):
    snaps, finite = td.capture_trajectory(pvec, ic, N, L, dt, STEPS, N_SNAP)
    if not finite:
        return None
    dx = L / N
    snap_nodes = [td.detect_nodes(s, dx) for s in snaps]
    tracks = td.track_nodes(snap_nodes, N)
    E_tot = np.array([float(np.sum(np.abs(s) ** 2)) for s in snaps])
    gph = np.unwrap(np.array([float(np.angle(np.sum(s * np.abs(s) ** 2))) for s in snaps]))
    gidx = np.arange(len(gph)); gsl, gint = np.polyfit(gidx, gph, 1)
    dphi_glob = gph - (gsl * gidx + gint)
    return [p for p in (td.prep_track(t, E_tot, dphi_glob) for t in tracks) if p is not None]


def pair_pcoup(di, dj, rng):
    Tn = N_SNAP + 1
    return td.temporal_pair_metrics(di, dj, Tn, rng)["phase_couple_excess"]


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "ADAPTIVE_HUNT_2026062*")))[-1]
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    def f(r, k):
        try: return float(r[k])
        except: return float("nan")
    bnd = [r for r in rows if r["reject"] == "" and 2 <= f(r, "intact_nodes") <= 20
           and f(r, "curv") < 1.0 and 0.1 <= f(r, "er") <= 5.0]
    bnd.sort(key=lambda r: f(r, "iso_surv"))
    ic = intact_ic(N, L)
    rng = np.random.default_rng(td.SURR_SEED)

    # build prepped node sets for the first few independent candidates
    sets = []
    for r in bnd[:4]:
        pv = [float(r[k]) for k in order]; par = {k: float(r[k]) for k in order}
        nodes = prepped_nodes(pv, par, ic)
        if nodes and len(nodes) >= 2:
            sets.append((r["gen"], nodes))
    print(f"prepped {len(sets)} candidate node-sets from {os.path.basename(d)}\n")

    within, cross = [], []
    for gi, (g, nodes) in enumerate(sets):
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                within.append(pair_pcoup(nodes[a], nodes[b], rng))
    # cross-candidate: nodes from different sims => true coupling = 0
    for gi in range(len(sets)):
        for gj in range(gi + 1, len(sets)):
            for a in sets[gi][1]:
                for b in sets[gj][1]:
                    cross.append(pair_pcoup(a, b, rng))

    within = np.array(within); cross = np.array(cross)
    print("phase_coupling_excess WITHIN candidate (in-situ, real pairs):")
    print(f"   n={len(within)}  mean={within.mean():.3f}  med={np.median(within):.3f}  "
          f"min={within.min():.3f}  max={within.max():.3f}")
    print("phase_coupling_excess CROSS candidate (independence floor, true coupling=0):")
    print(f"   n={len(cross)}  mean={cross.mean():.3f}  med={np.median(cross):.3f}  "
          f"p95={np.percentile(cross,95):.3f}  max={cross.max():.3f}")
    sep = within.mean() - cross.mean()
    pooled = np.sqrt(0.5 * (within.var() + cross.var())) + 1e-9
    print(f"\nseparation: within - cross = {sep:.3f}   (cohen-d ~ {sep/pooled:.2f})")
    frac = float(np.mean(within[:, None] > np.percentile(cross, 95)))
    print(f"fraction of within-pairs above cross p95: {frac:.2f}")
    if within.mean() > cross.mean() + 0.10 and np.median(within) > np.percentile(cross, 95):
        print("VERDICT: in-situ phase coupling EXCEEDS the independence floor "
              "-> structured pairwise signal (scout-level; needs CuPy validation).")
    else:
        print("VERDICT: in-situ phase coupling NOT separable from the independence floor "
              "-> likely pipeline/common-mode bias, NOT genuine transfer.")


if __name__ == "__main__":
    main()
