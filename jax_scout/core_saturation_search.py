"""
PHASE C — TRUE LONG-TIME SATURATION search (corrected criterion; NOT T=1600 sustain). Following the
feb56dc7 discovery: classify by whether the energy SATURATES (flat late-window slope) into a localized,
node-count-stable steady state over T=4000, around the feb56dc7 regime (eta loss-side + strong cubic
gain). Bare S-NCGL, vmap sweep_probe (er(t) trajectory + psi_final). Answers: is the multi-node
repulsive state generic, or do 1/2/3-node true saturators exist?

Classes: TRUE_SATURATED_BOUND_STATE / NEAR_SATURATED_BOUND_STATE / TRANSIENT_GROWER_REJECT /
LATE_BLOWUP_REJECT / FRAGMENTATION_REJECT / DELOCALIZED_HALO_REJECT / SPIN_DOWN_REJECT.
Records final node count of saturators (1/2/3/4/multi). Time-boxed; incremental CSV; seeds feb56dc7.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/core_saturation_search.py --hours 2
"""
import os, sys, csv, json, time, argparse, subprocess
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td
from jax_scout.afield_current_coupled import multiseed_ic, L as L_, dt as DT

order = physics.SWEEP_PARAM_ORDER     # [D, eta, rho_vac, omega0, a_coupling, s, f, a]
SEED = 20260619
PARAM_SEED = 20260622
CLASSIFIER_VERSION = "PHASE_C_SATURATION_CLASSIFIER_v3"
IC_NORM_PER_BLOB_FIXED = "per_blob_fixed"
IC_NORM_TOTAL_MASS_FIXED = "total_mass_fixed"
IC_NORM_CHOICES = [IC_NORM_PER_BLOB_FIXED, IC_NORM_TOTAL_MASS_FIXED]
# feb56dc7-regime sampling: eta loss-side, strong cubic gain a; other params permissive (logged)
REGIME = {"param_eta": [-0.02, 0.15], "param_a": [0.20, 0.50]}
# |late er slope| per step. Calibrated to the N=48 ground-truth saturator feb56dc7 (~9e-5 at T=4000,
# still finishing its climb; goes ~1e-6 by T=6000/N=96). Transient growers sit ~4e-4..1.3e-3, a clean
# ~4x gap above. SAT captures feb56dc7; finalists are re-validated at N=96/T=6000 where it flattens.
SAT_SLOPE, NEAR_SLOPE = 1.5e-4, 3.5e-4
# v2 long-time stability gate: normalized late-half energy drift (fractional change of a linear fit
# over [T/2, T]). The raw late slope misses slow growers/decayers (e.g. k6_mid) that sit in the
# in-band energy range; the drift catches them. Window-agnostic: weak at short discovery T (so it does
# not disturb T=4000 shortlisting), strong at long validation T (>=12000 it bites). Calibrated on the
# five T=24000 trajectories (feb stable 0.080 < 0.15 < 0.331 nearest failure) —
# docs/PHASE_C_STABILITY_GATE_CALIBRATION.md. Provisional threshold (one stable exemplar).
LATE_DRIFT_MAX = 0.15
# v3 breathing-aware exception: a large late-half drift that is just the downswing of a *bounded
# oscillation* must not be rejected as decay/growth. A genuine breathing bound state (1) never falls
# meaningfully below its starting energy (floor_ratio = er_min/er0 >= BREATHING_FLOOR_RATIO_MIN) and
# (2) ends below its own peak (er_fin <= (1-BREATHING_PEAK_MARGIN)*er_max), i.e. it came back down from
# a peak rather than monotonically decaying (er_fin~er_min) or growing (er_fin~er_max). This excludes the
# k6_mid-type monotonic in-band grower (er_fin~er_max -> peak-margin fails) and floorward decayers
# (floor_ratio fails), while rescuing feb at T=24000 (its breathing downswing). Calibrated on the real
# T=24000 traces; window-agnostic and verified not to change any T=12000 verdict. See
# docs/PHASE_C_GATE_V3_BREATHING_BOUND_STATE.md.
BREATHING_FLOOR_RATIO_MIN = 0.85
BREATHING_PEAK_MARGIN = 0.05
FEB = {"param_D": 2.7329, "param_eta": 0.0704, "param_rho_vac": 1.1866, "param_omega0": 0.0,
       "param_a_coupling": 2.3098, "param_s": 0.0129, "param_f": -0.4861, "param_a": 0.4802}


def classifier_spec():
    return {
        "version": CLASSIFIER_VERSION,
        "sat_slope": SAT_SLOPE,
        "near_slope": NEAR_SLOPE,
        "late_drift_max": LATE_DRIFT_MAX,
        "late_drift_metric": "normalized_late_half_linear_fit_fractional_change",
        "breathing_floor_ratio_min": BREATHING_FLOOR_RATIO_MIN,
        "breathing_peak_margin": BREATHING_PEAK_MARGIN,
        "breathing_exception": "accept bounded breathing (er_min/er0>=floor_min and er_fin<=(1-peak_margin)*er_max) despite drift",
        "blowup_er_max": 3.0,
        "spin_down_floor": 0.3,
        "energy_true_min": 0.5,
        "energy_true_max": 2.5,
        "fragmentation_node_cap": 8,
        "core_density_floor": 0.15,
        "negative_slopes_can_be_true": True,
        "requires_n_mid_eq_n_fin": False,
    }


def ic_descriptor(ic_counts, ic_seed=SEED, ic_norm=IC_NORM_PER_BLOB_FIXED, target_initial_mass=None):
    return {
        "generator": "multiseed_ic",
        "generator_path": "jax_scout.afield_current_coupled.multiseed_ic",
        "family": "summed_gaussians_plus_noise",
        "ic_seed": int(ic_seed),
        "param_seed": PARAM_SEED,
        "blob_counts": [int(k) for k in ic_counts],
        "ic_norm": ic_norm,
        "target_initial_mass": target_initial_mass,
        "blob_width_rule": "fixed_w=L/12",
        "placement_rule": "uniform_box_over_-L/2_to_L/2",
        "phase_rule": "real_positive_gaussians_plus_complex_gaussian_noise",
        "noise_rule": "0.01_complex_gaussian",
        "amplitude_rule": "per_blob_amplitude_fixed_at_1",
        "normalized_across_k": ic_norm == IC_NORM_TOTAL_MASS_FIXED,
    }


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            timeout=2.0,
        ).strip()
    except Exception:
        return "UNKNOWN"


def get_git_dirty():
    try:
        tracked = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=2.0,
        )
        cached = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "HEAD", "--"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=2.0,
        )
        return tracked.returncode != 0 or cached.returncode != 0
    except Exception:
        return True


def measure_ic(psi0):
    rho = np.abs(psi0) ** 2
    return {
        "initial_mass": float(np.sum(rho)),
        "initial_max_density": float(np.max(rho)),
    }


def build_bounds():
    bounds = json.load(open(os.path.join(ROOT, "jax_scout", "gain_bounds.json")))
    lo = np.array([bounds[k][0] for k in order])
    hi = np.array([bounds[k][1] for k in order])
    for k, (a, b) in REGIME.items():        # tighten eta & a to the feb56dc7 regime
        i = order.index(k)
        lo[i], hi[i] = a, b
    return bounds, lo, hi


def build_ic(N, K, seed=SEED, ic_norm=IC_NORM_PER_BLOB_FIXED, target_initial_mass=None):
    psi0 = np.asarray(multiseed_ic(N, seed, K))
    raw_stats = measure_ic(psi0)
    if ic_norm == IC_NORM_TOTAL_MASS_FIXED:
        if target_initial_mass is None or target_initial_mass <= 0:
            raise ValueError("total_mass_fixed requires a positive target_initial_mass")
        scale = np.sqrt(float(target_initial_mass) / max(raw_stats["initial_mass"], 1e-30))
        psi0 = psi0 * scale
    elif ic_norm != IC_NORM_PER_BLOB_FIXED:
        raise ValueError(f"Unsupported ic_norm '{ic_norm}'")
    stats = {
        "K": int(K),
        "ic_seed": int(seed),
        "ic_norm": ic_norm,
        "target_initial_mass": None if target_initial_mass is None else float(target_initial_mass),
        **measure_ic(psi0),
    }
    return psi0, stats


def evaluate_candidate(finite, energy_row, psi_mid, psi_fin, ic_e, dx):
    er = np.asarray(energy_row, dtype=np.float64) / ic_e
    viable = bool(finite) and np.isfinite(er).all() and float(np.max(er)) <= 3.0 and float(er[-1]) >= 0.3
    n_mid = len(td.detect_nodes(psi_mid, dx)) if viable else 0
    n_fin = len(td.detect_nodes(psi_fin, dx)) if viable else 0
    core_fin = float(np.max(np.abs(psi_fin) ** 2)) if viable else 0.0
    klass, metrics = classify(bool(finite), er, n_mid, n_fin, core_fin)
    return klass, metrics, er


def run_probe(params, N, T, K, seed=SEED, ic_norm=IC_NORM_PER_BLOB_FIXED, target_initial_mass=None):
    dx = L_ / N
    psi0, ic_stats = build_ic(N, K, seed, ic_norm=ic_norm, target_initial_mass=target_initial_mass)
    ic_e = ic_stats["initial_mass"] + 1e-30
    pvec = np.asarray([params[k] for k in order], dtype=np.float64)
    psi_mid, psi_fin, energy, max_amp, finite = physics.probe_one(
        jnp.asarray(pvec),
        jnp.asarray(psi0),
        N,
        L_,
        DT,
        T,
        jnp.float64,
        jnp.complex128,
    )
    psi_mid = np.asarray(psi_mid)
    psi_fin = np.asarray(psi_fin)
    energy = np.asarray(energy)
    max_amp = np.asarray(max_amp)
    finite = bool(np.asarray(finite))
    klass, metrics, er = evaluate_candidate(finite, energy, psi_mid, psi_fin, ic_e, dx)
    return {
        "klass": klass,
        "metrics": metrics,
        "er": er,
        "energy": energy,
        "max_amp": max_amp,
        "finite": finite,
        "psi0": psi0,
        "psi_mid": psi_mid,
        "psi_fin": psi_fin,
        "ic_e": ic_e,
        "ic_stats": ic_stats,
    }


def classify(finite, er, n_mid, n_fin, core_fin):
    er_max = float(np.max(er)); er_fin = float(er[-1])
    base = {"er_fin": er_fin, "er_max": er_max, "n_mid": n_mid, "n_fin": n_fin, "core_fin": core_fin}
    half = len(er) // 2
    late = np.asarray(er[half:], dtype=np.float64)
    xs = np.arange(len(late))
    if len(xs) > 2:
        coef = np.polyfit(xs, late, 1)
        slope = float(coef[0])
        fit_start = float(coef[1])
        fit_end = float(coef[0] * (len(late) - 1) + coef[1])
        late_drift = (fit_end - fit_start) / (abs(fit_start) + 1e-9)
    else:
        slope, late_drift = 0.0, 0.0
    er0 = float(er[0]); er_min = float(np.min(er))
    floor_ratio = er_min / (abs(er0) + 1e-9)
    bounded_breathing = (
        er_max <= 3.0
        and 0.5 <= er_fin <= 2.5
        and floor_ratio >= BREATHING_FLOOR_RATIO_MIN
        and er_fin <= (1.0 - BREATHING_PEAK_MARGIN) * er_max
    )
    base["late_slope"] = slope
    base["late_drift"] = late_drift
    base["er0"] = er0
    base["er_min"] = er_min
    base["floor_ratio"] = floor_ratio
    base["bounded_breathing"] = bool(bounded_breathing)
    if not finite or not np.isfinite(er_max) or er_max > 3.0:
        return "LATE_BLOWUP_REJECT", base
    if er_fin < 0.3:
        return "SPIN_DOWN_REJECT", base
    if n_fin > 8 or n_mid > 8:
        return "FRAGMENTATION_REJECT", base
    if n_fin < 1 or core_fin < 0.15:
        return "DELOCALIZED_HALO_REJECT", base       # energy not localized into cores
    if slope > NEAR_SLOPE:
        return "TRANSIENT_GROWER_REJECT", base
    # v2 long-time stability gate + v3 breathing exception: a large normalized late-half drift is a
    # transient (slow decay/growth the raw slope misses) UNLESS it is the downswing of a bounded
    # oscillation (bounded_breathing) — a genuine breathing bound state, which we accept.
    if abs(late_drift) > LATE_DRIFT_MAX:
        if bounded_breathing:
            return "TRUE_SATURATED_BOUND_STATE", base
        return ("TRANSIENT_GROWER_REJECT" if late_drift > 0 else "SPIN_DOWN_REJECT"), base
    if abs(slope) <= SAT_SLOPE and 0.5 <= er_fin <= 2.5:
        return "TRUE_SATURATED_BOUND_STATE", base
    if abs(slope) <= NEAR_SLOPE and 0.5 <= er_fin <= 2.5:
        return "NEAR_SATURATED_BOUND_STATE", base
    return "SPIN_DOWN_REJECT" if slope < 0 else "TRANSIENT_GROWER_REJECT", base


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--N", type=int, default=48); ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--ic-counts", default="6", help="comma list of IC blob counts K to round-robin (tests if node count tracks IC vs field-preferred)")
    ap.add_argument("--ic-norm", choices=IC_NORM_CHOICES, default=IC_NORM_PER_BLOB_FIXED)
    ap.add_argument("--target-initial-mass", type=float, default=None)
    ap.add_argument("--ic-seed", type=int, default=SEED, help="IC generator seed; default preserves historical Phase C behavior")
    ap.add_argument("--max-batches", type=int, default=None, help="optional hard cap on completed batches for small pilots")
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    ic_counts = [int(x) for x in args.ic_counts.split(",")]
    N, T, B = args.N, args.T, args.batch; dx = L_ / N
    _, lo, hi = build_bounds()
    rng = np.random.default_rng(PARAM_SEED)
    ic_cache = {
        K: build_ic(N, K, args.ic_seed, ic_norm=args.ic_norm, target_initial_mass=args.target_initial_mass)
        for K in set(ic_counts)
    }
    multi_ic = len(ic_counts) > 1
    tag = "CORE_SAT_CALIB" if args.calibrate else ("CORE_SAT_HUNT" if multi_ic else "CORE_SAT_PILOT")
    outdir = os.path.join(ROOT, "sweep_runs", f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"); os.makedirs(outdir, exist_ok=True)
    cols = ["idx", "label", "ic_blobs", "ic_seed", "ic_norm", "target_initial_mass", "initial_mass", "initial_max_density",
            "klass", "er_fin", "er_max", "late_slope", "n_mid", "n_fin", "core_fin", *order]
    log = open(os.path.join(outdir, "all_evals.csv"), "w", newline=""); cw = csv.DictWriter(log, fieldnames=cols, extrasaction="ignore"); cw.writeheader()
    deadline = float("inf") if args.calibrate else time.time() + args.hours * 3600
    maxb = 1 if args.calibrate else (args.max_batches if args.max_batches is not None else 10 ** 9)
    command = " ".join(sys.argv)
    commit = get_git_commit()
    dirty = get_git_dirty()
    t0 = time.time(); idx = 0; nb = 0; counts = {}; sat_nodes = {}; sat_by_K = {}
    print(f"=== PHASE C SATURATION SEARCH N={N} T={T} | regime eta{REGIME['param_eta']} a{REGIME['param_a']} "
          f"| IC blob counts {ic_counts} | ic_seed={args.ic_seed} | ic_norm={args.ic_norm}"
          + (f" target_mass={args.target_initial_mass:.6f}" if args.target_initial_mass is not None else "") + " ===",
          flush=True)
    while time.time() < deadline and nb < maxb:
        nb += 1
        K = ic_counts[(nb - 1) % len(ic_counts)]
        psi0_np, ic_stats = ic_cache[K]
        ic_e = ic_stats["initial_mass"] + 1e-30
        psi0 = jnp.asarray(psi0_np)
        rows, labels = [], []
        if nb == 1 and K == 6:
            rows.append(np.array([FEB[k] for k in order])); labels.append("ref_feb56dc7")
        while len(rows) < B:
            rows.append(lo + rng.random(len(order)) * (hi - lo)); labels.append("rand")
        pm = jnp.asarray(np.stack(rows)); tb = time.time()
        psi_mid, psi_fin, energy, max_amp, finite = physics.sweep_probe(pm, psi0, N, L_, DT, T, jnp.float64, jnp.complex128)
        energy = np.asarray(energy); finite = np.asarray(finite); psi_mid = np.asarray(psi_mid); psi_fin = np.asarray(psi_fin)
        for b in range(B):
            kl, m, _ = evaluate_candidate(bool(finite[b]), energy[b], psi_mid[b], psi_fin[b], ic_e, dx)
            counts[kl] = counts.get(kl, 0) + 1
            if kl == "TRUE_SATURATED_BOUND_STATE":
                sat_nodes[m["n_fin"]] = sat_nodes.get(m["n_fin"], 0) + 1
                sat_by_K.setdefault(K, {})[m["n_fin"]] = sat_by_K.setdefault(K, {}).get(m["n_fin"], 0) + 1
            cw.writerow({"idx": idx, "label": labels[b], "ic_blobs": K, "ic_seed": args.ic_seed, "ic_norm": args.ic_norm,
                         "target_initial_mass": ic_stats["target_initial_mass"], "initial_mass": ic_stats["initial_mass"],
                         "initial_max_density": ic_stats["initial_max_density"], "klass": kl, **m,
                         **{k: round(float(rows[b][i]), 4) for i, k in enumerate(order)}}); idx += 1
        log.flush(); el = (time.time() - t0) / 3600
        print(f"[batch {nb} K={K} t={el:.2f}h] evals={idx} TRUE_SAT={counts.get('TRUE_SATURATED_BOUND_STATE',0)} "
              f"NEAR={counts.get('NEAR_SATURATED_BOUND_STATE',0)} GROW={counts.get('TRANSIENT_GROWER_REJECT',0)} "
              f"BLOW={counts.get('LATE_BLOWUP_REJECT',0)} SPIN={counts.get('SPIN_DOWN_REJECT',0)} "
              f"HALO={counts.get('DELOCALIZED_HALO_REJECT',0)} | sat nodes={sat_nodes} by_K={sat_by_K}  ({time.time()-tb:.0f}s)", flush=True)
        ic_by_k = {str(k): stats for k, (_, stats) in ic_cache.items()}
        json.dump({"N": N, "T": T, "n_eval": idx, "elapsed_h": el, "counts": counts, "sat_node_counts": sat_nodes,
                   "sat_node_counts_by_IC_blobs": sat_by_K, "ic_counts": ic_counts, "regime": REGIME, "order": order,
                   "dt": DT, "git_commit": commit, "git_dirty": dirty, "command": command, "parameter_rng_seed": PARAM_SEED,
                   "classifier": classifier_spec(), "ic_seed": args.ic_seed,
                   "classifier_version": CLASSIFIER_VERSION,
                   "ic": ic_descriptor(ic_counts, args.ic_seed, args.ic_norm, args.target_initial_mass),
                   "ic_by_k": ic_by_k},
                  open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    log.close()
    print(f"\n=== PHASE C DONE: {idx} evals in {(time.time()-t0)/3600:.2f}h ===")
    print(f"counts={counts}\nTRUE_SATURATED node-count distribution: {sat_nodes}")
    print(f"TRUE_SATURATED node-count BY IC blob count: {sat_by_K}")
    print("  (does final node count TRACK the IC blob count -> field allows that multiplicity; "
          "or always 4-5 regardless -> field-preferred plurality?)")
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()
