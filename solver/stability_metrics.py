"""solver/stability_metrics.py -- production stability metrics (H7 production alignment).

Read-only reduction of a CuPy run's RAW energy trajectory into the summary metrics that
`tools.stability_objective.stability_score` consumes, so the re-aimed Hunter can score the *production*
(CuPy) path the same way the jax_scout re-discovery scored the mirror.

**No physics.** This module only *observes* numbers (sum|psi|^2 per telemetry sample) and reduces them; it
never touches psi evolution, the RHS, the operators, or the gate. It is **pure numpy (no cupy import)** so it
is unit-testable on any box, and it mirrors `jax_scout.core_saturation_search.classify`'s er-math EXACTLY:

    energy(t) = sum(|psi(t)|^2)            # RAW: no rho-floor, no dV (the css/objective convention)
    ic_e      = sum(|psi_0|^2)             # == core_saturation_search.measure_ic()["initial_mass"]
    er(t)     = energy(t) / ic_e
    late-half linear fit -> late_slope (per step), late_drift (fractional change over the late half)
    floor_ratio = min(er) / |er[0]|

The one cadence subtlety vs css: css samples er every step, so its `late_slope` is per-step. Production
telemetry samples every ~10 steps, so we fit against the ACTUAL step numbers and report
`late_slope_50pct_per1k` = (slope per step) * 1000 -- cadence-independent and identical to what the objective
saw on the jax_scout path.
"""
import numpy as np


def er_trace(energy, ic_e):
    """er(t) = energy(t) / ic_e, matching core_saturation_search.evaluate_candidate."""
    return np.asarray(energy, dtype=np.float64) / (float(ic_e) + 1e-30)


def compute(steps, energy, ic_e, T):
    """Reduce a raw-energy trajectory to the objective's stability_metrics dict.

    steps  : int step index for each telemetry sample (real step numbers -> cadence-independent slope)
    energy : raw sum(|psi|^2) per sample
    ic_e   : sum(|psi_0|^2) (initial raw energy) used to normalise er
    T      : total intended step count (the run window; the objective's window-gate uses it)
    """
    steps = np.asarray(steps, dtype=np.float64)
    er = er_trace(energy, ic_e)
    n = int(er.size)
    if n == 0:
        return {"reject": "NO_RAW_ENERGY", "T": int(T), "n_samples": 0}
    er_fin = float(er[-1]); er_max = float(np.max(er)); er0 = float(er[0]); er_min = float(np.min(er))
    floor_ratio = er_min / (abs(er0) + 1e-9)
    half = n // 2
    late = er[half:]; late_x = steps[half:]
    if late.size > 2:
        coef = np.polyfit(late_x, late, 1)                     # slope PER STEP (real step numbers)
        slope_per_step = float(coef[0])
        fit_start = float(coef[0] * late_x[0] + coef[1])       # fitted er at start of late half (== css)
        fit_end = float(coef[0] * late_x[-1] + coef[1])        # fitted er at end of late half
        late_drift = (fit_end - fit_start) / (abs(fit_start) + 1e-9)
    else:
        slope_per_step, late_drift = 0.0, 0.0
    return {
        "er_fin": er_fin,
        "er_max": er_max,
        "er0": er0,
        "er_min": er_min,
        "floor_ratio": floor_ratio,
        "late_slope_50pct_per1k": slope_per_step * 1000.0,     # per-1000-STEPS, == css late_slope*1000
        "late_drift": late_drift,
        "T": int(T),
        "n_samples": n,
        "energy_definition": "sum(abs(psi)**2) raw (no floor, no dV); er=energy/ic_e; ic_e=sum(abs(psi0)**2)",
    }


def from_history(history, ic_e, T):
    """Build stability_metrics from solver/run.py's `history` list of telemetry dicts.

    Uses the 'raw_energy' field (sum|psi|^2, added by the run loop as a read-only observer); ignores the
    legacy floored/dV 'energy' field so the metric matches the css/objective convention. Never raises.
    """
    if not history:
        return {"reject": "NO_HISTORY", "T": int(T), "n_samples": 0}
    steps, energy = [], []
    for h in history:
        if "raw_energy" in h and h["raw_energy"] is not None:
            steps.append(h["step"]); energy.append(h["raw_energy"])
    if not energy:
        return {"reject": "NO_RAW_ENERGY", "T": int(T), "n_samples": 0}
    return compute(steps, energy, ic_e, T)
