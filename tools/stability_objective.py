#!/usr/bin/env python3
"""H7.1 — stability objective scorer (design → offline). STANDALONE; NOT yet wired into aste_hunter.

Replaces the retired prime-log-SSE steering objective with a score toward the *validated* Phase C target: a
gain/loss-balanced standing attractor. Scores the offline-computable stability components from a run's recorded
metrics (late-slope→0, long-window boundedness, v3 bounded-breathing, energy-band safety, window-length gate).
The **indivisibility** component is a run-based perturbation-response measurement (design spec §3) and is left as
an explicit hook (`None`) here — it cannot be scored from a summary alone.

Governing rule: Hunter *proposes*, gate *certifies*. This scorer guides search; `core_saturation_search.classify`
(v3) remains the promotion authority. See docs/HUNTER_REAIM_DESIGN_SPEC.md. Pure-Python; no jax/cupy; no solver,
geometry, gate, or hunter changes.
"""
import math

# --- calibrated to the validated gate (PHASE_C_GATE_CALIBRATION_SUMMARY.md) ---
ER_BAND = (0.5, 2.5)          # long-window energy-retention band
BLOWUP = 3.0                  # er_max above this = grower/blow-up reject
SPIN_FLOOR = 0.3              # er_fin below this = spun-down reject
FLOOR_MIN = 0.85             # v3 breathing floor_ratio = er_min/er0
PEAK_MARGIN = 0.05           # v3 breathing peak margin
MIN_STABILITY_T = 24000      # short windows cannot CERTIFY stability (window-artifact lesson)
SLOPE_SCALE = 0.005          # late-slope /1k normalization (a×1.15 ~ -0.0004 -> ~0.92)

PRIME_SSE_RETIRED = True      # log_prime_sse is an exploratory recorded diagnostic ONLY, never steering


def _f(m, *keys, default=None):
    for k in keys:
        if k in m and m[k] not in ("", None):
            try:
                return float(m[k])
            except (TypeError, ValueError):
                pass
    return default


def stability_score(m):
    """Score one run's metrics toward gain/loss-balanced standing-attractor stability.

    m: dict-like of recorded metrics (er_fin, er_max, floor_ratio, late_slope_50pct_per1k or late_drift, T, ...).
    Returns dict: {score, certifiable, reject, components{...}, indivisibility}. Higher score = better;
    reject!=None and score<=0 for disqualified states. `certifiable` gates on the long-window requirement.
    """
    er_fin = _f(m, "er_fin", default=0.0)
    er_max = _f(m, "er_max", default=er_fin)
    floor = _f(m, "floor_ratio", default=0.0)
    slope = abs(_f(m, "late_slope_50pct_per1k", "late_slope_10pct_per1k", "late_drift", default=1.0))
    T = _f(m, "T", default=0.0)

    # hard disqualifiers (mirror the gate's reject classes) ---------------------
    if er_max > BLOWUP:
        return {"score": -1.0, "reject": "GROWER_BLOWUP", "certifiable": False, "components": {}, "indivisibility": None}
    if er_fin < SPIN_FLOOR or floor < 0.5:
        return {"score": -0.5, "reject": "SPIN_DOWN", "certifiable": False, "components": {}, "indivisibility": None}

    # graded stability components (each in [0,1], higher = better) ---------------
    c_slope = math.exp(-slope / SLOPE_SCALE)                                  # late-slope -> 0 : PRIMARY (a*)
    c_band = 1.0 if ER_BAND[0] <= er_fin <= ER_BAND[1] else 0.3               # long-window boundedness
    c_breath = 1.0 if (floor >= FLOOR_MIN and er_fin <= (1.0 - PEAK_MARGIN) * er_max) else 0.6  # v3 breather
    certifiable = T >= MIN_STABILITY_T                                        # window-length gate

    score = 0.60 * c_slope + 0.25 * c_band + 0.15 * c_breath
    if not certifiable:
        score *= 0.5   # short-window results are discounted, never promoted on their own
    return {"score": round(score, 4), "reject": None, "certifiable": certifiable,
            "components": {"slope": round(c_slope, 4), "band": c_band, "breath": c_breath},
            "indivisibility": None}   # run-based perturbation hook (design spec §3) — not summary-derivable


def seed_robust_score(rows):
    """Seed robustness: aggregate per-seed stability scores for the SAME config.
    Rewards existence-of-stability across seeds; penalizes if any seed rejects.
    rows: iterable of metric dicts (one per seed)."""
    scored = [stability_score(r) for r in rows]
    if not scored:
        return {"score": 0.0, "n_seeds": 0, "all_certifiable": False, "any_reject": True}
    rejected = any(s["reject"] for s in scored)
    base = min(s["score"] for s in scored)                 # weakest seed governs (robustness)
    if rejected:
        base = min(base, -0.25)
    return {"score": round(base, 4), "n_seeds": len(scored),
            "all_certifiable": all(s["certifiable"] for s in scored), "any_reject": rejected}


if __name__ == "__main__":
    # tiny self-demo (no data dependency)
    demo = {"a*-stationary": {"er_fin": 2.04, "er_max": 2.07, "floor_ratio": 1.0, "late_slope_50pct_per1k": -0.0004, "T": 72000},
            "slow-decayer":   {"er_fin": 1.36, "er_max": 1.89, "floor_ratio": 1.0, "late_slope_50pct_per1k": -0.008, "T": 72000},
            "grower":         {"er_fin": 2.76, "er_max": 2.76, "floor_ratio": 1.0, "late_slope_50pct_per1k": 0.008, "T": 72000},
            "short-window":   {"er_fin": 2.04, "er_max": 2.07, "floor_ratio": 1.0, "late_slope_50pct_per1k": -0.0004, "T": 6000}}
    for k, v in demo.items():
        s = stability_score(v)
        print(f"  {k:16s} score={s['score']:+.3f} certifiable={s['certifiable']} reject={s['reject']}")
