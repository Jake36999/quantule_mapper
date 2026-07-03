"""H7.2 — unit tests for the re-aimed stability objective (tools/stability_objective.py).

Self-contained (synthetic metrics; no sweep-data dependency); runs on the dev box (no jax/cupy). Verifies the
scorer ranks the validated Phase C target (a*-stationary) above decayers/growers/window-artifacts and enforces
the disqualifiers + long-window gate. The offline re-score over REAL load-bearing CSVs is reported separately in
docs/HUNTER_REAIM_OFFLINE_RESCORE.md.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.stability_objective import stability_score, seed_robust_score, PRIME_SSE_RETIRED

STATIONARY = {"er_fin": 2.04, "er_max": 2.07, "floor_ratio": 1.0, "late_slope_50pct_per1k": -0.0004, "T": 144000}
DECAYER    = {"er_fin": 1.36, "er_max": 1.89, "floor_ratio": 1.0, "late_slope_50pct_per1k": -0.008, "T": 72000}
GROWER     = {"er_fin": 2.76, "er_max": 2.76, "floor_ratio": 1.0, "late_slope_50pct_per1k": 0.008, "T": 72000}
BLOWUP     = {"er_fin": 5.0, "er_max": 30.0, "floor_ratio": 1.0, "late_slope_50pct_per1k": 0.05, "T": 24000}
SPINDOWN   = {"er_fin": 0.10, "er_max": 1.6, "floor_ratio": 0.1, "late_slope_50pct_per1k": -0.02, "T": 72000}


def test_prime_sse_retired():
    assert PRIME_SSE_RETIRED is True   # log_prime_sse is exploratory-only, never steering


def test_stationary_beats_decayer_beats_grower():
    s = {k: stability_score(v)["score"] for k, v in
         dict(stat=STATIONARY, dec=DECAYER, grow=GROWER).items()}
    assert s["stat"] > s["dec"] > s["grow"], s


def test_blowup_and_spindown_rejected():
    b, d = stability_score(BLOWUP), stability_score(SPINDOWN)
    assert b["reject"] == "GROWER_BLOWUP" and b["score"] < 0
    assert d["reject"] == "SPIN_DOWN" and d["score"] < 0


def test_window_length_gate():
    short = dict(STATIONARY); short["T"] = 6000
    long_s = stability_score(STATIONARY)
    short_s = stability_score(short)
    assert long_s["certifiable"] and not short_s["certifiable"]
    assert short_s["score"] < long_s["score"]   # short-window discounted, never promoted alone


def test_indivisibility_is_a_pending_hook():
    # cannot be scored from a summary — must remain an explicit run-based hook (design spec §3)
    assert stability_score(STATIONARY)["indivisibility"] is None


def test_seed_robustness_weakest_governs_and_reject_propagates():
    # one weak-but-ok seed drags the aggregate below the strong seed
    agg_ok = seed_robust_score([STATIONARY, DECAYER])
    assert agg_ok["score"] == stability_score(DECAYER)["score"] and agg_ok["any_reject"] is False
    # any rejecting seed disqualifies the config
    agg_bad = seed_robust_score([STATIONARY, SPINDOWN])
    assert agg_bad["any_reject"] is True and agg_bad["score"] < 0
