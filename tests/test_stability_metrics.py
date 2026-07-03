"""H7 production alignment -- tests for solver/stability_metrics.py (pure numpy, no cupy/jax).

Proves the production metric math is IDENTICAL to jax_scout.core_saturation_search.classify's er-math (so the
re-aimed Hunter scores the CuPy path exactly as it scored the jax_scout mirror), that it is cadence-independent
(production samples every ~10 steps; css samples every step), and that its output feeds
tools.stability_objective.stability_score correctly. Runs on the dev box (no GPU).
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solver.stability_metrics import compute, from_history          # pure numpy
from tools.stability_objective import stability_score               # jax-free


def _css_reference(er):
    """Verbatim replica of core_saturation_search.classify's er-math (lines 219-233), the source of truth."""
    er = np.asarray(er, dtype=np.float64)
    er_max = float(np.max(er)); er_fin = float(er[-1])
    half = len(er) // 2
    late = np.asarray(er[half:], dtype=np.float64)
    xs = np.arange(len(late))
    if len(xs) > 2:
        coef = np.polyfit(xs, late, 1)
        slope = float(coef[0]); fit_start = float(coef[1])
        fit_end = float(coef[0] * (len(late) - 1) + coef[1])
        late_drift = (fit_end - fit_start) / (abs(fit_start) + 1e-9)
    else:
        slope, late_drift = 0.0, 0.0
    er0 = float(er[0]); er_min = float(np.min(er))
    floor_ratio = er_min / (abs(er0) + 1e-9)
    return dict(er_fin=er_fin, er_max=er_max, late_slope=slope, late_drift=late_drift, floor_ratio=floor_ratio)


# a few representative er trajectories (sampled every step, ic_e=1)
def _flat(n=400, val=2.06):    # a*-like: rises then holds flat
    er = np.full(n, val); er[: n // 4] = np.linspace(1.0, val, n // 4); return er
def _decayer(n=400):           # slow decay
    return np.concatenate([np.linspace(1.0, 1.7, n // 4), np.linspace(1.7, 1.28, n - n // 4)])
def _grower(n=400):            # runs above band
    return np.linspace(1.0, 2.9, n)


def test_matches_css_math_exactly():
    for er in (_flat(), _decayer(), _grower()):
        steps = np.arange(er.size)                     # cadence 1 == css
        m = compute(steps, er, ic_e=1.0, T=er.size)
        ref = _css_reference(er)
        assert np.isclose(m["er_fin"], ref["er_fin"])
        assert np.isclose(m["er_max"], ref["er_max"])
        assert np.isclose(m["floor_ratio"], ref["floor_ratio"])
        # css late_slope is per-step; our per-1k is *1000
        assert np.isclose(m["late_slope_50pct_per1k"], ref["late_slope"] * 1000.0)
        assert np.isclose(m["late_drift"], ref["late_drift"])


def test_cadence_independent_slope():
    # a linear er over 0..T: sampling every step vs every 10 steps must give the same per-1k slope
    T = 36000; m_per_step = 3e-6
    full = np.arange(0, T)
    er_full = 1.5 + m_per_step * full
    sparse = np.arange(0, T, 10)
    er_sparse = 1.5 + m_per_step * sparse
    a = compute(full, er_full, 1.0, T)["late_slope_50pct_per1k"]
    b = compute(sparse, er_sparse, 1.0, T)["late_slope_50pct_per1k"]
    assert np.isclose(a, b, atol=1e-6)
    assert np.isclose(a, m_per_step * 1000.0, atol=1e-6)     # == slope-per-step * 1000


def test_feeds_objective_ranking():
    # the production metrics, at a certifiable window, must rank a*-flat > decayer and reject the grower
    T = 36000
    flat = compute(np.arange(_flat().size), _flat(), 1.0, T)
    decay = compute(np.arange(_decayer().size), _decayer(), 1.0, T)
    grow = compute(np.arange(_grower().size), _grower(), 1.0, T)
    s_flat = stability_score(flat); s_decay = stability_score(decay); s_grow = stability_score(grow)
    assert s_flat["score"] > s_decay["score"]                 # flat a* beats slow decayer
    assert s_flat["score"] > s_grow["score"]                  # and beats the grower
    assert s_flat["certifiable"]                               # flat long-window certifies
    assert s_grow["components"]["band"] == 0.3                 # grower penalized (er_fin above band, like a*1.25)
    # a true blow-up (er_max > BLOWUP=3.0) is HARD-rejected, mirroring the css gate
    blow = compute(np.arange(400), np.linspace(1.0, 3.6, 400), 1.0, T)
    assert stability_score(blow)["reject"] == "GROWER_BLOWUP"


def test_from_history_uses_raw_energy_not_floored():
    # from_history must read 'raw_energy' (css convention), ignoring the legacy floored/dV 'energy'
    er = _flat()
    hist = [{"step": int(s), "energy": 999.0 + s, "raw_energy": float(e)} for s, e in enumerate(er)]
    m_hist = from_history(hist, ic_e=1.0, T=er.size)
    m_direct = compute(np.arange(er.size), er, ic_e=1.0, T=er.size)
    assert np.isclose(m_hist["late_slope_50pct_per1k"], m_direct["late_slope_50pct_per1k"])
    assert np.isclose(m_hist["er_fin"], m_direct["er_fin"])


def test_graceful_empty():
    assert from_history([], 1.0, 100)["reject"] == "NO_HISTORY"
    assert from_history([{"step": 0, "energy": 1.0}], 1.0, 100)["reject"] == "NO_RAW_ENERGY"
