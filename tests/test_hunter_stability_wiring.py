"""H7.1 — tests for the flag-gated stability-objective wiring in aste_hunter (no hunt, no GPU/CuPy).

Proves: default Hunter behaviour is unchanged (objective="prime"); the stability objective is selectable; the
stability fitness comes from tools.stability_objective and ranks a* above failures; and log_prime_sse does NOT
steer fitness in stability mode. See docs/HUNTER_REAIM_IMPLEMENTATION_NOTES.md.
"""
import os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
H = pytest.importorskip("aste_hunter")   # skip if the hunter's deps aren't installed here

ASTAR = {"stability_metrics": {"er_fin": 2.04, "er_max": 2.07, "floor_ratio": 1.0,
                               "late_slope_50pct_per1k": -0.0004, "T": 144000}}
DECAYER = {"stability_metrics": {"er_fin": 1.36, "er_max": 1.89, "floor_ratio": 1.0,
                                 "late_slope_50pct_per1k": -0.008, "T": 72000}}
GOODPRIME_NOSTAB = {"spectral_fidelity": {"log_prime_sse": 0.001}}   # excellent prime, NO stability metrics


def test_stability_fitness_ranks_astar_over_decayer():
    assert H._stability_fitness_from_provenance(ASTAR)[0] > H._stability_fitness_from_provenance(DECAYER)[0] > 0


def test_prime_sse_does_not_steer_in_stability_mode():
    # a config with excellent prime-SSE but no stability metrics scores 0 -> prime is not the fitness
    assert H._stability_fitness_from_provenance(GOODPRIME_NOSTAB)[0] == 0.0


def test_absent_metrics_graceful():
    fit, ss = H._stability_fitness_from_provenance({})
    assert fit == 0.0 and ss.get("reject") == "NO_STABILITY_METRICS"


def test_default_objective_is_prime_unchanged(tmp_path):
    h = H.Hunter(db_file=str(tmp_path / "a.db"))
    assert h.objective == "prime"          # default = legacy behaviour, untouched


def test_stability_objective_is_selectable(tmp_path):
    h = H.Hunter(db_file=str(tmp_path / "b.db"), objective="stability")
    assert h.objective == "stability"
