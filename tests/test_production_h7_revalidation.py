"""A5 -- tests for the production H7 re-validation evaluator (tools/production_h7_revalidation.py).

Exercises the reachable/scientific core (score production provenance -> A5 verdict) with mocked provenance
reports; no CuPy/GPU. The worker + validation_pipeline legs are box-gated and covered by the runbook.
"""
import os, sys, json
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.production_h7_revalidation as P

# stability_metrics blocks (shape solver/stability_metrics.compute emits)
SM_ASTAR_LONG = {"er_fin": 2.06, "er_max": 2.075, "floor_ratio": 1.0, "late_slope_50pct_per1k": -0.0007, "T": 36000}
SM_DECAY_LONG = {"er_fin": 1.28, "er_max": 1.73, "floor_ratio": 1.0, "late_slope_50pct_per1k": -0.015, "T": 36000}
SM_GROW_LONG = {"er_fin": 2.9, "er_max": 2.9, "floor_ratio": 1.0, "late_slope_50pct_per1k": 0.017, "T": 36000}
SM_ASTAR_SHORT = {"er_fin": 2.06, "er_max": 2.075, "floor_ratio": 1.0, "late_slope_50pct_per1k": -0.0007, "T": 12000}
SM_FLAT_HIGH = {"er_fin": 2.0, "er_max": 2.02, "floor_ratio": 1.0, "late_slope_50pct_per1k": -0.0002, "T": 36000}


def _write_provenance(prov_dir, name_to_sm):
    """Write provenance_{config_hash}.json for each named cell, using the harness's own config_hash."""
    prov_dir.mkdir(exist_ok=True)
    for (name, af, t, role) in P.CELLS:
        if name not in name_to_sm:
            continue
        ch = P.cell_config(name, af, t)["config_hash"]
        (prov_dir / f"provenance_{ch}.json").write_text(
            json.dumps({"spectral_fidelity": {"log_prime_sse": 999.0}, "stability_metrics": name_to_sm[name]}),
            encoding="utf-8")


def test_config_hash_deterministic_and_window_distinct():
    # same call -> same hash; a* long vs short (same params, different T) -> distinct hashes
    assert P.cell_config("astar_longT", 1.15, P.LONG_T)["config_hash"] == P.cell_config("astar_longT", 1.15, P.LONG_T)["config_hash"]
    assert P.cell_config("astar_longT", 1.15, P.LONG_T)["config_hash"] != P.cell_config("astar_shortT", 1.15, P.SHORT_T)["config_hash"]
    # a* param_a is feb * 1.15
    assert abs(P.cell_config("astar_longT", 1.15, P.LONG_T)["param_a"] - P.FEB["param_a"] * 1.15) < 1e-9


def test_build_configs_writes_worker_params(tmp_path):
    idx = P.build_configs(str(tmp_path))
    assert len(idx) == len(P.CELLS)
    cfg = json.load(open(os.path.join(str(tmp_path), "astar_longT.params.json")))
    # worker_cupy --params shape: physics params + simulation + global_seed + config_hash
    assert cfg["simulation"]["t_steps"] == P.LONG_T and cfg["simulation"]["n_grid"] == P.N_GRID
    assert "param_a" in cfg and "config_hash" in cfg and cfg["global_seed"] == P.SEED


def test_evaluate_pass(tmp_path):
    prov = tmp_path / "prov"
    _write_provenance(prov, {"astar_longT": SM_ASTAR_LONG, "decayer_longT": SM_DECAY_LONG,
                             "grower_longT": SM_GROW_LONG, "astar_shortT": SM_ASTAR_SHORT})
    res = P.evaluate(str(prov))
    assert res["verdict"] == "PRODUCTION_H7_REVALIDATION_PASS", res
    assert all(res["checks"].values())
    # a* is certifiable + top; short-window a* is NOT certifiable (no artifact promotion)
    assert res["checks"]["astar_certifiable"] and res["checks"]["shortT_not_certifiable"]


def test_evaluate_review_when_failure_outranks_astar(tmp_path):
    # FAIL trigger: the objective ranks a control above a* (here a* decays, grower looks flat/high)
    prov = tmp_path / "prov"
    _write_provenance(prov, {"astar_longT": SM_DECAY_LONG, "decayer_longT": SM_DECAY_LONG,
                             "grower_longT": SM_FLAT_HIGH, "astar_shortT": SM_ASTAR_SHORT})
    res = P.evaluate(str(prov))
    assert res["verdict"] == "PRODUCTION_H7_REVALIDATION_REVIEW"
    assert "astar_top_of_longT" in res["failed_checks"]


def test_evaluate_incomplete_when_missing(tmp_path):
    prov = tmp_path / "prov"
    _write_provenance(prov, {"astar_longT": SM_ASTAR_LONG})   # only one cell present
    res = P.evaluate(str(prov))
    assert res["verdict"] == "INCOMPLETE"


def test_evaluate_resolves_identity_folded_provenance(tmp_path):
    # the evaluator must find provenance even when the writer folded seed/run_id into the filename (A4b)
    from orchestrator.run_identity import provenance_filename
    prov = tmp_path / "prov"; prov.mkdir()
    smmap = {"astar_longT": SM_ASTAR_LONG, "decayer_longT": SM_DECAY_LONG,
             "grower_longT": SM_GROW_LONG, "astar_shortT": SM_ASTAR_SHORT}
    for (name, af, t, role) in P.CELLS:
        ch = P.cell_config(name, af, t)["config_hash"]
        fname = provenance_filename(ch, seed=620, run_id="deadbeef", utc_date="2026-07-03")
        (prov / fname).write_text(json.dumps({"stability_metrics": smmap[name]}), encoding="utf-8")
    res = P.evaluate(str(prov))
    assert res["verdict"] == "PRODUCTION_H7_REVALIDATION_PASS"
