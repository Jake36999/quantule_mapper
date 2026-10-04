"""Stale-flagging of runs after a stepper fix (tools/stepper_staleness.py, jax_scout/provenance.py).

The labels are descriptive -- nothing here may ever block a run -- but they must be RIGHT in the safe
direction: a run that used a fixed stepper on pre-fix code must never come out CURRENT.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import stepper_staleness as stal  # noqa: E402
from jax_scout import provenance  # noqa: E402

FIX = {"id": "etdrk4-2026-10", "fix_commit": "e270cdc", "fix_date": "2026-10-02",
       "affects_steppers": ["ETDRK4", "ETDRK4-cupy"], "revalidated": {"OLD_RUN": "NEW_RUN"},
       "not_affected": {"EXEMPT_RUN": "reason given"}}


def test_recorded_steppers_win():
    s = {"provenance": {"steppers": ["TG-RK4"], "harness": "feb_astar_confirm.py"}}
    assert stal.resolve_steppers(s, "dissipative-S-NCGL") == (["TG-RK4"], "recorded")


def test_flat_stamp_steppers_are_read():
    assert stal.resolve_steppers({"steppers": "ETDRK4,KG-strang"}, "x") == (["ETDRK4", "KG-strang"], "recorded")


def test_harness_import_scan_finds_etdrk4():
    st, src = stal.resolve_steppers({"provenance": {"harness": "feb_astar_confirm.py"}}, "unclassified")
    assert src == "harness" and "ETDRK4" in st


def test_harness_import_scan_kg_harness_has_no_etdrk4():
    st, src = stal.resolve_steppers({"provenance": {"harness": "phase_d_c3_wave.py"}}, "unclassified")
    assert "ETDRK4" not in st


def test_substrate_and_runid_fallbacks():
    assert stal.resolve_steppers({}, "KG-conservative") == (["KG-strang"], "substrate")
    assert stal.resolve_steppers({}, "phase-D-transport", "PHASE_D_C2_SCOUT_1") == (["ETDRK4"], "run_id")
    assert stal.resolve_steppers({}, "phase-D-transport", "PHASE_D_C3_QBALL") == (["KG-strang"], "run_id")
    assert stal.resolve_steppers({}, "mystery", "x") == ([], "unknown")


def test_fix_does_not_apply_to_other_steppers():
    assert stal.staleness("R", ["KG-strang", "TG-RK4"], "dda7bd2", "2026-07-01", FIX) is None


def test_pre_fix_commit_is_stale_and_post_fix_is_current():
    stale = stal.staleness("R", ["ETDRK4"], "dda7bd2", "2026-10-01", FIX)
    assert stale[0] == "STALE_PENDING_REVALIDATION" and "ancestry" in stale[1]
    cur = stal.staleness("R", ["ETDRK4"], "e270cdc", "2026-10-02", FIX)
    assert cur[0] == "CURRENT"


def test_unplaceable_epoch_falls_back_to_date_conservatively():
    assert stal.staleness("R", ["ETDRK4"], "pre-clean-slate", "2026-06-01", FIX)[0] == "STALE_PENDING_REVALIDATION"
    # same day as the fix: cannot tell which side, so stay stale
    assert stal.staleness("R", ["ETDRK4"], "unknown", "2026-10-02", FIX)[0] == "STALE_PENDING_REVALIDATION"
    assert stal.staleness("R", ["ETDRK4"], "unknown", "2026-10-03", FIX)[0] == "CURRENT"


def test_revalidated_and_exempt_runs():
    assert stal.staleness("OLD_RUN", ["ETDRK4"], "dda7bd2", "2026-07-01", FIX)[0] == "REVALIDATED"
    assert stal.staleness("EXEMPT_RUN", ["ETDRK4"], "dda7bd2", "2026-07-01", FIX)[0] == "NOT_AFFECTED"


def test_registry_parses_and_names_e270cdc():
    fixes = stal.load_fixes()
    assert any(f["fix_commit"] == "e270cdc" for f in fixes)


def test_provenance_stamp_records_loaded_steppers():
    import importlib
    importlib.import_module("jax_scout.provenance")
    sys.modules.setdefault("jax_scout.phase_d_c3_wave", type(sys)("fake"))
    try:
        assert "KG-strang" in provenance.stamp()["steppers"]
        assert "KG-strang" in provenance.flat_stamp()["steppers"]
    finally:
        if getattr(sys.modules.get("jax_scout.phase_d_c3_wave"), "__name__", "") == "fake":
            del sys.modules["jax_scout.phase_d_c3_wave"]
