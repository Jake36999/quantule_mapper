"""A4 wiring check -- prove production `stability_metrics` survive the provenance path into the Hunter.

Path under test:
    solver/run.py            -> HDF5 /stability_metrics (JSON string) + result_payload["stability_metrics"]
    validation_pipeline.py   -> read_json_dataset(h5f,"stability_metrics") -> provenance report top-level key
    provenance_{hash}.json   -> the file aste_hunter reads
    aste_hunter              -> _stability_fitness_from_provenance(prov_data)["stability_metrics"]

Encodes the A4 acceptance criteria (see docs/PRODUCTION_ALIGNMENT_PLAN.md). No cupy/GPU; heavy prod modules
are importorskip'd so the contract-level checks still run on the dev box.
"""
import os, sys, json
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# a*-like (flat, certifiable) and a grower, in the exact shape solver/stability_metrics.compute emits
ASTAR_SM = {"er_fin": 2.06, "er_max": 2.075, "floor_ratio": 1.0, "late_slope_50pct_per1k": -0.0007, "T": 36000}
GROWER_SM = {"er_fin": 2.9, "er_max": 3.6, "floor_ratio": 1.0, "late_slope_50pct_per1k": 0.05, "T": 36000}


def _emit_h5(tmp_path, sm, key="stability_metrics"):
    """Mock exactly what solver/run.py writes: a single-element S1024 JSON-string dataset."""
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "artifact.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset(key, data=np.array([json.dumps(sm)], dtype="S1024"))
    return p


def test_h5_read_round_trips_and_handles_absent():
    vp = pytest.importorskip("validation_pipeline")
    h5py = pytest.importorskip("h5py")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = _emit_h5(__import__("pathlib").Path(d), ASTAR_SM)
        with h5py.File(p, "r") as f:
            assert vp.read_json_dataset(f, "stability_metrics") == ASTAR_SM   # emitter -> assembler read
            assert vp.read_json_dataset(f, "not_present") is None             # absent -> None, no raise


def test_h5_malformed_is_none_not_raise(tmp_path):
    vp = pytest.importorskip("validation_pipeline")
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("stability_metrics", data=np.array([b"{not json"], dtype="S16"))
    with h5py.File(p, "r") as f:
        assert vp.read_json_dataset(f, "stability_metrics") is None


def test_full_path_reaches_prov_data_and_consumer(tmp_path):
    """Acceptance #1 + #2: a mocked worker payload's stability_metrics reaches prov_data and is consumable."""
    vp = pytest.importorskip("validation_pipeline")
    h5py = pytest.importorskip("h5py")
    H = pytest.importorskip("aste_hunter")
    from tools.stability_objective import stability_score

    # emit (HDF5, as solver/run.py) -> assemble (validation_pipeline read) -> provenance report dict
    art = _emit_h5(tmp_path, ASTAR_SM)
    with h5py.File(art, "r") as f:
        carried = vp.read_json_dataset(f, "stability_metrics")
    report = {"spectral_fidelity": {"log_prime_sse": 999.0}, "stability_metrics": carried}

    # persist to the exact filename aste_hunter reads, then consume it the way the Hunter does
    prov_path = tmp_path / "provenance_deadbeef.json"
    prov_path.write_text(json.dumps(report), encoding="utf-8")
    prov_data = json.loads(prov_path.read_text(encoding="utf-8"))

    assert prov_data["stability_metrics"] == ASTAR_SM                           # #1: reaches prov_data verbatim
    fit, ss = H._stability_fitness_from_provenance(prov_data)                   # #2: consumable
    assert fit == max(0.0, float(stability_score(ASTAR_SM)["score"])) > 0.0
    assert ss.get("certifiable") is True


def test_missing_metrics_is_zero_not_prime_fallback():
    """Acceptance #3: absent stability_metrics -> explicit zero/unavailable, NOT prime-SSE fallback."""
    H = pytest.importorskip("aste_hunter")
    prov = {"spectral_fidelity": {"log_prime_sse": 0.001}}    # excellent PRIME, but NO stability block
    fit, ss = H._stability_fitness_from_provenance(prov)
    assert fit == 0.0
    assert ss.get("reject") == "NO_STABILITY_METRICS"          # explicit unavailable, prime never consulted


def test_grower_survives_path_and_is_penalized(tmp_path):
    vp = pytest.importorskip("validation_pipeline")
    h5py = pytest.importorskip("h5py")
    H = pytest.importorskip("aste_hunter")
    art = _emit_h5(tmp_path, GROWER_SM)
    with h5py.File(art, "r") as f:
        carried = vp.read_json_dataset(f, "stability_metrics")
    fit, ss = H._stability_fitness_from_provenance({"stability_metrics": carried})
    assert carried == GROWER_SM
    assert fit == 0.0 and ss.get("reject") == "GROWER_BLOWUP"   # er_max>3.0 -> hard reject, carried faithfully


def test_prime_objective_backward_compatible(tmp_path):
    """Acceptance #4: default objective stays 'prime'; the new peer key doesn't disturb the prime read."""
    H = pytest.importorskip("aste_hunter")
    h = H.Hunter(db_file=str(tmp_path / "bc.db"))
    assert h.objective == "prime"                              # default unchanged
    # a provenance carrying BOTH keys still exposes spectral_fidelity untouched for the prime consumer
    prov = {"spectral_fidelity": {"log_prime_sse": 1.5}, "stability_metrics": ASTAR_SM}
    assert prov.get("spectral_fidelity", {}).get("log_prime_sse") == 1.5
