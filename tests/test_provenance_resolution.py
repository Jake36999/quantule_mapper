"""A4b -- provenance filename contract reconciliation.

The writer (validation_pipeline) folds seed/run_id/utc into the provenance filename via
run_identity.provenance_path_for_artifact when the artifact carries an /identity group; the reader
(aste_hunter) historically read the plain provenance_{config_hash}.json. run_identity.resolve_provenance_report
is now the single shared resolver, and aste_hunter uses it for BOTH objectives. These tests prove the seam is
closed and backward-compatible. No cupy/GPU; aste_hunter is importorskip'd.
"""
import os, sys, json, sqlite3
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ASTAR_SM = {"er_fin": 2.06, "er_max": 2.075, "floor_ratio": 1.0, "late_slope_50pct_per1k": -0.0007, "T": 36000}


def _folded_name(ri, config_hash):
    """The exact seed/run_id-folded filename the writer would produce for an identity-stamped artifact."""
    name = ri.provenance_filename(config_hash, seed=620, run_id="abcd1234ef", utc_date="2026-07-03")
    assert name != f"provenance_{config_hash}.json"     # sanity: it really differs from the plain name
    return name


# ---- resolver-level (criteria 1, 2, 5) ------------------------------------------------------------
def test_plain_legacy_path_resolves(tmp_path):
    ri = pytest.importorskip("orchestrator.run_identity")
    ch = "aa11bb22cc33"
    p = tmp_path / f"provenance_{ch}.json"; p.write_text("{}", encoding="utf-8")
    got = ri.resolve_provenance_report(str(tmp_path), ch)
    assert got is not None and os.path.samefile(got, str(p))


def test_identity_folded_path_is_discoverable(tmp_path):
    ri = pytest.importorskip("orchestrator.run_identity")
    ch = "dd44ee55ff66"
    p = tmp_path / _folded_name(ri, ch); p.write_text("{}", encoding="utf-8")     # only the folded file exists
    got = ri.resolve_provenance_report(str(tmp_path), ch)
    assert got is not None and os.path.samefile(got, str(p))


def test_missing_provenance_returns_none(tmp_path):
    ri = pytest.importorskip("orchestrator.run_identity")
    assert ri.resolve_provenance_report(str(tmp_path), "ghosthash") is None


def test_no_hash_collision_across_configs(tmp_path):
    # a different config_hash's folded report must NOT be returned for our hash
    ri = pytest.importorskip("orchestrator.run_identity")
    (tmp_path / _folded_name(ri, "otherhash999")).write_text("{}", encoding="utf-8")
    assert ri.resolve_provenance_report(str(tmp_path), "ourhash111") is None


# ---- Hunter e2e (criteria 3, 4, 5) ---------------------------------------------------------------
def _insert_run(db_file, config_hash):
    c = sqlite3.connect(db_file)
    c.execute("INSERT OR REPLACE INTO runs (config_hash, seed, status, generation) VALUES (?,0,'pending',0)",
              (config_hash,))
    c.commit(); c.close()


def _read_run(db_file, config_hash):
    c = sqlite3.connect(db_file); c.row_factory = sqlite3.Row
    row = c.execute("SELECT status, fitness FROM runs WHERE config_hash=?", (config_hash,)).fetchone()
    c.close(); return dict(row) if row else None


def test_hunter_stability_reads_stability_metrics_from_folded(tmp_path):
    """Criterion 3: stability mode consumes stability_metrics from the resolved (folded) provenance file."""
    H = pytest.importorskip("aste_hunter")
    ri = pytest.importorskip("orchestrator.run_identity")
    prov_dir = tmp_path / "prov"; prov_dir.mkdir()
    ch = "cafe12345678"
    h = H.Hunter(db_file=str(tmp_path / "led.db"), objective="stability")
    _insert_run(h.db_file, ch)
    (prov_dir / _folded_name(ri, ch)).write_text(
        json.dumps({"spectral_fidelity": {"log_prime_sse": 999.0}, "stability_metrics": ASTAR_SM}), encoding="utf-8")
    h.process_generation_results(str(prov_dir), [ch])
    row = _read_run(h.db_file, ch)
    assert row["status"] == "completed"
    assert row["fitness"] is not None and row["fitness"] > 0.0     # a* stability score reached fitness


def test_hunter_prime_reads_spectral_from_folded(tmp_path):
    """Criterion 4: default (prime) mode still reads spectral_fidelity from the resolved (folded) file."""
    H = pytest.importorskip("aste_hunter")
    ri = pytest.importorskip("orchestrator.run_identity")
    prov_dir = tmp_path / "prov"; prov_dir.mkdir()
    ch = "abc123def456"
    h = H.Hunter(db_file=str(tmp_path / "led.db"))
    assert h.objective == "prime"                                  # default unchanged
    _insert_run(h.db_file, ch)
    # fast_energy_ratio < 4.0 -> prefilter rejects -> clean status='completed', fitness 0.0 (no crash)
    (prov_dir / _folded_name(ri, ch)).write_text(
        json.dumps({"spectral_fidelity": {"log_prime_sse": 999.0, "fast_energy_ratio": 1.0},
                    "aletheia_metrics": {}, "stability_metrics": ASTAR_SM}), encoding="utf-8")
    h.process_generation_results(str(prov_dir), [ch])
    row = _read_run(h.db_file, ch)
    assert row["status"] == "completed"       # resolver found the folded file; prime path read spectral_fidelity


def test_hunter_missing_provenance_fails_safely(tmp_path):
    """Criterion 5: no provenance file -> safe failure (status='failed'), not a crash."""
    H = pytest.importorskip("aste_hunter")
    prov_dir = tmp_path / "prov"; prov_dir.mkdir()
    ch = "nofilehash00"
    h = H.Hunter(db_file=str(tmp_path / "led.db"), objective="stability")
    _insert_run(h.db_file, ch)
    h.process_generation_results(str(prov_dir), [ch])
    assert _read_run(h.db_file, ch)["status"] == "failed"
