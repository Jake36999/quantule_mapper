"""Run gallery + viewer (tools/viewer_data.py, tools/serve_viewer.py, tools/render_queue_worker.py).

Runs on a synthetic sweep_runs/ tree; no GPU and no JAX needed. The key guarantees:
  * a field is served downsampled, never loaded in full when only headers are needed;
  * suggested windows mark behaviour CHANGES (curvature), not steady drift;
  * the page can only QUEUE a re-run; nothing in the server runs a simulation.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import viewer_data as vd  # noqa: E402


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep_runs"
    run = sweep / "FEB_TEST_RUN"
    run.mkdir(parents=True)
    rng = np.random.default_rng(0)
    psi = (rng.standard_normal((32, 32, 32)) + 1j * rng.standard_normal((32, 32, 32))).astype(np.complex128)
    er = np.concatenate([np.linspace(1, 2, 200), 2 - 1e-4 * np.arange(1800)]).astype(np.float32)
    np.savez(run / "a1.15_cell_probe.npz", psi_fin=psi, er=er)
    json.dump({"N": 32, "K": 6, "seed": 20260619, "rows": [{"key": "a1.15_cell", "a_factor": 1.15, "T": 2000,
               "seed": 20260619, "er_max": 2.0}]}, open(run / "feb_test_summary.json", "w"))
    monkeypatch.setattr(vd, "SWEEP", str(sweep))
    monkeypatch.setattr(vd, "QUEUE", str(tmp_path / "specs" / "queue"))
    return tmp_path


def test_npz_headers_are_read_without_loading(tree):
    shapes = vd._npz_shapes(str(tree / "sweep_runs" / "FEB_TEST_RUN" / "a1.15_cell_probe.npz"))
    assert shapes["psi_fin"] == ((32, 32, 32), "complex128") and shapes["er"][0] == (2000,)


def test_run_detail_finds_fields_series_and_rerun(tree):
    d = vd.run_detail("FEB_TEST_RUN")
    assert [f["key"] for f in d["fields"]] == ["psi_fin"] and d["fields"][0]["complex"]
    assert [s["key"] for s in d["series"]] == ["er"] and d["rerunnable"]


@pytest.mark.parametrize("q", vd.QUANTITIES)
def test_field_volume_downsamples_and_reduces(tree, q):
    hdr, raw = vd.field_volume("FEB_TEST_RUN", "a1.15_cell_probe.npz", "psi_fin", q, target=16)
    a = np.frombuffer(raw, dtype=np.float32)
    assert hdr["shape"] == [16, 16, 16] and hdr["full_n"] == 32 and a.size == 16 ** 3
    assert np.isclose(a.min(), hdr["min"]) and np.isclose(a.max(), hdr["max"])


def test_paths_cannot_escape_the_run(tree):
    with pytest.raises(FileNotFoundError):
        vd.field_volume("FEB_TEST_RUN", "../../etc/passwd", "x")
    with pytest.raises(ValueError):
        vd.run_detail("../FEB_TEST_RUN")


def test_windows_mark_behaviour_changes_not_steady_drift():
    t = np.linspace(0, 100, 5000)
    assert vd.activity_windows(t, 2 - 0.01 * t) == []                                   # pure drift
    y = 2 - 0.01 * t + np.where((t > 40) & (t < 55), 0.2 * np.sin(2 * np.pi * t / 2), 0)
    w = vd.activity_windows(t, y)
    assert len(w) == 1 and w[0]["t0"] < 41 and w[0]["t1"] > 54                          # one window, the episode


def test_window_rerun_stops_at_the_window_end(tree):
    r = vd.rerun_spec("FEB_TEST_RUN", "a1.15_cell", record={"every": 0.05, "window": [1.0, 2.0], "volume": 16})
    pr = r["spec"]["protocol"]
    assert r["status"] == "OK" and abs(pr["T"] - 2.0) < 1e-9 and pr["record"]["window"] == [1.0, 2.0]
    assert abs(r["spec"]["substrate"]["params"]["param_a"] - 0.4802 * 1.15) < 1e-12
    e = vd.estimate(r["spec"])
    assert e["steps"] == 400 and e["frames"] == 21


def test_queue_writes_a_job_file_and_runs_nothing(tree):
    spec = vd.rerun_spec("FEB_TEST_RUN", "a1.15_cell", record={"every": 0.5})["spec"]
    out = vd.queue_render(spec)
    assert out["status"] == "QUEUED"
    jobs = vd.queue_status()["jobs"]
    assert len(jobs) == 1 and jobs[0]["state"] == "queued"
    assert not (tree / "sweep_runs" / spec["id"].upper()).exists()


def test_server_serves_page_and_has_no_run_endpoint(tree):
    import serve_viewer
    srv = serve_viewer.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % srv.server_address[1]
    try:
        html = urllib.request.urlopen(base + "/").read()
        assert b"Run viewer" in html and b"render_queue_worker" in html
        for path in ("/api/run_now", "/api/launch", "/api/start"):
            req = urllib.request.Request(base + path, data=b"{}", headers={"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req)
            assert e.value.code == 404
        r = json.loads(urllib.request.urlopen(base + "/api/field?id=FEB_TEST_RUN&file=a1.15_cell_probe.npz"
                                              "&key=psi_fin&quantity=phase&n=8").read())
        assert r["shape"] == [8, 8, 8] and len(r["data"]) > 0
    finally:
        srv.shutdown()


def test_web_pages_are_not_gitignored():
    """Regression: `UI/*` in .gitignore matched `ui/` case-insensitively on Windows, so the spec editor
    page was never committed (found 2026-10-07). The pages now live in web/; make sure git sees them."""
    import subprocess
    for page in ("web/viewer/index.html", "web/spec_editor/index.html"):
        assert os.path.exists(os.path.join(ROOT, page)), page
        r = subprocess.run(["git", "check-ignore", "-q", page], cwd=ROOT)
        assert r.returncode == 1, "%s is git-ignored" % page


def test_numbered_snapshot_packs_become_one_group(tree):
    """sample000/010/020 packs in one folder are ONE experiment over time, not three unrelated tabs."""
    sd = tree / "sweep_runs" / "FEB_TEST_RUN" / "source_snapshots"
    sd.mkdir()
    for i, n in enumerate((0, 10, 20)):
        np.savez(sd / ("two_packet_sample%03d.npz" % n), t=np.float64(0.5 * i),
                 rho=np.full((8, 8, 8), 1.0 + i), phi=np.ones((8, 8, 8), complex) * (i + 1))
    d = vd.run_detail("FEB_TEST_RUN")
    g = [x for x in d["groups"] if x["group"].endswith("two_packet")]
    assert len(g) == 1 and g[0]["t"] == [0.0, 0.5, 1.0] and set(g[0]["fields"]) == {"rho", "phi"}
    assert not any(f["file"].startswith("source_snapshots/") for f in d["fields"])
    s = vd.group_series("FEB_TEST_RUN", g[0]["group"], "abs2")
    assert s["stats"]["rho"]["mean"] == [1.0, 2.0, 3.0]   # real fields are shown as-is (rho is already a density)
    assert s["stats"]["phi"]["max"] == [1.0, 4.0, 9.0]


def test_screening_runs_are_flagged_by_precision(tmp_path):
    def spec(d, body):
        d.mkdir()
        (d / "spec.json").write_text(json.dumps(body))
        return str(d / "spec.json")
    assert vd._precision(spec(tmp_path / "a", {"protocol": {}})) is None
    assert vd._precision(spec(tmp_path / "b", {"protocol": {"precision": "fp32"}})) == "fp32"
    assert vd._precision(spec(tmp_path / "c", {"protocol": {},
                                               "sweep": {"axes": {"protocol.precision": ["fp64", "fp32"]}}})) == "mixed"
    assert vd._precision(str(tmp_path / "missing.json")) is None
