"""tools/serve_spec_ui.py (Phase E3): HTTP round trip on a random local port. No browser needed."""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import serve_spec_ui as ui  # noqa: E402
import irer_specs  # noqa: E402


@pytest.fixture()
def server(tmp_path, monkeypatch):
    for d in ("drafts", "proposed", "approved"):
        (tmp_path / "specs" / d).mkdir(parents=True)
    (tmp_path / "docs" / "registry").mkdir(parents=True)
    shutil.copy(os.path.join(ROOT, "docs", "registry", "components.json"), tmp_path / "docs" / "registry")
    monkeypatch.setattr(ui, "ROOT", str(tmp_path))
    monkeypatch.setattr(ui.rt, "list_components", lambda root: json.load(
        open(tmp_path / "docs" / "registry" / "components.json")) | {"status": "OK"})
    srv = ui.make_server(0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:%d" % srv.server_address[1], tmp_path
    srv.shutdown()


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"{}") if "json" in r.headers.get("Content-Type", "") else r.read()
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


SPEC = irer_specs.load(os.path.join(ROOT, "specs", "approved", "astar-probe-pilot.json"))


def test_page_and_schema_are_served(server):
    base, _ = server
    code, html = call(base, "/")
    assert code == 200 and b"Experiment spec editor" in html and b"cannot launch" in html.lower()
    assert call(base, "/api/schema")[1]["title"] == "IRER experiment spec"


def test_form_defaults_are_pruned_before_validation(server):
    base, _ = server
    s = json.loads(json.dumps(SPEC))
    s["sweep"] = {"axes": {}}                     # what react-jsonschema-form sends for an untouched sweep
    s["invariants"] = {}
    assert call(base, "/api/validate", {"spec": s})[1]["status"] == "VALID"


def test_save_then_promote_round_trip(server):
    base, tmp = server
    s = dict(SPEC, id="ui-roundtrip")
    code, out = call(base, "/api/save", {"spec": s})
    assert code == 200 and (tmp / "specs" / "drafts" / "ui-roundtrip.json").exists()
    code, out = call(base, "/api/promote", {"id": "ui-roundtrip", "from": "drafts"})
    assert code == 200 and out["run_command"].endswith("specs/approved/ui-roundtrip.json")
    assert not (tmp / "specs" / "drafts" / "ui-roundtrip.json").exists()
    assert call(base, "/api/save", {"spec": s})[0] == 409            # id now approved


def test_invalid_spec_is_rejected_and_nothing_written(server):
    base, tmp = server
    bad = dict(SPEC, id="ui-bad")
    bad.pop("prediction")
    code, out = call(base, "/api/save", {"spec": bad})
    assert code == 400 and not list((tmp / "specs" / "drafts").iterdir())


def test_no_endpoint_launches_anything(server):
    base, _ = server
    for path in ("/api/run", "/api/launch", "/api/start"):
        assert call(base, path, {"id": "x"})[0] == 404
