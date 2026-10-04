"""mcp_server.research_tools (Phase E2): read tools over a synthetic project tree, and the fences on the
two write tools. No MCP SDK needed -- the server only wraps these functions."""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from mcp_server import research_tools as rt  # noqa: E402
from jax_scout.snapshots import TelemetryWriter  # noqa: E402

SPEC = json.load(open(os.path.join(ROOT, "specs", "approved", "astar-probe-pilot.json"), encoding="utf-8"))


@pytest.fixture()
def proj(tmp_path):
    """A tiny project: index with 2 runs (one stale), one run dir with telemetry, registry files."""
    import build_results_index as bri
    (tmp_path / "docs" / "runs").mkdir(parents=True)
    (tmp_path / "docs" / "registry").mkdir(parents=True)
    db = tmp_path / "docs" / "runs" / "_index.sqlite"
    con = sqlite3.connect(str(db))
    con.executescript(bri.SCHEMA)
    row = lambda rid, date, sub, st: (rid, date, "b", "fam", sub, "Branch - Stability - Index", "feb_x", None,  # noqa: E731
                                      "abc", "recorded", None, 1, "summary.json", 1.0, 0, 0, 0, None, "", st, "recorded")
    con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row("OLD_RUN", "2026-07-01", "dissipative-S-NCGL", "ETDRK4"))
    con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row("TG_RUN", "2026-09-01", "TG-dual-substrate", "KG-strang,TG-RK4"))
    con.execute("INSERT INTO component_fixes VALUES ('etdrk4-2026-10','ETDRK4','e270cdc','2026-10-02','ETDRK4','d','s')")
    con.execute("INSERT INTO run_staleness VALUES ('OLD_RUN','etdrk4-2026-10','STALE_PENDING_REVALIDATION','old')")
    con.execute("INSERT INTO run_params VALUES ('OLD_RUN','param_a',0.55,NULL)")
    con.commit()
    con.close()
    run = tmp_path / "sweep_runs" / "TG_RUN"
    tel = TelemetryWriter(str(run / "telemetry" / "well"), invariants={"r": 1e-3})
    for i in range(5):
        tel.record(i, r=1e-6 if i < 4 else 1e-2)
    tel.close()
    (run / "summary.json").write_text(json.dumps({"verdict": "X"}))
    (tmp_path / "docs" / "registry" / "harness_registry.json").write_text(json.dumps(
        {"harnesses": [{"id": "feb_x", "status": "ACTIVE"}, {"id": "old", "status": "UNREVIEWED"}]}))
    (tmp_path / "docs" / "registry" / "components.json").write_text(json.dumps(
        {"substrates": [{"name": "etdrk4-sncgl"}], "ics": [{"name": "multiseed"}],
         "observers": [{"name": "energy_ratio"}, {"name": "nodes"}]}))
    (tmp_path / "specs" / "approved").mkdir(parents=True)
    (tmp_path / "specs" / "approved" / "astar-probe-pilot.json").write_text(json.dumps(SPEC))
    import shutil
    shutil.copytree(os.path.join(ROOT, "irer_specs"), tmp_path / "irer_specs")
    return str(tmp_path)


def test_list_runs_filters(proj):
    assert rt.list_runs(proj)["n"] == 2
    assert [r["run_id"] for r in rt.list_runs(proj, stepper="TG-RK4")["runs"]] == ["TG_RUN"]
    assert [r["run_id"] for r in rt.list_runs(proj, stale_only=True)["runs"]] == ["OLD_RUN"]


def test_get_run_and_staleness(proj):
    r = rt.get_run(proj, "OLD_RUN")
    assert r["params"]["param_a"] == 0.55 and r["staleness"][0]["status"] == "STALE_PENDING_REVALIDATION"
    assert rt.get_run(proj, "NOPE")["status"] == "NOT_FOUND"
    assert rt.get_run(proj, "../etc/passwd")["status"] == "BAD_ID"


def test_tail_telemetry_reports_breaches(proj):
    t = rt.tail_telemetry(proj, "TG_RUN", n=2)
    arm = t["arms"]["telemetry/well"]
    assert arm["n_samples"] == 5 and len(arm["tail"]) == 2 and arm["breaches"][0]["key"] == "r"


def test_stale_and_registry_and_components(proj):
    assert rt.list_stale_runs(proj)["n"] == 1
    assert rt.harness_registry(proj, "ACTIVE")["n"] == 1
    assert rt.list_components(proj)["status"] == "OK"


def test_missing_index_is_a_status_not_an_exception(tmp_path):
    assert rt.list_runs(str(tmp_path))["status"] == "NO_INDEX"


def test_propose_spec_writes_only_to_proposed_and_never_overwrites(proj):
    s = dict(SPEC, id="agent-idea-1")
    r = rt.propose_spec(proj, s, author="test")
    assert r["status"] == "PROPOSED" and r["path"] == "specs/proposed/agent-idea-1.json"
    assert rt.propose_spec(proj, s)["status"] == "EXISTS"
    assert rt.propose_spec(proj, dict(SPEC))["status"] == "EXISTS"           # id already approved
    assert os.path.exists(os.path.join(proj, "runtime_logs", "research_tools_audit.jsonl"))


def test_propose_spec_rejects_invalid_and_unknown_components(proj):
    bad = dict(SPEC, id="agent-bad")
    bad.pop("prediction")
    assert rt.propose_spec(proj, bad)["status"] == "INVALID"
    unk = json.loads(json.dumps(SPEC))
    unk["id"] = "agent-unknown"
    unk["substrate"]["name"] = "warp-drive"
    r = rt.propose_spec(proj, unk)
    assert r["status"] == "INVALID" and any("warp-drive" in e for e in r["errors"])


def test_propose_spec_cannot_escape_the_folder(proj):
    s = dict(SPEC, id="../../evil")
    assert rt.propose_spec(proj, s)["status"] == "INVALID"                  # schema id pattern


def test_draft_reading_is_fenced_and_labelled(proj):
    r = rt.draft_reading(proj, "TG_RUN", "the drift looks like box contamination", author="qwen")
    assert r["status"] == "DRAFTED" and r["path"].startswith("docs/runs/_machine_drafts/")
    text = open(os.path.join(proj, r["path"]), encoding="utf-8").read()
    assert "MACHINE-DRAFTED" in text and "qwen" in text
    assert rt.draft_reading(proj, "NOT_A_RUN", "x")["status"] == "NOT_FOUND"
    assert rt.draft_reading(proj, "../x", "x")["status"] == "BAD_INPUT"


def test_there_is_no_launch_tool():
    names = [n for n in dir(rt) if not n.startswith("_")]
    assert not any(w in n for n in names for w in ("launch", "run_spec", "start", "execute", "submit"))
