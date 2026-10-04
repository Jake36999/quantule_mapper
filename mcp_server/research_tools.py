"""mcp_server.research_tools — agent-facing tools over the CURRENT research platform (Phase E2).

The older tools in this package (data_access / write_tools) read the legacy orchestrator ledger
(simulation_ledger.db, config_hash runs). These read what the project uses now:

    docs/runs/_index.sqlite          results index (tools/build_results_index.py)
    sweep_runs/<run>/telemetry*.jsonl live invariants (jax_scout.snapshots.TelemetryWriter)
    specs/{drafts,proposed,approved} experiment specs (irer_specs)
    docs/registry/*.json             harness registry, component fixes, component list

THE BOUNDARY. Every tool is READ-ONLY except two, and both write only into fenced folders:
    propose_spec   -> specs/proposed/<id>.json   (validated; never overwrites; never launches)
    draft_reading  -> docs/runs/_machine_drafts/<run_id>.md   (labelled MACHINE-DRAFTED)
There is deliberately NO tool that launches a run. A human promotes a proposed spec to approved/ and
runs tools/run_spec.py. Same rule as the HUD: no control path into the simulation.

No MCP SDK, GPU or JAX dependency: importable and testable anywhere.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,200}$")


def _paths(root: str) -> dict:
    return {
        "index": os.path.join(root, "docs", "runs", "_index.sqlite"),
        "sweep": os.path.join(root, "sweep_runs"),
        "specs": os.path.join(root, "specs"),
        "registry": os.path.join(root, "docs", "registry"),
        "drafts": os.path.join(root, "docs", "runs", "_machine_drafts"),
        "audit": os.path.join(root, "runtime_logs", "research_tools_audit.jsonl"),
    }


def _ro(db: str) -> Optional[sqlite3.Connection]:
    if not os.path.exists(db):
        return None
    con = sqlite3.connect("file:%s?mode=ro" % db.replace("\\", "/"), uri=True)
    con.row_factory = sqlite3.Row
    return con


def _audit(root: str, event: dict) -> None:
    p = _paths(root)["audit"]
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"utc": _dt.datetime.utcnow().isoformat() + "Z", **event}) + "\n")
    except OSError:
        pass


def _safe_id(x: str) -> bool:
    return bool(x) and bool(_ID_RE.match(x)) and ".." not in x


def _no_index(root):
    return {"status": "NO_INDEX", "hint": "run: python tools/build_results_index.py",
            "expected": _paths(root)["index"]}


# ================================================================== read-only

def list_runs(root: str, substrate: Optional[str] = None, branch: Optional[str] = None,
              stepper: Optional[str] = None, harness: Optional[str] = None, stale_only: bool = False,
              limit: int = 50) -> Dict[str, Any]:
    """Runs from the results index, newest first, filtered by substrate / branch / stepper / harness."""
    con = _ro(_paths(root)["index"])
    if con is None:
        return _no_index(root)
    q = ("SELECT r.run_id, r.date, r.substrate, r.family, r.branch, r.harness, r.steppers, r.verdict, "
         "r.code_epoch, (SELECT GROUP_CONCAT(s.fix_id || ':' || s.status) FROM run_staleness s "
         " WHERE s.run_id = r.run_id) AS staleness FROM runs r WHERE 1=1")
    args: list = []
    for col, val in (("r.substrate", substrate), ("r.branch", branch), ("r.harness", harness)):
        if val:
            q += " AND %s = ?" % col
            args.append(val)
    if stepper:
        q += " AND (',' || r.steppers || ',') LIKE ?"
        args.append("%%,%s,%%" % stepper)
    if stale_only:
        q += " AND r.run_id IN (SELECT run_id FROM v_stale_runs)"
    q += " ORDER BY r.date DESC, r.run_id DESC LIMIT ?"
    args.append(max(1, min(int(limit), 500)))
    rows = [dict(r) for r in con.execute(q, args)]
    con.close()
    return {"status": "OK", "n": len(rows), "runs": rows}


def get_run(root: str, run_id: str, max_metrics: int = 200) -> Dict[str, Any]:
    """One run: index row, params, metrics (truncated), staleness, edges, and its summary.json."""
    if not _safe_id(run_id):
        return {"status": "BAD_ID"}
    con = _ro(_paths(root)["index"])
    if con is None:
        return _no_index(root)
    row = con.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        con.close()
        return {"status": "NOT_FOUND", "run_id": run_id}
    out = {"status": "OK", "run": dict(row)}
    out["params"] = {r["key"]: (r["num"] if r["num"] is not None else r["txt"])
                     for r in con.execute("SELECT key, num, txt FROM run_params WHERE run_id = ?", (run_id,))}
    out["metrics"] = [dict(r) for r in con.execute(
        "SELECT key, num, txt, row_idx FROM run_metrics WHERE run_id = ? LIMIT ?", (run_id, int(max_metrics)))]
    out["staleness"] = [dict(r) for r in con.execute(
        "SELECT fix_id, status, reason FROM run_staleness WHERE run_id = ?", (run_id,))]
    out["edges"] = [dict(r) for r in con.execute(
        "SELECT * FROM edges WHERE (src_kind='run' AND src_id=?) OR (dst_kind='run' AND dst_id=?)",
        (run_id, run_id))]
    con.close()
    sp = os.path.join(_paths(root)["sweep"], run_id, "summary.json")
    if os.path.exists(sp):
        try:
            text = open(sp, encoding="utf-8").read()
            out["summary_json"] = json.loads(text) if len(text) < 200_000 else {"truncated": True,
                                                                                  "bytes": len(text)}
        except (OSError, ValueError):
            pass
    return out


def tail_telemetry(root: str, run_id: str, n: int = 50) -> Dict[str, Any]:
    """Last n telemetry samples per arm of a run, with declared invariants and every breach so far."""
    if not _safe_id(run_id):
        return {"status": "BAD_ID"}
    run = os.path.join(_paths(root)["sweep"], run_id)
    if not os.path.isdir(run):
        return {"status": "NOT_FOUND", "run_id": run_id}
    from jax_scout.snapshots import read_telemetry, invariant_breaches, TELEMETRY_FILE
    arms = {}
    for dirpath, dirnames, filenames in os.walk(run):
        dirnames[:] = [d for d in dirnames if d not in ("rendered", "snapshots")]
        if TELEMETRY_FILE in filenames:
            meta, rows = read_telemetry(dirpath)
            arm = os.path.relpath(dirpath, run).replace(os.sep, "/")
            arms["." if arm == "." else arm] = {
                "meta": meta, "n_samples": len(rows), "tail": rows[-max(1, min(int(n), 2000)):],
                "breaches": [{"key": k, "t": t, "value": v, "tolerance": tol}
                             for k, t, v, tol in invariant_breaches(meta, rows)][:200]}
    if not arms:
        return {"status": "NO_TELEMETRY", "run_id": run_id}
    return {"status": "OK", "run_id": run_id, "arms": arms}


def list_stale_runs(root: str, fix_id: Optional[str] = None) -> Dict[str, Any]:
    """Runs that used a since-fixed component on pre-fix code (docs/registry/COMPONENT_FIXES.json)."""
    con = _ro(_paths(root)["index"])
    if con is None:
        return _no_index(root)
    q, a = "SELECT * FROM v_stale_runs", []
    if fix_id:
        q += " WHERE fix_id = ?"
        a.append(fix_id)
    rows = [dict(r) for r in con.execute(q + " ORDER BY date", a)]
    fixes = [dict(r) for r in con.execute("SELECT * FROM component_fixes")]
    con.close()
    return {"status": "OK", "n": len(rows), "fixes": fixes, "stale_runs": rows}


def list_components(root: str) -> Dict[str, Any]:
    """Registered substrates, ICs and observers (names, versions, docs, param schemas)."""
    try:
        sys.path.insert(0, root)
        from jax_scout import registry  # needs JAX
        return {"status": "OK", "source": "live", **registry.describe()}
    except Exception:
        p = os.path.join(_paths(root)["registry"], "components.json")
        if os.path.exists(p):
            return {"status": "OK", "source": "docs/registry/components.json",
                    **json.load(open(p, encoding="utf-8"))}
        return {"status": "UNAVAILABLE", "hint": "python -m jax_scout.registry --export (in the JAX env)"}


def get_schema(root: str) -> Dict[str, Any]:
    """The experiment-spec JSON Schema."""
    sys.path.insert(0, root)
    import irer_specs
    return {"status": "OK", "schema": irer_specs.SCHEMA}


def list_specs(root: str, status: Optional[str] = None) -> Dict[str, Any]:
    """Specs by folder: drafts / proposed / approved."""
    out = []
    for st in ("drafts", "proposed", "approved"):
        if status and st != status:
            continue
        for p in sorted(glob.glob(os.path.join(_paths(root)["specs"], st, "*.json"))):
            try:
                s = json.load(open(p, encoding="utf-8"))
            except (OSError, ValueError):
                out.append({"status": st, "file": os.path.basename(p), "error": "unreadable"})
                continue
            out.append({"status": st, "id": s.get("id"), "title": s.get("title"),
                        "substrate": (s.get("substrate") or {}).get("name"),
                        "prediction": (s.get("prediction") or {}).get("statement")})
    return {"status": "OK", "n": len(out), "specs": out}


def get_spec(root: str, spec_id: str) -> Dict[str, Any]:
    if not _safe_id(spec_id):
        return {"status": "BAD_ID"}
    for st in ("approved", "proposed", "drafts"):
        p = os.path.join(_paths(root)["specs"], st, spec_id + ".json")
        if os.path.exists(p):
            return {"status": "OK", "folder": st, "spec": json.load(open(p, encoding="utf-8"))}
    return {"status": "NOT_FOUND", "spec_id": spec_id}


def harness_registry(root: str, status: Optional[str] = None) -> Dict[str, Any]:
    p = os.path.join(_paths(root)["registry"], "harness_registry.json")
    if not os.path.exists(p):
        return {"status": "UNAVAILABLE", "hint": "python tools/build_harness_registry.py"}
    hs = json.load(open(p, encoding="utf-8")).get("harnesses", [])
    if status:
        hs = [h for h in hs if h.get("status") == status]
    return {"status": "OK", "n": len(hs), "harnesses": hs}


# ================================================================== write-limited

def propose_spec(root: str, spec: Dict[str, Any], author: str = "agent") -> Dict[str, Any]:
    """Validate a spec and file it under specs/proposed/<id>.json. Never overwrites, never launches.
    A human reviews it, moves it to specs/approved/, and runs tools/run_spec.py."""
    sys.path.insert(0, root)
    import irer_specs
    if not isinstance(spec, dict):
        return {"status": "INVALID", "errors": ["spec must be a JSON object"]}
    errs = irer_specs.validate(spec)
    if not errs:
        comps = list_components(root)
        if comps.get("status") == "OK":
            known = {k: {r["name"] for r in comps[k]} for k in ("substrates", "ics", "observers")}
            if spec["substrate"]["name"] not in known["substrates"]:
                errs.append("unknown substrate '%s'" % spec["substrate"]["name"])
            if spec["protocol"]["ic"]["name"] not in known["ics"]:
                errs.append("unknown ic '%s'" % spec["protocol"]["ic"]["name"])
            errs += ["unknown observer '%s'" % o["name"] for o in spec["observers"]
                     if o["name"] not in known["observers"]]
    if errs:
        return {"status": "INVALID", "errors": errs}
    sid = spec["id"]
    for st in ("drafts", "proposed", "approved"):
        if os.path.exists(os.path.join(_paths(root)["specs"], st, sid + ".json")):
            return {"status": "EXISTS", "errors": ["spec id '%s' already exists in specs/%s/" % (sid, st)]}
    target_dir = os.path.join(_paths(root)["specs"], "proposed")
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, sid + ".json")
    if os.path.commonpath([os.path.abspath(target), os.path.abspath(target_dir)]) != os.path.abspath(target_dir):
        return {"status": "BAD_ID"}
    with open(target, "x", encoding="utf-8") as fh:          # 'x': fail rather than overwrite
        json.dump(spec, fh, indent=2)
        fh.write("\n")
    _audit(root, {"event": "propose_spec", "spec_id": sid, "author": author})
    return {"status": "PROPOSED", "path": os.path.relpath(target, root).replace(os.sep, "/"),
            "next": "a human reviews it, moves it to specs/approved/, then runs tools/run_spec.py"}


def draft_reading(root: str, run_id: str, text: str, author: str = "agent") -> Dict[str, Any]:
    """Append a MACHINE-DRAFTED reading of a run to docs/runs/_machine_drafts/<run_id>.md. It never
    edits the human paired-reading blocks; a reviewer copies what survives review."""
    if not _safe_id(run_id) or not isinstance(text, str) or not text.strip():
        return {"status": "BAD_INPUT"}
    if not os.path.isdir(os.path.join(_paths(root)["sweep"], run_id)):
        con = _ro(_paths(root)["index"])
        known = con is not None and con.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if con is not None:
            con.close()
        if not known:
            return {"status": "NOT_FOUND", "run_id": run_id}
    d = _paths(root)["drafts"]
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, run_id + ".md")
    new = not os.path.exists(p)
    stamp = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    with open(p, "a", encoding="utf-8") as fh:
        if new:
            fh.write("---\ntags: [machine-draft, reading]\nrun_id: %s\nstatus: unreviewed\n---\n\n"
                     "# Machine-drafted readings: %s\n\n> [!warning] MACHINE-DRAFTED. Not a reviewed "
                     "reading. Nothing here counts until a human copies it into the run's paired-reading "
                     "block.\n" % (run_id, run_id))
        fh.write("\n## %s -- %s\n\n%s\n" % (stamp, author, text.strip()))
    _audit(root, {"event": "draft_reading", "run_id": run_id, "author": author, "chars": len(text)})
    return {"status": "DRAFTED", "path": os.path.relpath(p, root).replace(os.sep, "/")}
