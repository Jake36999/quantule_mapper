#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Manifest results index — a queryable projection of the vault, rebuilt from disk.

WHY THIS SHAPE (Jake, 2026-08-25). The previous store (`orchestrator/` + `queue_runtime.db`,
1,850 rows, March 2026) failed for a specific reason: it was hardcoded for a FINALISED system.
It recorded numbered results plus a config hash and nothing else, so it could describe exactly
one variant of the model. The moment runs from different variants, eras or goals were mixed in,
the data polluted — there was no field that said which system a row belonged to.

So the fix is not better storage, it is a MISSING DIMENSION. This schema carries the semantic
context the old one lacked:

  substrate    which physical model the run belongs to (dissipative S-NCGL / NLS / KG /
               TG dual-substrate / Gravity-D mirror). The primary anti-pollution key.
  code_epoch   the git sha it ran against, or `pre-clean-slate` where git cannot say
               (the repo was clean-slated in 909e6e2, 2026-07-01).
  branch       the research pathway, matching the vault's main/side branch discipline.
  dof_free /   how many parameters were free vs fixed in advance. Threat T2 in
  dof_fixed    docs/SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW.md — a claim's evidential value
               depends on this, so it belongs in the record, not in a reader's memory.

And params/metrics are stored LONG (run_id, key, value) rather than as fixed columns. Fixed
columns are what forced the old store to assume one variant; long format absorbs heterogeneous
configs without a migration.

NOT A SERVICE. This is a build artifact: one pass over sweep_runs/ and docs/, emitting a single
SQLite file. No daemon, no worker pool, no live queue — those are what rotted last time, and
they are the wrong shape for three runs a month. Regenerable at any moment, so it can never
disagree with the runs it describes.

The headline query Jake asked for -- "search a branch and get its docs, results and runs" -- is
the view `v_branch_contents`.

Usage:
    .venv/Scripts/python.exe tools/build_results_index.py [--out PATH] [--parquet]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess
import datetime as _dt

import build_run_catalogue as brc  # reuse the extractors; do not duplicate them
import stepper_staleness as stal  # Phase B: which runs used a stepper that was later fixed


_TS = re.compile(r"_(\d{8})_(\d{6})$")


def commit_timeline():
    """[(unix_time, short_sha, subject)] oldest-first, for retro-dating runs."""
    try:
        raw = subprocess.run(["git", "log", "--reverse", "--format=%at|%h|%s"],
                             cwd=REPO, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    out = []
    for line in raw.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            try:
                out.append((float(parts[0]), parts[1], parts[2]))
            except ValueError:
                pass
    return out


def run_start_time(run_id, run_dir):
    """Best available start/finish time for a run, and how it was obtained."""
    m = _TS.search(run_id)
    if m:
        try:
            return _dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp(), "run-id"
        except ValueError:
            pass
    # no timestamp in the name: fall back to the mtime of the summary, i.e. when it finished.
    for cand in ("summary.json", "RUN_COMPLETE.json", ""):
        p = os.path.join(run_dir, cand) if cand else run_dir
        try:
            return os.path.getmtime(p), "mtime"
        except OSError:
            continue
    return None, None


def infer_epoch(run_id, timeline, run_dir=None):
    """Which commit was HEAD when this run started?

    Run ids carry a local-time start stamp (…_YYYYMMDD_HHMMSS). Mapping it to the latest
    commit at or before that moment gives the code the run MOST LIKELY used.

    This is an INFERENCE, not a record. It can be wrong if the working tree was dirty, if the
    run used an older checkout, or if the clock/timezone differed. It is therefore stored with
    code_epoch_source='inferred' and must never be treated as equivalent to a recorded commit.
    Commits here are days apart, so an hour of timezone ambiguity does not change the answer.
    """
    if not timeline:
        return None, None
    t, how = run_start_time(run_id, run_dir or "")
    if t is None:
        return None, None
    prior = [c for c in timeline if c[0] <= t]
    if not prior:
        return None, None
    return prior[-1][1], "inferred" if how == "run-id" else "inferred-mtime"

REPO = brc.REPO
VAULT = brc.VAULT
DEFAULT_OUT = os.path.join(VAULT, "runs", "_index.sqlite")

# ---------------------------------------------------------------- semantic dimensions

# The anti-pollution key: which physical model a run actually exercises.
SUBSTRATE = {
    "Stability": "dissipative-S-NCGL",
    "Phase-C": "dissipative-S-NCGL",
    "Substrate-hunt": "dissipative-S-NCGL",
    "Core-basin": "dissipative-S-NCGL",
    "Feb-basin": "dissipative-S-NCGL",
    "C2-NLS": "NLS-conservative",
    "C3-KG": "KG-conservative",
    "Phase-D": "phase-D-transport",
    "TG-B2": "TG-dual-substrate",
    "TG-B1S": "TG-dual-substrate",
    "TG-S": "TG-dual-substrate",
    "TG-other": "TG-dual-substrate",
    "Gravity-D": "gravity-D-spatial-mirror",
    "Validation": "infrastructure",
    "Baseline": "infrastructure",
}
CLEAN_SLATE_DATE = "2026-07-01"

SCHEMA = """

CREATE TABLE branches (
    branch      TEXT PRIMARY KEY,      -- 'Branch - Gravity - Index'
    name        TEXT,
    kind        TEXT,                  -- main | side
    sector      TEXT,
    n_runs      INTEGER,
    n_docs      INTEGER
);

CREATE TABLE documents (
    path        TEXT PRIMARY KEY,      -- vault-relative, e.g. gravity_maturity/TG_P1_...md
    title       TEXT,
    kind        TEXT,                  -- result | plan | record | index | checklist | run-note
    date        TEXT,
    branch      TEXT,
    sector      TEXT,
    verdict     TEXT,
    status      TEXT,
    has_what_changed INTEGER,          -- the hand-written consequences section (0/1)
    n_issues_open    INTEGER
);

CREATE TABLE runs (
    run_id      TEXT PRIMARY KEY,
    date        TEXT,
    sector      TEXT,
    family      TEXT,
    substrate   TEXT,                  -- ANTI-POLLUTION KEY: which model this run exercises
    branch      TEXT,
    harness     TEXT,                  -- harness id: recorded in provenance, else registry prefix match
    git_commit  TEXT,
    code_epoch  TEXT,                  -- git sha, or 'pre-clean-slate'
    code_epoch_source TEXT,            -- recorded | inferred | unavailable
    verdict     TEXT,
    complete    INTEGER,
    source      TEXT,                  -- summary.json | derived
    elapsed_h   REAL,
    n_csv       INTEGER,
    n_visual    INTEGER,
    dof_free    INTEGER,               -- threat T2: parameters free in this run
    dof_fixed   INTEGER,               -- parameters pinned before it
    run_dir     TEXT,                  -- gitignored local path
    steppers    TEXT,                  -- comma list: ETDRK4 | ETDRK4-cupy | KG-strang | TG-RK4 | GravityD-RK4
    stepper_source TEXT                -- recorded | harness | substrate | run_id | unknown (tools/stepper_staleness.py)
);

-- Instrument fixes (docs/registry/COMPONENT_FIXES.json) and, per run, whether its code had the fix.
-- DESCRIPTIVE ONLY: a label for review, never a gate. Added 2026-10-04 (IMPLEMENTATION_PLAN_2026-10 B2).
CREATE TABLE component_fixes (
    fix_id TEXT PRIMARY KEY, component TEXT, fix_commit TEXT, fix_date TEXT,
    affects_steppers TEXT, doc TEXT, summary TEXT
);
-- Harness registry (docs/registry/harness_registry.json, tools/build_harness_registry.py). Added
-- 2026-10-04 (IMPLEMENTATION_PLAN_2026-10 D3). Descriptive only.
CREATE TABLE harnesses (
    harness_id TEXT PRIMARY KEY, file TEXT, status TEXT, branch TEXT, superseded_by TEXT,
    produces TEXT, invariants TEXT, summary TEXT, has_manifest INTEGER
);
CREATE TABLE run_staleness (
    run_id TEXT, fix_id TEXT, status TEXT, reason TEXT,
    PRIMARY KEY (run_id, fix_id)
);

-- LONG format: heterogeneous configs across variants without a schema migration.
CREATE TABLE run_params (
    run_id      TEXT, key TEXT, num REAL, txt TEXT,
    PRIMARY KEY (run_id, key)
);
CREATE TABLE run_metrics (
    run_id      TEXT, key TEXT, num REAL, txt TEXT, row_idx INTEGER DEFAULT 0,
    PRIMARY KEY (run_id, key, row_idx)
);

CREATE TABLE artifacts (
    run_id      TEXT, kind TEXT, vault_path TEXT,   -- plot | figure | render
    PRIMARY KEY (run_id, vault_path)
);

-- One edge table for every relation, so new relation kinds need no migration.
CREATE TABLE edges (
    src_kind TEXT, src_id TEXT, rel TEXT, dst_kind TEXT, dst_id TEXT,
    PRIMARY KEY (src_kind, src_id, rel, dst_kind, dst_id)
);

CREATE TABLE verdicts (
    verdict TEXT PRIMARY KEY, n_runs INTEGER, first_seen TEXT, last_seen TEXT,
    catalog_tracked INTEGER
);

CREATE VIEW v_stale_runs AS
    SELECT s.run_id, s.fix_id, s.reason, r.date, r.substrate, r.family, r.steppers, r.stepper_source,
           r.code_epoch, r.verdict
      FROM run_staleness s JOIN runs r USING (run_id)
     WHERE s.status = 'STALE_PENDING_REVALIDATION';

CREATE INDEX idx_runs_branch    ON runs(branch);
CREATE INDEX idx_runs_substrate ON runs(substrate);
CREATE INDEX idx_runs_verdict   ON runs(verdict);
CREATE INDEX idx_docs_branch    ON documents(branch);
CREATE INDEX idx_edges_src      ON edges(src_kind, src_id);
CREATE INDEX idx_edges_dst      ON edges(dst_kind, dst_id);

-- THE HEADLINE QUERY: everything associated with a research pathway, in one place.
CREATE VIEW v_branch_contents AS
    SELECT branch, 'run'      AS item_kind, run_id AS item, date, verdict, substrate
      FROM runs
    UNION ALL
    SELECT branch, 'document' AS item_kind, path   AS item, date, verdict, NULL
      FROM documents;

-- Runs grouped by the dimension whose absence polluted the old store.
CREATE VIEW v_substrate_summary AS
    SELECT substrate, COUNT(*) AS n_runs,
           SUM(verdict IS NOT NULL) AS n_with_verdict,
           MIN(date) AS first_run, MAX(date) AS last_run
      FROM runs GROUP BY substrate;

-- Provenance quality. An inferred epoch is a best guess from the run's start time; it must
-- never be read as equivalent to a commit the harness actually recorded.
CREATE VIEW v_provenance AS
    SELECT code_epoch_source, COUNT(*) AS n_runs,
           SUM(verdict IS NOT NULL) AS n_with_verdict, MIN(date) AS first_run, MAX(date) AS last_run
      FROM runs GROUP BY code_epoch_source;

-- Documents with no recorded consequence: the review queue.
CREATE VIEW v_docs_needing_consequences AS
    SELECT path, date, branch, verdict FROM documents
     WHERE has_what_changed = 0 AND kind != 'index';
"""


def flatten(prefix, obj, out, depth=0):
    """Flatten nested summary structures into long key/value rows."""
    if depth > 3:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out, depth + 1)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append((prefix, float(obj), None))
    elif isinstance(obj, bool):
        out.append((prefix, 1.0 if obj else 0.0, str(obj).lower()))
    elif isinstance(obj, str) and len(obj) < 400:
        out.append((prefix, None, obj))


def doc_kind(path, text):
    b = os.path.basename(path).lower()
    if b.startswith("branch - ") and "index" in b:
        return "index"
    if "checklist" in b:
        return "checklist"
    if re.search(r"tags:.*\bindex\b", text[:400]):
        return "index"
    if "_RESULTS" in path or "results" in b:
        return "result"
    if "_PLAN" in path or "_RFC" in path or "_DESIGN" in path:
        return "plan"
    return "record"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--parquet", action="store_true", help="also emit runs as parquet if pandas is present")
    args = ap.parse_args()

    # ---- gather runs using the catalogue's own extractors (single source of truth) ----
    all_dirs = sorted(d for d in os.listdir(brc.SWEEP)
                      if os.path.isdir(os.path.join(brc.SWEEP, d)))
    recs = []
    for d in all_dirs:
        p = os.path.join(brc.SWEEP, d)
        try:
            if os.path.exists(os.path.join(p, "summary.json")):
                recs.append(brc.extract(p))
            elif brc.triage_class(brc.walk_run(p)) == "substantive":
                recs.append(brc.extract_nosummary(p))
        except Exception as e:  # noqa: BLE001
            print("  skip %s (%s)" % (d, e))

    if os.path.exists(args.out):
        os.remove(args.out)
    for suffix in ("-wal", "-shm"):
        if os.path.exists(args.out + suffix):
            os.remove(args.out + suffix)
    con = sqlite3.connect(args.out)
    con.executescript(SCHEMA)
    TIMELINE = commit_timeline()
    print('  git timeline: %d commits' % len(TIMELINE))

    # ---- documents -------------------------------------------------------------
    docs = {}
    for root, dirs, files in os.walk(VAULT):
        dirs[:] = [x for x in dirs if x not in brc_skip()]
        for f in files:
            if not f.endswith(".md"):
                continue
            rel = os.path.relpath(os.path.join(root, f), VAULT).replace(os.sep, "/")
            try:
                t = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            fm = t.split("---", 2)[1] if t.startswith("---") else ""
            get = lambda k: (re.search(r"^%s:\s*(.+)$" % k, fm, re.M) or [None, ""])[1].strip().strip('"')
            branch = get("branch") or (brc.SECTOR_INDEX.get(get("sector")) if get("sector") else "")
            n_open = len(re.findall(r"\|\s*OPEN\s*\|", t))
            docs[rel] = (rel, os.path.basename(rel)[:-3], doc_kind(rel, t), get("date"),
                         branch, get("sector"), get("verdict") or None, get("status"),
                         1 if "## What changed as a result" in t else 0, n_open)
    con.executemany("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)", docs.values())

    # ---- runs ------------------------------------------------------------------
    catalog = ""
    cpath = os.path.join(VAULT, "IRER_MASTER_HYPOTHESIS_CATALOG.md")
    if os.path.exists(cpath):
        catalog = open(cpath, encoding="utf-8", errors="replace").read()

    FIXES = stal.load_fixes()
    HARN = []
    try:
        HARN = json.load(open(os.path.join(VAULT, "registry", "harness_registry.json"),
                              encoding="utf-8")).get("harnesses", [])
    except (OSError, ValueError):
        pass
    con.executemany("INSERT OR REPLACE INTO harnesses VALUES (?,?,?,?,?,?,?,?,?)",
                    [(h["id"], h["file"], h["status"], h.get("branch"), h.get("superseded_by"),
                      ",".join(h.get("produces") or []), ",".join(h.get("invariants") or []),
                      h.get("summary"), 1 if h.get("manifest") else 0) for h in HARN])
    by_file = {os.path.basename(h["file"]): h["id"] for h in HARN}
    prefixes = sorted(((p, h["id"]) for h in HARN for p in (h.get("produces") or [])),
                      key=lambda x: -len(x[0]))

    def harness_of(run_id, summary):
        prov = summary.get("provenance") if isinstance(summary.get("provenance"), dict) else {}
        name = prov.get("harness") or summary.get("harness")
        if name and os.path.basename(name) in by_file:
            return by_file[os.path.basename(name)], "recorded"
        for p, hid in prefixes:
            if run_id.startswith(p):
                return hid, "prefix"
        return None, None
    con.executemany("INSERT OR REPLACE INTO component_fixes VALUES (?,?,?,?,?,?,?)",
                    [(f["id"], f.get("component"), f.get("fix_commit"), f.get("fix_date"),
                      ",".join(f.get("affects_steppers", [])), f.get("doc"), f.get("summary"))
                     for f in FIXES])

    for r in recs:
        fam = r["family"]
        gc = r.get("git_commit")
        if not gc and isinstance(r["summary"].get("provenance"), dict):
            gc = r["summary"]["provenance"].get("commit")   # write_json nests the stamp
        if gc:
            epoch, epoch_src = str(gc)[:12], "recorded"
        else:
            inferred, how = infer_epoch(r["run_id"], TIMELINE, r["run_dir"])
            if inferred:
                epoch, epoch_src = inferred, how
            elif r["date"] < CLEAN_SLATE_DATE:
                epoch, epoch_src = "pre-clean-slate", "unavailable"
            else:
                epoch, epoch_src = "unknown", "unavailable"
        cfg = r["summary"].get("config") if isinstance(r["summary"].get("config"), dict) else {}
        free = [k for k in cfg if k in brc_phys_params()]
        substrate = SUBSTRATE.get(fam, "unclassified")
        steppers, step_src = stal.resolve_steppers(r["summary"], substrate, r["run_id"])
        hid, hsrc = harness_of(r["run_id"], r["summary"])
        if hid:
            con.execute("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?)",
                        ("run", r["run_id"], "produced_by", "harness", hid))
        con.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["run_id"], r["date"], r["band"], fam, substrate,
             brc.SECTOR_INDEX.get(r["band"], ""), hid or "",
             gc, epoch, epoch_src, r["verdict"], 1 if r["complete"] else 0,
             "derived" if r.get("derived") else "summary.json",
             r.get("elapsed_h"), len(r["csvs"]), len(brc.find_images(r["run_dir"])),
             len(free), None, r["run_dir"], ",".join(steppers), step_src))
        for fx in FIXES:
            prov = r["summary"].get("provenance") if isinstance(r["summary"].get("provenance"), dict) else {}
            st = stal.staleness(r["run_id"], steppers, epoch, r["date"], fx,
                                component_hashes=prov.get("component_hashes"))
            if st:
                con.execute("INSERT OR REPLACE INTO run_staleness VALUES (?,?,?,?)",
                            (r["run_id"], fx["id"], st[0], st[1]))
                if st[0] == "REVALIDATED":
                    con.execute("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?)",
                                ("run", r["run_id"], "revalidated_by", "run",
                                 fx["revalidated"][r["run_id"]]))

        rows = []
        flatten("", cfg, rows)
        con.executemany("INSERT OR REPLACE INTO run_params VALUES (?,?,?,?)",
                        [(r["run_id"], k, n, t) for k, n, t in rows if k])
        mrows = []
        for key in ("rows", "results"):
            v = r["summary"].get(key)
            if isinstance(v, list):
                for i, row in enumerate(v[:200]):
                    if isinstance(row, dict):
                        sub = []
                        flatten("", row, sub)
                        mrows += [(r["run_id"], k, n, t, i) for k, n, t in sub if k]
                break
        for k, v in r["summary"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mrows.append((r["run_id"], k, float(v), None, 0))
        con.executemany("INSERT OR REPLACE INTO run_metrics VALUES (?,?,?,?,?)", mrows)

        arts = []
        for kind, folder in (("plot", "_plots"), ("figure", "_figures"), ("render", "_renders")):
            d = os.path.join(VAULT, "runs", folder, r["run_id"])
            if os.path.isdir(d):
                arts += [(r["run_id"], kind, "runs/%s/%s/%s" % (folder, r["run_id"], f))
                         for f in sorted(os.listdir(d))]
        con.executemany("INSERT OR REPLACE INTO artifacts VALUES (?,?,?)", arts)

        con.execute("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?)",
                    ("run", r["run_id"], "belongs_to", "branch", brc.SECTOR_INDEX.get(r["band"], "")))

    # ---- doc -> run and doc -> doc edges ---------------------------------------
    run_ids = {r["run_id"] for r in recs}
    stems = {os.path.basename(p)[:-3]: p for p in docs}
    for rel in docs:
        try:
            t = open(os.path.join(VAULT, rel), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for rid in run_ids:
            if rid in t:
                con.execute("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?)",
                            ("document", rel, "references", "run", rid))
        for stem, target in stems.items():
            if len(stem) > 8 and target != rel and stem in t:
                con.execute("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?)",
                            ("document", rel, "cites", "document", target))
        if docs[rel][4]:
            con.execute("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?)",
                        ("document", rel, "belongs_to", "branch", docs[rel][4]))

    # ---- lineage: previous/next run within a substrate --------------------------
    bysub = {}
    for r in recs:
        bysub.setdefault(SUBSTRATE.get(r["family"], "unclassified"), []).append(r)
    for sub, rs in bysub.items():
        rs.sort(key=lambda x: (x["date"], x["run_id"]))
        for i in range(len(rs) - 1):
            con.execute("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?)",
                        ("run", rs[i]["run_id"], "precedes", "run", rs[i + 1]["run_id"]))

    # ---- verdicts ---------------------------------------------------------------
    con.execute("""INSERT INTO verdicts (verdict, n_runs, first_seen, last_seen, catalog_tracked)
                   SELECT verdict, COUNT(*), MIN(date), MAX(date), 0 FROM runs
                    WHERE verdict IS NOT NULL GROUP BY verdict""")
    for (v,) in con.execute("SELECT verdict FROM verdicts").fetchall():
        con.execute("UPDATE verdicts SET catalog_tracked=? WHERE verdict=?",
                    (1 if v and v in catalog else 0, v))

    # ---- branch rollup ----------------------------------------------------------
    for band, bname in brc.SECTOR_INDEX.items():
        nr = con.execute("SELECT COUNT(*) FROM runs WHERE branch=?", (bname,)).fetchone()[0]
        nd = con.execute("SELECT COUNT(*) FROM documents WHERE branch=?", (bname,)).fetchone()[0]
        con.execute("INSERT OR REPLACE INTO branches VALUES (?,?,?,?,?,?)",
                    (bname, brc.SECTOR_TITLE.get(band, band), brc.SECTOR_KIND.get(band, "side"),
                     band, nr, nd))

    con.commit()

    # ---- report -----------------------------------------------------------------
    q = lambda s: con.execute(s).fetchall()
    print("WROTE %s (%.1f KB)" % (args.out, os.path.getsize(args.out) / 1024))
    print("  runs=%d  documents=%d  params=%d  metrics=%d  artifacts=%d  edges=%d  verdicts=%d" % tuple(
        q("SELECT (SELECT COUNT(*) FROM runs),(SELECT COUNT(*) FROM documents),"
          "(SELECT COUNT(*) FROM run_params),(SELECT COUNT(*) FROM run_metrics),"
          "(SELECT COUNT(*) FROM artifacts),(SELECT COUNT(*) FROM edges),"
          "(SELECT COUNT(*) FROM verdicts)")[0]))
    print("\n  by substrate (the anti-pollution key):")
    for s, n, nv, f, l in q("SELECT * FROM v_substrate_summary ORDER BY n_runs DESC"):
        print("    %-26s %3d runs (%2d with verdict)  %s .. %s" % (s, n, nv, f, l))

    print("\n  stepper fixes (docs/registry/COMPONENT_FIXES.json) -- descriptive, never a gate:")
    for fid, st, n in q("SELECT fix_id, status, COUNT(*) FROM run_staleness GROUP BY fix_id, status "
                        "ORDER BY fix_id, status"):
        print("    %-18s %-28s %3d runs" % (fid, st, n))

    if args.parquet:
        try:
            import pandas as pd
            df = pd.read_sql_query("SELECT * FROM runs", con)
            pq = args.out.replace(".sqlite", ".parquet")
            df.to_parquet(pq, index=False)
            print("\n  parquet: %s (%d rows)" % (pq, len(df)))
        except Exception as e:  # noqa: BLE001
            print("\n  parquet skipped (%s)" % e)
    con.execute('PRAGMA journal_mode=DELETE')
    con.execute('VACUUM')
    con.close()


def brc_skip():
    return {"_plots", "_figures", "_renders", "_gallery", ".obsidian", ".trash",
            "transcripts", "claud_memory_files"}


def brc_phys_params():
    return {"c", "m", "a", "s", "f", "w", "alpha_T", "omega_T", "omega_G", "gamma_T",
            "gamma_G", "kappa_TG", "epsilon_G", "cT", "cG", "core_radius"}


if __name__ == "__main__":
    sys.exit(main())
