#!/usr/bin/env python
"""Harness registry — what each experiment script IS, generated from the scripts themselves.

WHY (docs/SESSION_SYNTHESIS_2026-09-17.md section 6). There are ~126 harnesses and no way to ask: is
this current? what superseded it? which enquiry does it serve? which runs did it produce? The "harness"
label was doing lifecycle work it had no vocabulary for.

THE MANIFEST. A harness declares one module-level dict literal:

    HARNESS = {
        "id": "feb_astar_confirm",                 # stable id (defaults to the file stem)
        "branch": "Branch - Stability - Index",    # the enquiry it serves
        "status": "ACTIVE",                        # ACTIVE | SUPERSEDED | RETIRED | PROPOSED
        "superseded_by": None,                     # required when SUPERSEDED
        "invariants": [],                          # names it streams as telemetry invariants
        "produces": ["FEB_ASTAR_CONFIRM"],         # run-id prefixes it writes under sweep_runs/
        "summary": "one line: what question it answers",
    }

It is read with ast.literal_eval and NEVER by importing the script, so a harness that no longer imports
(broken dependency, missing GPU) still registers. Harnesses without a manifest are listed as UNREVIEWED
with what can be inferred (run-id prefixes from the source), not edited by hand.

DESCRIPTIVE, NEVER PRESCRIPTIVE. Nothing here is consulted before a harness runs. The previous attempt
at this (orchestrator/) died because it gated experimentation; this records what exists.

Outputs:
    docs/registry/harness_registry.json              machine-readable (read by build_results_index)
    docs/research_infrastructure/HARNESS_REGISTRY.md generated view

Usage:
    python tools/build_harness_registry.py [--check FILE ...]
      --check: exit 1 if any listed file writes to sweep_runs/ without a valid HARNESS (CI uses this on
               changed files only).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ("jax_scout", "tools")
STATUSES = ("ACTIVE", "SUPERSEDED", "RETIRED", "PROPOSED")
OUT_JSON = os.path.join(ROOT, "docs", "registry", "harness_registry.json")
OUT_MD = os.path.join(ROOT, "docs", "research_infrastructure", "HARNESS_REGISTRY.md")
_PREFIX_RE = re.compile(r"""["']([A-Z][A-Z0-9_]{3,})_\{""")   # f"PHASE_D_C1_TRANSPORT_{time...}"


_WRITE_HINTS = ("makedirs", "mkdir", "write_json", "write_text", "savez", "to_csv", "DictWriter",
                "json.dump", "np.save", '"w"', "'w'")


def writes_sweep_runs(src: str) -> bool:
    """True if the script builds a sweep_runs/ path IN CODE (not in a docstring or comment) and also
    writes files. A module may opt out with NOT_A_HARNESS = "reason" (e.g. a monitor or an index
    builder that reads runs and writes only derived views)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "sweep_runs" in src
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr)                 and isinstance(getattr(body[0], "value", None), ast.Constant):
            docstrings.add(id(body[0].value))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NOT_A_HARNESS"
                                                 for t in node.targets):
            return False
    literal = any(isinstance(n, ast.Constant) and isinstance(n.value, str) and "sweep_runs" in n.value
                  and id(n) not in docstrings for n in ast.walk(tree))
    return literal and any(h in src for h in _WRITE_HINTS)


def read_manifest(src: str):
    """-> (dict or None, error or None). Literal-only; never executes the module."""
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return None, "syntax error: %s" % exc
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HARNESS"
                                                 for t in node.targets):
            try:
                val = ast.literal_eval(node.value)
            except ValueError:
                return None, "HARNESS is not a literal dict"
            if not isinstance(val, dict):
                return None, "HARNESS is not a dict"
            return val, None
    return None, None


def validate(man: dict) -> list:
    errs = []
    if man.get("status") not in STATUSES:
        errs.append("status must be one of %s" % "|".join(STATUSES))
    if man.get("status") == "SUPERSEDED" and not man.get("superseded_by"):
        errs.append("SUPERSEDED requires superseded_by")
    for k in ("branch", "summary"):
        if not man.get(k):
            errs.append("missing %s" % k)
    if not isinstance(man.get("produces", []), list):
        errs.append("produces must be a list of run-id prefixes")
    return errs


def scan():
    entries = []
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        for f in sorted(os.listdir(base)):
            if not f.endswith(".py") or f.startswith("_"):
                continue
            rel = "%s/%s" % (d, f)
            src = open(os.path.join(base, f), encoding="utf-8", errors="replace").read()
            man, err = read_manifest(src)
            if man is None and not writes_sweep_runs(src):   # a declared manifest is always listed
                continue
            inferred = sorted(set(_PREFIX_RE.findall(src)))
            if man is None:
                entries.append({"id": f[:-3], "file": rel, "status": "UNREVIEWED", "branch": None,
                                "superseded_by": None, "invariants": [], "produces": inferred,
                                "summary": None, "manifest": False, "error": err})
            else:
                e = {"id": man.get("id") or f[:-3], "file": rel, "status": man.get("status"),
                     "branch": man.get("branch"), "superseded_by": man.get("superseded_by"),
                     "invariants": list(man.get("invariants") or []),
                     "produces": list(man.get("produces") or inferred), "summary": man.get("summary"),
                     "manifest": True, "error": "; ".join(validate(man)) or None}
                entries.append(e)
    return entries


def write_outputs(entries):
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"_about": "Generated by tools/build_harness_registry.py -- do not edit. "
                             "Descriptive only: never consulted before a harness runs.",
                   "harnesses": entries}, fh, indent=2)
    by = {}
    for e in entries:
        by.setdefault(e["status"], []).append(e)
    order = ["ACTIVE", "PROPOSED", "SUPERSEDED", "RETIRED", "UNREVIEWED"]
    L = ["---", "tags: [registry, infra, generated]", "branch: Branch - Validation - Index",
         "status: running", "---", "",
         "# Harness registry (generated)", "",
         "> [!info] Generated by `tools/build_harness_registry.py` from each script's `HARNESS` manifest "
         "(read by AST, never imported). **Do not edit.** Descriptive only — nothing consults this before "
         "a harness runs. To add or change an entry, edit the `HARNESS = {...}` dict in the script.", "",
         "| status | count |", "|---|---|"]
    L += ["| %s | %d |" % (s, len(by.get(s, []))) for s in order if by.get(s)]
    for s in order:
        if not by.get(s):
            continue
        L += ["", "## %s" % s, ""]
        if s == "UNREVIEWED":
            L += ["Scripts that write to `sweep_runs/` but declare no manifest. `produces` is inferred "
                  "from run-id prefixes in the source.", ""]
            L += ["| file | inferred run-id prefixes |", "|---|---|"]
            L += ["| `%s` | %s |" % (e["file"], ", ".join("`%s`" % p for p in e["produces"]) or "—")
                  for e in by[s]]
        else:
            L += ["| id | file | branch | produces | invariants | summary | issues |",
                  "|---|---|---|---|---|---|---|"]
            for e in by[s]:
                L.append("| %s | `%s` | %s | %s | %s | %s | %s |" % (
                    e["id"], e["file"], ("[[%s]]" % e["branch"]) if e["branch"] else "—",
                    ", ".join("`%s`" % p for p in e["produces"]) or "—",
                    ", ".join("`%s`" % i for i in e["invariants"]) or "—",
                    e["summary"] or "—",
                    "; ".join(x for x in (e["error"], ("superseded by `%s`" % e["superseded_by"])
                                          if e["superseded_by"] else None) if x)))
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def check(files) -> int:
    bad = []
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(ROOT, f)
        if not (p.endswith(".py") and os.path.exists(p)):
            continue
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        if not rel.split("/")[0] in SCAN_DIRS:
            continue
        src = open(p, encoding="utf-8", errors="replace").read()
        if not writes_sweep_runs(src):
            continue
        man, err = read_manifest(src)
        if man is None:
            bad.append((rel, err or "writes to sweep_runs/ but declares no HARNESS manifest"))
        elif validate(man):
            bad.append((rel, "; ".join(validate(man))))
    for rel, why in bad:
        print("HARNESS MANIFEST: %s -- %s" % (rel, why))
    if bad:
        print("\nAdd a HARNESS = {...} dict (see tools/build_harness_registry.py). This checks the "
              "DECLARATION only; it never stops a harness from running.")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", nargs="*", help="validate these files only (CI: changed files)")
    args = ap.parse_args()
    if args.check is not None:
        return check(args.check)
    entries = scan()
    write_outputs(entries)
    n = {}
    for e in entries:
        n[e["status"]] = n.get(e["status"], 0) + 1
    print("harness registry: %d scripts write to sweep_runs/  %s" % (len(entries), n))
    print("  -> %s\n  -> %s" % (os.path.relpath(OUT_JSON, ROOT), os.path.relpath(OUT_MD, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
