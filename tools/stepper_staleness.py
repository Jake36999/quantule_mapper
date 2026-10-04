"""Which runs used a stepper that was later fixed? (Phase B of IMPLEMENTATION_PLAN_2026-10.)

WHY. On 2026-10-02 two ETDRK4 bugs were fixed (e270cdc). Working out which runs they touched took an
afternoon of reading harnesses. This module turns it into a query: every run gets a set of steppers,
and every fix in docs/registry/COMPONENT_FIXES.json marks the runs that loaded an affected stepper on
code that did not yet contain the fix as STALE_PENDING_REVALIDATION.

DESCRIPTIVE ONLY. This labels runs; it never blocks one. (orchestrator/ died because it gated work.)

Stepper resolution, most to least trustworthy:
  recorded   the run's provenance stamp lists `steppers` (jax_scout/provenance.py, from 2026-10-04).
  harness    the stamp names the harness script; its transitive in-repo imports are scanned by AST
             (never imported or executed) and mapped through provenance.STEPPER_MODULES.
  substrate  only the run's substrate/family is known; a fixed table maps it to a stepper.
  run_id     the substrate is mixed (e.g. phase-D-transport); a run-id prefix table decides.
  unknown    none of the above.
Over-inclusion is deliberate: flagging a run for re-validation that turns out unaffected is cheap;
missing one is the failure this prevents.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXES_PATH = os.path.join(ROOT, "docs", "registry", "COMPONENT_FIXES.json")

# Imported lazily-safe: provenance has no heavy imports.
import sys as _sys  # noqa: E402
_sys.path.insert(0, ROOT)
from jax_scout.provenance import STEPPER_MODULES  # noqa: E402

#: Fallback when nothing better is known. Keys are build_results_index SUBSTRATE values.
SUBSTRATE_STEPPERS = {
    "dissipative-S-NCGL": ["ETDRK4"],
    "NLS-conservative": ["ETDRK4"],
    "KG-conservative": ["KG-strang"],
    "TG-dual-substrate": ["TG-RK4"],
    "gravity-D-spatial-mirror": ["GravityD-RK4"],
}


#: For mixed substrates. First match wins, so specific prefixes come first.
RUNID_PREFIX_STEPPERS = [
    ("PHASE_D_C3", ["KG-strang"]),
    ("PHASE_D_", ["ETDRK4"]),                      # C1 dispersive / C2 NLS / node coupling: physics.step
    ("CORRECTED_PHYSICS_JAX_SCOUT", ["ETDRK4"]),   # the physics.py baseline scout
]


def _module_path(mod: str):
    rel = mod.replace(".", os.sep)
    for cand in (rel + ".py", os.path.join(rel, "__init__.py")):
        p = os.path.join(ROOT, cand)
        if os.path.exists(p):
            return p
    return None


def _imports_of(path: str):
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except (OSError, SyntaxError):
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out |= {node.module + "." + a.name for a in node.names}
    return out


@lru_cache(maxsize=None)
def harness_steppers(harness: str) -> tuple:
    """Steppers reachable from a harness script by transitive in-repo imports (AST only)."""
    if not harness:
        return ()
    name = os.path.basename(harness)
    start = None
    for sub in ("jax_scout", "tools", ""):
        p = os.path.join(ROOT, sub, name)
        if os.path.exists(p):
            start = p
            break
    if start is None:
        return ()
    seen_mods, stack, found = set(), [start], set()
    seen_paths = set()
    while stack:
        p = stack.pop()
        if p in seen_paths:
            continue
        seen_paths.add(p)
        for mod in _imports_of(p):
            if mod in STEPPER_MODULES:
                found.add(STEPPER_MODULES[mod])
            if mod in seen_mods:
                continue
            seen_mods.add(mod)
            mp = _module_path(mod)
            if mp:
                stack.append(mp)
    return tuple(sorted(found))


def resolve_steppers(summary: dict, substrate: str, run_id: str = ""):
    """-> (list_of_steppers, source)."""
    prov = summary.get("provenance") if isinstance(summary.get("provenance"), dict) else {}
    rec = prov.get("steppers")
    if rec is None:
        flat = summary.get("steppers")
        if isinstance(flat, str):
            rec = [x for x in flat.split(",") if x]
        elif isinstance(flat, list):
            rec = flat
    if rec:
        return sorted(rec), "recorded"
    harness = prov.get("harness") or summary.get("harness")
    hs = harness_steppers(harness) if harness else ()
    if hs:
        return list(hs), "harness"
    if substrate in SUBSTRATE_STEPPERS:
        return list(SUBSTRATE_STEPPERS[substrate]), "substrate"
    for prefix, st in RUNID_PREFIX_STEPPERS:
        if run_id.startswith(prefix):
            return list(st), "run_id"
    return [], "unknown"


def load_fixes(path: str = FIXES_PATH):
    try:
        return json.load(open(path, encoding="utf-8")).get("fixes", [])
    except (OSError, ValueError):
        return []


@lru_cache(maxsize=None)
def _contains(fix_commit: str, epoch: str):
    """True if `epoch` already contains `fix_commit`; None if git cannot say."""
    try:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", fix_commit, epoch], cwd=ROOT,
                           capture_output=True, timeout=30)
    except Exception:
        return None
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    return None


def staleness(run_id: str, steppers, epoch: str, run_date: str, fix: dict):
    """-> (status, reason) for one run against one fix, or None if the fix does not apply."""
    if not set(steppers) & set(fix.get("affects_steppers", [])):
        return None
    if run_id in fix.get("revalidated", {}):
        return "REVALIDATED", "revalidated by %s" % fix["revalidated"][run_id]
    if run_id in fix.get("not_affected", {}):
        return "NOT_AFFECTED", fix["not_affected"][run_id]
    has_fix = None
    if epoch and epoch not in ("pre-clean-slate", "unknown"):
        has_fix = _contains(fix["fix_commit"], epoch)
    if has_fix is None:
        # git cannot place the run: fall back to dates. Same-day runs stay stale (conservative).
        has_fix = bool(run_date) and run_date > fix.get("fix_date", "9999")
        how = "date"
    else:
        how = "ancestry"
    if has_fix:
        return "CURRENT", "code contains %s (%s)" % (fix["fix_commit"], how)
    return "STALE_PENDING_REVALIDATION", "code predates %s (%s)" % (fix["fix_commit"], how)
