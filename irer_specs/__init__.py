"""irer_specs — experiments as data (IMPLEMENTATION_PLAN_2026-10 Phase E1).

An experiment is a small JSON record (an ExperimentSpec), not a script. It names a SUBSTRATE (stepper +
operator), a PROTOCOL (initial condition, grid, dt, duration, sampling), the OBSERVERS to record, an
optional SWEEP, a PREDICTION written before launch, and soft REQUIRES edges to other specs' verdicts.
A few generic executors (tools/run_spec.py) run any spec; components are looked up by name in
jax_scout/registry.py.

DEPENDENCY-FREE on purpose: the JAX environment (WSL ~/jax_irer) has no pydantic/jsonschema, and the
spec layer must validate identically there, in .venv and in CI. The schema is plain JSON Schema
(schemas/experiment_spec.schema.json, generated from SCHEMA below) and `validate` implements exactly
the subset of keywords that schema uses.

DESCRIPTIVE, NEVER PRESCRIPTIVE: an unmet `requires` edge is a WARNING. Nothing here can stop a run.
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import re

SPEC_VERSION = 1
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(ROOT, "schemas", "experiment_spec.schema.json")

_NUM = {"type": "number"}
_COMPONENT = {
    "type": "object",
    "required": ["name"],
    "properties": {"name": {"type": "string"}, "version": {"type": "integer", "minimum": 1},
                   "params": {"type": "object"}},
    "additionalProperties": False,
}

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "irer/experiment_spec/v%d" % SPEC_VERSION,
    "title": "IRER experiment spec",
    "description": "One experiment as data: substrate + protocol + observers + prediction. "
                   "Executed by tools/run_spec.py; components resolved in jax_scout/registry.py.",
    "type": "object",
    "required": ["spec_version", "id", "title", "substrate", "protocol", "observers", "prediction"],
    "additionalProperties": False,
    "properties": {
        "spec_version": {"type": "integer", "enum": [SPEC_VERSION]},
        "id": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9_.-]{2,80}$",
               "description": "stable id; becomes the run-id prefix (upper-cased)"},
        "title": {"type": "string", "minLength": 3},
        "description": {"type": "string"},
        "branch": {"type": "string", "description": "the enquiry it serves, e.g. 'Branch - Transport - Index'"},
        "substrate": _COMPONENT,
        "protocol": {
            "type": "object",
            "required": ["ic", "grid", "dt", "T"],
            "additionalProperties": False,
            "properties": {
                "ic": _COMPONENT,
                "grid": {"type": "object", "required": ["N", "L"], "additionalProperties": False,
                         "properties": {"N": {"type": "integer", "minimum": 4}, "L": {"type": "number", "exclusiveMinimum": 0}}},
                "dt": {"type": "number", "exclusiveMinimum": 0},
                "T": {"type": "number", "exclusiveMinimum": 0, "description": "physical duration"},
                "sample_every": {"type": "number", "exclusiveMinimum": 0,
                                 "description": "physical time between observer samples (default T/100)"},
                "seed": {"type": "integer"},
                "stop": {"type": "object", "additionalProperties": False,
                         "properties": {"max_wall_h": _NUM, "nonfinite": {"type": "boolean"}}},
            },
        },
        "observers": {"type": "array", "minItems": 1, "items": _COMPONENT},
        "invariants": {"type": "object", "description": "{observer_key: tolerance} streamed as telemetry invariants",
                       "additionalProperties": {"type": "number"}},
        "sweep": {
            "type": "object", "additionalProperties": False, "required": ["axes"],
            "properties": {
                "mode": {"type": "string", "enum": ["grid", "zip"]},
                "axes": {"type": "object", "minProperties": 1,
                         "description": "{dotted.path: [values]} e.g. 'substrate.params.param_a'",
                         "additionalProperties": {"type": "array", "minItems": 1}},
            },
        },
        "prediction": {
            "type": "object", "required": ["statement", "quantities"], "additionalProperties": False,
            "description": "written BEFORE launch: what the equations say should happen",
            "properties": {
                "statement": {"type": "string", "minLength": 10},
                "derivation": {"type": "string"},
                "quantities": {"type": "array", "items": {
                    "type": "object", "required": ["key", "expected"], "additionalProperties": False,
                    "properties": {"key": {"type": "string", "description": "final-observer key, e.g. 'mass_ratio'"},
                                   "expected": _NUM,
                                   "tolerance": {"type": "number", "minimum": 0},
                                   "kind": {"type": "string", "enum": ["abs", "rel", "le", "ge"]}}}},
            },
        },
        "requires": {"type": "array", "items": {
            "type": "object", "required": ["spec_id", "verdict"], "additionalProperties": False,
            "properties": {"spec_id": {"type": "string"}, "verdict": {"type": "string"}}}},
        "budget_h": {"type": "number", "minimum": 0},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}


# ------------------------------------------------------------------ minimal validator

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


def _type_ok(v, t):
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, _TYPES[t])


def _validate(v, s, path, errs):
    t = s.get("type")
    if t and not _type_ok(v, t):
        errs.append("%s: expected %s, got %s" % (path or "$", t, type(v).__name__))
        return
    if "enum" in s and v not in s["enum"]:
        errs.append("%s: %r not in %r" % (path, v, s["enum"]))
    if isinstance(v, str):
        if "pattern" in s and not re.match(s["pattern"], v):
            errs.append("%s: %r does not match %s" % (path, v, s["pattern"]))
        if len(v) < s.get("minLength", 0):
            errs.append("%s: shorter than %d" % (path, s["minLength"]))
    if _type_ok(v, "number"):
        if "minimum" in s and v < s["minimum"]:
            errs.append("%s: %r < minimum %r" % (path, v, s["minimum"]))
        if "exclusiveMinimum" in s and v <= s["exclusiveMinimum"]:
            errs.append("%s: %r <= %r" % (path, v, s["exclusiveMinimum"]))
    if isinstance(v, list):
        if len(v) < s.get("minItems", 0):
            errs.append("%s: fewer than %d items" % (path, s["minItems"]))
        for i, x in enumerate(v):
            if "items" in s:
                _validate(x, s["items"], "%s[%d]" % (path, i), errs)
    if isinstance(v, dict):
        for k in s.get("required", []):
            if k not in v:
                errs.append("%s: missing required '%s'" % (path or "$", k))
        if len(v) < s.get("minProperties", 0):
            errs.append("%s: needs at least %d entries" % (path, s["minProperties"]))
        props = s.get("properties", {})
        extra = s.get("additionalProperties", True)
        for k, x in v.items():
            p = "%s.%s" % (path, k) if path else k
            if k in props:
                _validate(x, props[k], p, errs)
            elif extra is False:
                errs.append("%s: unknown key" % p)
            elif isinstance(extra, dict):
                _validate(x, extra, p, errs)


def prune_empty(obj):
    """Drop empty optional containers/strings ({} [] "" None), recursively. Form libraries fill optional
    objects with empty defaults (e.g. sweep: {axes: {}}), which would then fail validation; pruning
    turns them back into 'absent'. A required field that prunes away is still reported as missing."""
    if isinstance(obj, dict):
        out = {k: prune_empty(v) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v not in ({}, [], "", None)}
    if isinstance(obj, list):
        return [prune_empty(v) for v in obj]
    return obj


def validate(spec: dict, *, registry=None) -> list:
    """-> list of error strings (empty = valid). With `registry`, also checks component names exist."""
    errs = []
    _validate(spec, SCHEMA, "", errs)
    if registry is not None and not errs:
        errs += registry.check_spec(spec)
    return errs


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_schema(path: str = SCHEMA_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(SCHEMA, fh, indent=2)
        fh.write("\n")
    return path


# ------------------------------------------------------------------ sweep expansion

def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def expand(spec: dict) -> list:
    """-> [(point_label, concrete_spec)]: one per sweep point (just the spec itself if no sweep)."""
    sw = spec.get("sweep")
    if not sw:
        return [("", copy.deepcopy(spec))]
    axes = sw["axes"]
    names = list(axes)
    if sw.get("mode", "grid") == "zip":
        n = {len(v) for v in axes.values()}
        if len(n) != 1:
            raise ValueError("zip sweep needs equal-length axes")
        combos = list(zip(*[axes[k] for k in names]))
    else:
        combos = list(itertools.product(*[axes[k] for k in names]))
    out = []
    for combo in combos:
        s = copy.deepcopy(spec)
        s.pop("sweep", None)
        for k, v in zip(names, combo):
            _set(s, k, v)
        label = "__".join("%s=%s" % (k.split(".")[-1], v) for k, v in zip(names, combo))
        out.append((label, s))
    return out


# ------------------------------------------------------------------ prediction scoring

def check_prediction(prediction: dict, final: dict) -> dict:
    """Score the pre-registered prediction against final observer values. This is NOT a verdict:
    a verdict needs review (verdict.json stays PENDING_REVIEW)."""
    rows, met = [], True
    for q in prediction.get("quantities", []):
        key, exp = q["key"], float(q["expected"])
        tol, kind = float(q.get("tolerance", 0.0)), q.get("kind", "abs")
        got = final.get(key)
        if not isinstance(got, (int, float)):
            rows.append({"key": key, "expected": exp, "got": None, "ok": False, "why": "observer key missing"})
            met = False
            continue
        if kind == "rel":
            ok = abs(got - exp) <= tol * max(abs(exp), 1e-300)
        elif kind == "le":
            ok = got <= exp + tol
        elif kind == "ge":
            ok = got >= exp - tol
        else:
            ok = abs(got - exp) <= tol
        rows.append({"key": key, "expected": exp, "got": float(got), "tolerance": tol, "kind": kind, "ok": bool(ok)})
        met = met and ok
    return {"status": "PREDICTION_MET" if met else "PREDICTION_MISSED", "quantities": rows,
            "note": "scored automatically against the pre-registered prediction; not a verdict"}
