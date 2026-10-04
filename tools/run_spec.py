#!/usr/bin/env python
"""Run an experiment spec (IMPLEMENTATION_PLAN_2026-10 Phase E1).

    python tools/run_spec.py specs/approved/my_spec.json [--out DIR] [--dry-run]

One generic executor instead of one script per experiment. For each sweep point it:
  1. validates the spec (irer_specs.validate + registry names/versions);
  2. WARNS -- never blocks -- for each `requires` edge whose verdict is not found;
  3. builds the substrate, IC and observers from jax_scout/registry.py;
  4. advances in `sample_every` chunks, streaming every observer value to telemetry.jsonl
     (spec `invariants` become declared telemetry invariants, so tools/hud_monitor.py shows breaches live);
  5. writes spec.json (the concrete point), summary.json (stamped with provenance: commit, steppers,
     component hashes), verdict.json = PENDING_REVIEW, and RUN_COMPLETE.json;
  6. scores the pre-registered prediction into `prediction_check`. That is not a verdict -- a verdict is
     a reviewed judgement, which is why verdict.json starts as PENDING_REVIEW.

The run directory's top-level summary.json aggregates the points, so tools/build_results_index.py
indexes a spec run like any other run and links it to its spec (`spec_id`).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import irer_specs  # noqa: E402


def find_verdicts(spec_id, sweep_root):
    """{verdict: [run dirs]} recorded for spec_id. Reads verdict.json files; never fails."""
    out = {}
    for p in glob.glob(os.path.join(sweep_root, "*", "verdict.json")):
        try:
            v = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if v.get("spec_id") == spec_id:
            out.setdefault(v.get("verdict"), []).append(os.path.dirname(p))
    return out


def requires_warnings(spec, sweep_root):
    warns = []
    for r in spec.get("requires", []):
        got = find_verdicts(r["spec_id"], sweep_root)
        if r["verdict"] not in got:
            warns.append("requires %s == %s, but found %s -- running anyway (requires is descriptive)"
                         % (r["spec_id"], r["verdict"], sorted(k for k in got if k) or "no verdict"))
    return warns


def _slope(ts, ys):
    n = len(ts)
    if n < 2:
        return float("nan")
    mt, my = sum(ts) / n, sum(ys) / n
    den = sum((t - mt) ** 2 for t in ts)
    return sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / den if den else float("nan")


def run_point(spec, point_dir, *, log=print):
    from jax_scout import registry
    from jax_scout.provenance import write_json
    from jax_scout.snapshots import TelemetryWriter

    os.makedirs(point_dir, exist_ok=True)
    write_json(os.path.join(point_dir, "spec.json"), spec, stamp_metadata=False)
    pr = spec["protocol"]
    dt, T = float(pr["dt"]), float(pr["T"])
    n_total = int(round(T / dt))
    every = max(1, int(round(float(pr.get("sample_every", T / 100.0)) / dt)))
    max_wall = float((pr.get("stop") or {}).get("max_wall_h", math.inf)) * 3600.0
    stop_nonfinite = (pr.get("stop") or {}).get("nonfinite", True)

    t0 = time.time()
    sim, observers = registry.build(spec)
    tel = TelemetryWriter(point_dir, invariants=spec.get("invariants") or {},
                          meta={"harness": "run_spec.py", "spec_id": spec["id"]})

    def sample():
        row = {}
        for name, fn, params in observers:
            row.update(fn(sim, **params))
        tel.record(sim.t, **row)
        return row

    history = [(sim.t, sample())]
    done, stop_reason = 0, "completed"
    while done < n_total:
        n = min(every, n_total - done)
        sim.advance(n)
        done += n
        row = sample()
        history.append((sim.t, row))
        if stop_nonfinite and any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
            stop_reason = "nonfinite"
            break
        if time.time() - t0 > max_wall:
            stop_reason = "max_wall_h"
            break
    stats = tel.close()

    final = dict(history[-1][1])
    slopes = {}
    for o in spec["observers"]:
        for key in (o.get("params") or {}).get("slopes", []):
            half = [(t, r[key]) for t, r in history if t >= history[-1][0] / 2 and key in r]
            slopes[key + "__slope"] = _slope([t for t, _ in half], [y for _, y in half])
    final.update(slopes)
    check = irer_specs.check_prediction(spec["prediction"], final)
    summary = {"spec_id": spec["id"], "title": spec["title"], "substrate": spec["substrate"],
               "protocol": pr, "final": final, "prediction": spec["prediction"],
               "prediction_check": check, "stop_reason": stop_reason, "steps": done,
               "t_final": sim.t, "wall_s": round(time.time() - t0, 2), "telemetry": stats,
               "verdict": None}
    write_json(os.path.join(point_dir, "summary.json"), summary)
    write_json(os.path.join(point_dir, "verdict.json"),
               {"spec_id": spec["id"], "verdict": "PENDING_REVIEW", "prediction_check": check["status"],
                "note": "set by a reviewer; prediction_check is automatic and is not a verdict"},
               stamp_metadata=False)
    write_json(os.path.join(point_dir, "RUN_COMPLETE.json"), {"status": stop_reason})
    log("  %s: %s steps, t=%.4g, %s, %.1fs" % (os.path.basename(point_dir), done, sim.t, check["status"],
                                                summary["wall_s"]))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="validate and expand only")
    ap.add_argument("--sweep-root", default=os.path.join(ROOT, "sweep_runs"))
    args = ap.parse_args(argv)

    spec = irer_specs.load(args.spec)
    errs = irer_specs.validate(spec)
    if not errs:
        from jax_scout import registry
        errs = registry.check_spec(spec)
    if errs:
        print("INVALID SPEC %s:" % args.spec)
        for e in errs:
            print("  -", e)
        return 2
    for w in requires_warnings(spec, args.sweep_root):
        print("WARNING:", w)
    points = irer_specs.expand(spec)
    print("spec %s: %d point(s)" % (spec["id"], len(points)))
    if args.dry_run:
        for label, _ in points:
            print("  -", label or "(single)")
        return 0

    from jax_scout.provenance import write_json
    out = args.out or os.path.join(args.sweep_root, "%s_%s" % (spec["id"].upper().replace("-", "_").replace(".", "_"),
                                                               time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "spec.json"), spec, stamp_metadata=False)
    results = []
    for i, (label, concrete) in enumerate(points):
        pdir = os.path.join(out, label or "run") if len(points) > 1 else out
        try:
            s = run_point(concrete, pdir)
            results.append({"point": label, "dir": os.path.relpath(pdir, out), "final": s["final"],
                            "prediction_check": s["prediction_check"]["status"], "stop_reason": s["stop_reason"]})
        except Exception as exc:                   # one bad point must not lose the others
            print("  point %s FAILED: %s" % (label or "run", exc))
            results.append({"point": label, "error": str(exc)[:300]})
    if len(points) > 1:
        write_json(os.path.join(out, "summary.json"),
                   {"spec_id": spec["id"], "title": spec["title"], "n_points": len(points), "rows": results,
                    "verdict": None})
        write_json(os.path.join(out, "verdict.json"), {"spec_id": spec["id"], "verdict": "PENDING_REVIEW"},
                   stamp_metadata=False)
        write_json(os.path.join(out, "RUN_COMPLETE.json"), {"status": "completed"})
    print("=> %s" % out)
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
