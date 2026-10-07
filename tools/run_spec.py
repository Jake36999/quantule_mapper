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


def _setup_member(spec, point_dir):
    """Build one member: substrate + IC + observers + telemetry + optional recorder. No stepping."""
    from jax_scout import registry
    from jax_scout.provenance import write_json
    from jax_scout.snapshots import TelemetryWriter

    os.makedirs(point_dir, exist_ok=True)
    write_json(os.path.join(point_dir, "spec.json"), spec, stamp_metadata=False)
    pr = spec["protocol"]
    dt, T = float(pr["dt"]), float(pr["T"])
    n_total = int(round(T / dt))
    every = max(1, int(round(float(pr.get("sample_every", T / 100.0)) / dt)))
    sim, observers = registry.build(spec)
    m = {"spec": spec, "dir": point_dir, "sim": sim, "observers": observers, "n_total": n_total, "t0": time.time(),
         "tel": TelemetryWriter(point_dir, invariants=spec.get("invariants") or {},
                                meta={"harness": "run_spec.py", "spec_id": spec["id"]}),
         "final_only": {o["name"] for o in spec["observers"] if (o.get("params") or {}).get("final_only")},
         "stop_nonfinite": (pr.get("stop") or {}).get("nonfinite", True),
         "sample_steps": set(range(every, n_total + 1, every)) | {n_total},
         "recorder": None, "rec_steps": set(), "want": set(), "stop_reason": "completed", "stopped": False}
    # --- optional playable history (protocol.record) --------------------------------------------
    # Frames go to <point>/history/ through the same SnapshotWriter the harnesses use (slices + a
    # downsampled volume per frame). `window` restricts capture to [t_start, t_end]: the run still has to
    # be integrated from t=0 to t_end, but only the window is written, so a short dynamic episode can be
    # recorded at a high frame rate without storing the whole run.
    rec = pr.get("record")
    if rec:
        from jax_scout.snapshots import SnapshotWriter
        r_every = max(1, int(round(float(rec["every"]) / dt)))
        lo, hi = rec.get("window", [0.0, T])
        s_lo, s_hi = max(0, int(round(float(lo) / dt))), min(n_total, int(round(float(hi) / dt)))
        m["rec_steps"] = set(range(s_lo, s_hi + 1, r_every))
        m["recorder"] = SnapshotWriter(os.path.join(point_dir, "history"), enabled=True, every=1, volume_every=1,
                                       volume_target=int(rec.get("volume", 48)), maxqueue=64)
        m["want"] = set(rec.get("fields") or [])
    return m


def _sample(m, last=False):
    row = {}
    for name, fn, params in m["observers"]:
        if name in m["final_only"] and not last:
            continue
        row.update(fn(m["sim"], **params))
    m["tel"].record(m["sim"].t, **row)
    return row


def _capture(m):
    f = m["sim"].fields()
    m["recorder"].capture(m["sim"].t, {k: v for k, v in f.items() if not m["want"] or k in m["want"]},
                          dx=m["sim"].grid.dx)


def _run_members(members, advance_all, max_wall=math.inf):
    """Shared event loop: advance ALL members to the next sample/record step, then let each one sample,
    record and check its stop conditions. A single run is a batch of one. Members in a batch share their
    step schedule (enforced by the batch key), so one schedule drives them all."""
    m0 = members[0]
    for m in members:
        m["history"] = [(m["sim"].t, _sample(m, last=m["n_total"] == 0))]
        if m["recorder"] and 0 in m["rec_steps"]:
            _capture(m)
    t_start, done = time.time(), 0
    for target in sorted((m0["sample_steps"] | m0["rec_steps"]) - {0}):
        if target <= done:
            continue
        if all(m["stopped"] for m in members):
            break
        advance_all(target - done)
        done = target
        for m in members:
            if m["stopped"]:
                continue
            if m["recorder"] and done in m["rec_steps"]:
                _capture(m)
            if done not in m["sample_steps"]:
                continue
            row = _sample(m, last=done >= m["n_total"])
            m["history"].append((m["sim"].t, row))
            m["steps"] = done
            if m["stop_nonfinite"] and any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                m["stop_reason"], m["stopped"] = "nonfinite", True
        if time.time() - t_start > max_wall:
            for m in members:
                if not m["stopped"]:
                    m["stop_reason"], m["stopped"] = "max_wall_h", True
            break
    return done


def _finalize(m, done, log=print, batch=None):
    from jax_scout.provenance import write_json
    spec, point_dir, sim = m["spec"], m["dir"], m["sim"]
    if m["final_only"] and m["stop_reason"] != "completed":
        m["history"].append((sim.t, _sample(m, last=True)))     # an early stop still gets its final descriptors
    stats = m["tel"].close()
    rec_stats = m["recorder"].close() if m["recorder"] else None
    history = m["history"]
    final = dict(history[-1][1])
    slopes = {}
    for o in spec["observers"]:
        for key in (o.get("params") or {}).get("slopes", []):
            half = [(t, r[key]) for t, r in history if t >= history[-1][0] / 2 and key in r]
            slopes[key + "__slope"] = _slope([t for t, _ in half], [y for _, y in half])
    final.update(slopes)
    check = irer_specs.check_prediction(spec["prediction"], final)
    steps = m.get("steps", done)
    summary = {"spec_id": spec["id"], "title": spec["title"], "substrate": spec["substrate"],
               "protocol": spec["protocol"], "final": final, "prediction": spec["prediction"],
               "prediction_check": check, "stop_reason": m["stop_reason"], "steps": steps,
               "t_final": sim.t, "wall_s": round(time.time() - m["t0"], 2), "telemetry": stats,
               "history": rec_stats, "batch": batch, "verdict": None}
    write_json(os.path.join(point_dir, "summary.json"), summary)
    write_json(os.path.join(point_dir, "verdict.json"),
               {"spec_id": spec["id"], "verdict": "PENDING_REVIEW", "prediction_check": check["status"],
                "note": "set by a reviewer; prediction_check is automatic and is not a verdict"},
               stamp_metadata=False)
    write_json(os.path.join(point_dir, "RUN_COMPLETE.json"), {"status": m["stop_reason"]})
    log("  %s: %s steps, t=%.4g, %s, %.1fs%s" % (os.path.basename(point_dir), steps, sim.t, check["status"],
                                                  summary["wall_s"], " [batch %d/%d]" % (batch["index"] + 1, batch["size"]) if batch else ""))
    return summary


def run_point(spec, point_dir, *, log=print):
    """One point, stepped on its own."""
    m = _setup_member(spec, point_dir)
    max_wall = float((spec["protocol"].get("stop") or {}).get("max_wall_h", math.inf)) * 3600.0
    done = _run_members([m], m["sim"].advance, max_wall)
    return _finalize(m, done, log)


def schedule_key(spec):
    """Everything that must match for points to share one time-stepping schedule."""
    pr = spec["protocol"]
    return json.dumps({k: pr.get(k) for k in ("dt", "T", "sample_every", "record", "stop")}, sort_keys=True)


def run_batch(specs, dirs, *, log=print):
    """Several points stepped TOGETHER: one jax.vmap over (parameters, state) per advance
    (jax_scout.registry.BatchedETDRK4). Initial conditions, observers, telemetry and recording are the
    per-member single-run code, so a batched member's outputs match its single run (tested)."""
    from jax_scout import registry
    members = [_setup_member(s, d) for s, d in zip(specs, dirs)]
    batched = registry.BatchedETDRK4([m["sim"] for m in members], specs)
    max_wall = float((specs[0]["protocol"].get("stop") or {}).get("max_wall_h", math.inf)) * 3600.0
    done = _run_members(members, batched.advance, max_wall)
    return [_finalize(m, done, log, batch={"index": i, "size": len(members)}) for i, m in enumerate(members)]


def plan_batches(points, batch_size=None):
    """Group expanded points into batches that can share one vmapped call: same registry.batch_key
    (substrate, grid, dt, static operator args) and same schedule_key. Chunks respect the GPU memory
    budget (registry.max_batch). Anything that cannot batch runs as a batch of one."""
    from jax_scout import registry
    groups, order = {}, []
    for label, spec in points:
        bk = registry.batch_key(spec)
        key = (bk, schedule_key(spec)) if bk is not None else ("single", label)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((label, spec))
    batches = []
    for key in order:
        items = groups[key]
        n = items[0][1]["protocol"]["grid"]["N"]
        size = batch_size or registry.max_batch(n)
        for i in range(0, len(items), max(1, size)):
            batches.append(items[i:i + size])
    return batches


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="validate and expand only")
    ap.add_argument("--sweep-root", default=os.path.join(ROOT, "sweep_runs"))
    ap.add_argument("--no-batch", dest="batch", action="store_false",
                    help="step every sweep point on its own (default: vmap compatible ETDRK4 points together)")
    ap.add_argument("--batch-size", type=int, default=None, help="members per batch (default: GPU memory budget)")
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
    pdir = lambda label: os.path.join(out, label or "run") if len(points) > 1 else out  # noqa: E731
    batches = plan_batches(points, args.batch_size) if args.batch else [[p] for p in points]
    if args.batch and any(len(b) > 1 for b in batches):
        print("batched: %s" % ", ".join(str(len(b)) for b in batches))
    for batch in batches:
        labels = [lb for lb, _ in batch]
        try:
            if len(batch) == 1:
                summaries = [run_point(batch[0][1], pdir(labels[0]))]
            else:
                summaries = run_batch([sp for _, sp in batch], [pdir(lb) for lb in labels])
            for label, s in zip(labels, summaries):
                results.append({"point": label, "dir": os.path.relpath(pdir(label), out), "final": s["final"],
                                "prediction_check": s["prediction_check"]["status"], "stop_reason": s["stop_reason"]})
        except Exception as exc:                   # one bad batch must not lose the others
            print("  batch %s FAILED: %s" % (", ".join(lb or "run" for lb in labels), exc))
            results += [{"point": lb, "error": str(exc)[:300]} for lb in labels]
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
