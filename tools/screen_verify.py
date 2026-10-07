#!/usr/bin/env python
"""Screen in fp32, verify in fp64: pick the screening points that need a double-precision re-run.

    # 1. screen: a sweep spec run cheaply (fp32, small grid, batched)
    python tools/run_spec.py specs/approved/my-ensemble.json --precision fp32
    # 2. cluster the screen's end states
    python tools/basin_cluster.py sweep_runs/<SCREEN> --group-by param_a
    # 3. plan the fp64 re-runs (writes a spec to specs/proposed/, prints the command)
    python tools/screen_verify.py plan sweep_runs/<SCREEN> --group-by param_a --N 96
    # 4. run it (a human starts runs; this tool never launches anything)
    python tools/run_spec.py specs/proposed/<SCREEN-id>-verify.json
    # 5. did fp64 put each point in the basin fp32 said?
    python tools/screen_verify.py compare sweep_runs/<SCREEN> sweep_runs/<VERIFY>

WHY. fp32 is 3.7-4.5x faster on the GTX 1080 but drifts ~0.1-0.5% from fp64 over an a* replay
(docs/research_infrastructure/BATCHED_RUNS.md). That is far below the tens-of-percent gaps between
basins, so a screen's basin MAP is trustworthy in the bulk -- but not where it matters most: next to a
basin boundary, a 0.5% perturbation can change the outcome. So only those places are re-run in fp64:

  boundary        adjacent parameter points (along one --group-by axis) whose basin sets differ:
                  one replicate per basin at both points
  ic_split        a parameter point whose replicates land in more than one basin: one replicate per basin
  outlier         every member of a basin smaller than --min-basin (one-offs are the least trustworthy)
  representative  the member nearest each basin's centroid, so every basin is confirmed to exist in fp64
  screen_failed   points that stopped early or produced no descriptors

`compare` places each fp64 end state in the SCREEN's descriptor space (same columns, same scaling) and
assigns it to the nearest screen basin within --match-tol, or NEW. Writes VERIFY.md + verify.json into the
verify run. Descriptive only: a disagreement is reported, never auto-corrected.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import basin_cluster as bc  # noqa: E402
import irer_specs  # noqa: E402

REASON_ORDER = ("screen_failed", "outlier", "boundary", "ic_split", "representative")


# ----------------------------------------------------------------------------- shared

def _point_dirs(run_dir):
    for d in sorted(os.listdir(run_dir)):
        sd = os.path.join(run_dir, d)
        if os.path.isdir(sd) and os.path.exists(os.path.join(sd, "spec.json")):
            yield d, sd


def _scaling(screen_pts, cols):
    """The exact relative scaling basin_cluster.run used on the screen: log10 for LOG_COLUMNS, otherwise
    divided by the column's median magnitude over the screen."""
    X = np.array([[p["desc"][c] for c in cols] for p in screen_pts], dtype=float)
    med = {}
    for j, c in enumerate(cols):
        if c not in bc.LOG_COLUMNS:
            med[c] = float(np.median(np.abs(X[:, j]))) + 1e-12
    return med


def _z(desc, cols, med):
    return np.array([np.log10(max(desc[c], 1e-300)) if c in bc.LOG_COLUMNS else desc[c] / med[c] for c in cols])


def _load_screen(screen_dir):
    path = os.path.join(screen_dir, "basins.json")
    if not os.path.exists(path):
        raise SystemExit("no basins.json in %s -- run tools/basin_cluster.py on the screen first" % screen_dir)
    basins = json.load(open(path, encoding="utf-8"))
    pts = {p["label"]: p for p in bc.load_points(screen_dir)}
    cols = basins["columns_used"]
    med = _scaling(list(pts.values()), cols)
    label_basin = {m: b for b, info in basins["basins"].items() for m in info["members"]}
    cent = {b: np.mean([_z(pts[m]["desc"], cols, med) for m in info["members"]], axis=0)
            for b, info in basins["basins"].items()}
    return basins, pts, cols, med, label_basin, cent


# ----------------------------------------------------------------------------- plan

def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        p = prefix + k
        if isinstance(v, dict):
            out.update(_flatten(v, p + "."))
        else:
            out[p] = v
    return out


def select(screen_dir, group_by, min_basin=2):
    """-> {screen_label: [reasons]} for every point that should be re-run in fp64."""
    basins, pts, cols, med, label_basin, cent = _load_screen(screen_dir)
    group_by = group_by or basins.get("group_by") or []
    chosen = {}

    def add(label, why):
        chosen.setdefault(label, [])
        if why not in chosen[label]:
            chosen[label].append(why)

    def nearest(labels, b):
        return min(labels, key=lambda m: float(np.linalg.norm(_z(pts[m]["desc"], cols, med) - cent[b])))

    for d, sd in _point_dirs(screen_dir):              # stopped early, or no descriptors at all
        summ = os.path.join(sd, "summary.json")
        s = json.load(open(summ, encoding="utf-8")) if os.path.exists(summ) else {}
        if d not in pts or s.get("stop_reason") != "completed":
            add(d, "screen_failed")
    for b, info in basins["basins"].items():
        if info["size"] < min_basin:
            for m in info["members"]:
                add(m, "outlier")
        add(nearest(info["members"], b), "representative")

    # parameter points: {key: {basin: [labels]}}
    by_point = {}
    for label, p in pts.items():
        key = tuple(bc.axis_value(p["spec"], g) for g in group_by)
        by_point.setdefault(key, {}).setdefault(label_basin[label], []).append(label)

    def one_per_basin(key, why):
        for b, labels in by_point[key].items():
            add(nearest(labels, b), why)

    for key, per_basin in by_point.items():
        if len(per_basin) > 1:
            one_per_basin(key, "ic_split")
    for j, _ in enumerate(group_by):                   # neighbours along each axis, other axes held fixed
        lines = {}
        for key in by_point:
            lines.setdefault(key[:j] + key[j + 1:], []).append(key)
        for keys in lines.values():
            keys.sort(key=lambda k: (k[j] is None, k[j]))
            for a, b in zip(keys, keys[1:]):
                if set(by_point[a]) != set(by_point[b]):
                    one_per_basin(a, "boundary")
                    one_per_basin(b, "boundary")
    for label in chosen:
        chosen[label].sort(key=REASON_ORDER.index)
    return chosen


def build_verify_spec(screen_dir, chosen, N=None, spec_id=None):
    """One fp64 spec whose zip sweep reproduces exactly the chosen screen points (optionally at grid N)."""
    root = json.load(open(os.path.join(screen_dir, "spec.json"), encoding="utf-8"))
    order = sorted(chosen, key=lambda lb: (REASON_ORDER.index(chosen[lb][0]), lb))
    specs = [json.load(open(os.path.join(screen_dir, lb, "spec.json"), encoding="utf-8")) for lb in order]
    flat = [_flatten({k: v for k, v in s.items() if k not in ("id", "title", "sweep")}) for s in specs]
    paths = sorted(set().union(*flat)) if flat else []
    varying = [p for p in paths if len({json.dumps(f.get(p), sort_keys=True) for f in flat}) > 1]
    for p in varying:
        if any(p not in f for f in flat):
            raise SystemExit("cannot rebuild the points: '%s' is missing from some of them" % p)
    v = copy.deepcopy(specs[0]) if specs else copy.deepcopy(root)
    v.pop("sweep", None)
    v["id"] = spec_id or "%s-verify" % root["id"]
    v["title"] = "fp64 verification of %d screen point(s) from %s" % (len(order), os.path.basename(
        os.path.abspath(screen_dir)))
    v["description"] = ("Generated by tools/screen_verify.py from the fp32 screen %s. Points: %s." % (
        os.path.basename(os.path.abspath(screen_dir)),
        "; ".join("%s (%s)" % (lb, ", ".join(chosen[lb])) for lb in order)))[:4000]
    v["protocol"]["precision"] = "fp64"
    if N:
        v["protocol"]["grid"]["N"] = int(N)
    v.pop("requires", None)                            # the screen already ran; nothing to wait for
    axes = {p: [f[p] for f in flat] for p in varying if p not in ("protocol.precision", "protocol.grid.N")}
    if len(order) > 1:
        if not axes:
            raise SystemExit("chosen points do not differ in any parameter -- nothing to sweep")
        v["sweep"] = {"mode": "zip", "axes": axes}
    errs = irer_specs.validate(v)
    if errs:
        raise SystemExit("generated spec is invalid: %s" % errs)
    return v, order


def plan(screen_dir, group_by=None, N=None, min_basin=2, out_dir=None, spec_id=None):
    chosen = select(screen_dir, group_by, min_basin)
    if not chosen:
        raise SystemExit("nothing to verify")
    spec, order = build_verify_spec(screen_dir, chosen, N, spec_id)
    out_dir = out_dir or os.path.join(ROOT, "specs", "proposed")
    os.makedirs(out_dir, exist_ok=True)
    spec_path = os.path.join(out_dir, spec["id"] + ".json")
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    # expand() order == plan order, so compare can map verify point i back to screen label order[i]
    labels = [lb for lb, _ in irer_specs.expand(spec)]
    rows = [{"screen_label": lb, "verify_label": vl or "run", "reasons": chosen[lb]} for lb, vl in zip(order, labels)]
    n_screen = sum(1 for _ in _point_dirs(screen_dir))
    frac = len(rows) / max(n_screen, 1)
    warnings = []
    if frac > 0.5:
        # nearly every point sits next to a basin change: the screen has no coherent basin structure
        # (seen on the a* screen at N=32: fields never localise, so "basins" are a smear of dispersed
        # states). Verifying most of it costs about as much as running it in fp64 to begin with.
        warnings.append("verifying %d of %d screen points (%.0f%%): the screen shows little basin structure "
                        "-- check the grid resolves the states (e.g. d_n_nodes > 0), or screen at a larger N"
                        % (len(rows), n_screen, 100 * frac))
    manifest = {"screen_run": os.path.abspath(screen_dir), "verify_spec": os.path.abspath(spec_path),
                "grid_N": spec["protocol"]["grid"]["N"], "n_screen": n_screen, "warnings": warnings,
                "points": rows}
    with open(os.path.join(screen_dir, "verify_plan.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


# ----------------------------------------------------------------------------- compare

def compare(screen_dir, verify_dir, match_tol=0.2):
    basins, pts, cols, med, label_basin, cent = _load_screen(screen_dir)
    plan_path = os.path.join(screen_dir, "verify_plan.json")
    if not os.path.exists(plan_path):
        raise SystemExit("no verify_plan.json in %s -- run `screen_verify.py plan` first" % screen_dir)
    manifest = json.load(open(plan_path, encoding="utf-8"))
    rows = []
    for r in manifest["points"]:
        vdir = os.path.join(verify_dir, r["verify_label"]) if len(manifest["points"]) > 1 else verify_dir
        summ = os.path.join(vdir, "summary.json")
        row = {"screen_label": r["screen_label"], "reasons": r["reasons"],
               "screen_basin": label_basin.get(r["screen_label"])}
        if not os.path.exists(summ):
            rows.append(dict(row, fp64_basin=None, status="NOT_RUN"))
            continue
        s = json.load(open(summ, encoding="utf-8"))
        desc = s.get("final") or {}
        if s.get("stop_reason") != "completed" or any(c not in desc for c in cols):
            rows.append(dict(row, fp64_basin=None, status="FP64_FAILED", stop_reason=s.get("stop_reason")))
            continue
        z = _z(desc, cols, med)
        dist = {b: float(np.linalg.norm(z - c)) for b, c in cent.items()}
        b = min(dist, key=dist.get)
        fp64 = b if dist[b] <= match_tol else "NEW"
        if row["screen_basin"] is None:
            status = "SCREEN_FAILED_NOW_" + ("NEW" if fp64 == "NEW" else "BASIN_%s" % fp64)
        else:
            status = "AGREE" if fp64 == row["screen_basin"] else "DISAGREE"
        rows.append(dict(row, fp64_basin=fp64, status=status, distance=round(dist[b], 4),
                         n_nodes_fp64=desc.get("d_n_nodes"),
                         n_nodes_screen=(pts.get(r["screen_label"]) or {}).get("desc", {}).get("d_n_nodes")))
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out = {"screen_run": os.path.abspath(screen_dir), "verify_run": os.path.abspath(verify_dir),
           "match_tol": match_tol, "columns": cols, "counts": counts, "rows": rows}
    with open(os.path.join(verify_dir, "verify.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    L = ["# Screen vs fp64 verification", "",
         "Screen `%s` (fp32) against verify run `%s` (fp64, N=%s). Each fp64 end state is placed in the "
         "screen's descriptor space and assigned to the nearest screen basin within %.2g, otherwise NEW. "
         "Generated by `tools/screen_verify.py compare`; descriptive only." % (
             os.path.basename(out["screen_run"]), os.path.basename(out["verify_run"]),
             manifest.get("grid_N"), match_tol), "",
         "**%s**" % ", ".join("%s: %d" % kv for kv in sorted(counts.items())), "",
         "| screen point | why verified | screen basin | fp64 basin | status | nodes fp32 → fp64 |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %s | %s | %s | %s → %s |" % (
            r["screen_label"], ", ".join(r["reasons"]), r["screen_basin"], r["fp64_basin"], r["status"],
            r.get("n_nodes_screen"), r.get("n_nodes_fp64")))
    if counts.get("DISAGREE") or counts.get("NEW"):
        L += ["", "A DISAGREE row means the screen's basin map is wrong at that point: re-run its "
                  "neighbourhood in fp64 before drawing a boundary there."]
    with open(os.path.join(verify_dir, "VERIFY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="choose screen points to re-run in fp64 and write the verify spec")
    p.add_argument("screen_run")
    p.add_argument("--group-by", default="", help="parameter axes defining a point (default: basins.json's)")
    p.add_argument("--N", type=int, default=None, help="verify grid size (default: the screen's)")
    p.add_argument("--min-basin", type=int, default=2, help="basins smaller than this are outliers")
    p.add_argument("--out-dir", default=None, help="where the spec goes (default specs/proposed/)")
    p.add_argument("--id", default=None)
    c = sub.add_parser("compare", help="did fp64 land each point in the screen's basin?")
    c.add_argument("screen_run")
    c.add_argument("verify_run")
    c.add_argument("--match-tol", type=float, default=0.2)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        m = plan(a.screen_run, [g for g in a.group_by.split(",") if g], a.N, a.min_basin, a.out_dir, a.id)
        why = {}
        for r in m["points"]:
            for w in r["reasons"]:
                why[w] = why.get(w, 0) + 1
        print("%d point(s) to verify in fp64 at N=%s (%s)" % (len(m["points"]), m["grid_N"],
                                                              ", ".join("%s %d" % kv for kv in why.items())))
        for w in m["warnings"]:
            print("WARNING:", w)
        print("spec: %s" % m["verify_spec"])
        print("run:  python tools/run_spec.py %s" % os.path.relpath(m["verify_spec"], ROOT))
        print("then: python tools/screen_verify.py compare %s <verify run dir>" % a.screen_run)
    else:
        out = compare(a.screen_run, a.verify_run, a.match_tol)
        print(", ".join("%s: %d" % kv for kv in sorted(out["counts"].items())),
              "->", os.path.join(a.verify_run, "VERIFY.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
