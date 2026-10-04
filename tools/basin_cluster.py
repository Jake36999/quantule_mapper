#!/usr/bin/env python
"""Basin mapping, stage 2: cluster final states into basins (IMPLEMENTATION_PLAN_2026-10 Phase F2).

    python tools/basin_cluster.py sweep_runs/<ENSEMBLE_RUN> --group-by param_a [--min-cluster 2]

WHY. The Hunter's GA and the inverse-GP manifold tracker blend distinct basins: they average over
parameter points and seeds before asking whether the end states are the same kind of object
(docs/instrument_integrity/SEARCH_STACK_AUDIT_2026-10.md). This does it the other way round: run many
ICs/seeds per parameter point FORWARD (a spec with a sweep, observer `state_descriptors`), then group
the END STATES by what they are.

Input: the run directory written by tools/run_spec.py for a sweep spec (one sub-directory per point,
each with spec.json + summary.json). `--group-by` names the sweep axes that define a PARAMETER POINT
(e.g. param_a); every other axis (seed, K, IC) is treated as a replicate whose end state is clustered.

Method: intensive SHAPE descriptors (SHAPE_COLUMNS) are put on a RELATIVE scale (log10 for
scale-like columns, otherwise divided by the column's median magnitude). Replicates at each parameter point are clustered by average-linkage cut at `--match-tol`
(relative units), then local clusters are matched ACROSS points by nearest centroid within the same radius, so
a basin that exists at two parameter values keeps one label -- what lets basins be followed across
parameters before continuation (stage 3) takes over.

Output: <run>/basins.json and <run>/BASINS.md. Descriptive: basins are labelled, never ranked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


LOG_COLUMNS = {"d_mass", "d_amp", "d_contrast"}
SHOW = ("d_mass", "d_contrast", "d_n_nodes", "d_pair_mean", "d_rgyr")


def load_points(run_dir):
    pts = []
    for d in sorted(os.listdir(run_dir)):
        sd = os.path.join(run_dir, d)
        sp, ss = os.path.join(sd, "spec.json"), os.path.join(sd, "summary.json")
        if not (os.path.isdir(sd) and os.path.exists(sp) and os.path.exists(ss)):
            continue
        summ = json.load(open(ss, encoding="utf-8"))
        final = summ.get("final") or {}
        desc = {k: v for k, v in final.items() if k.startswith("d_") and isinstance(v, (int, float))}
        if not desc:
            continue
        pts.append({"label": d, "spec": json.load(open(sp, encoding="utf-8")), "desc": desc,
                    "stop_reason": summ.get("stop_reason")})
    return pts


def axis_value(spec, name):
    """Find a parameter by its last path component anywhere in substrate.params / protocol.ic.params."""
    for where in (spec.get("substrate", {}).get("params", {}), spec.get("protocol", {}).get("ic", {}).get("params", {}),
                  spec.get("protocol", {})):
        if name in where:
            return where[name]
    return None


def cluster(X, tol=0.2):
    """Average-linkage agglomerative clustering cut at distance `tol` (relative units): end states closer than
    tol in standardised shape space are one basin. Deterministic, and the same radius as the
    cross-point matching. (HDBSCAN was tried first and split basins on 1% jitter -- see
    tests/test_basin_cluster.py.)"""
    if len(X) < 2:
        return np.zeros(len(X), int), "single"
    from sklearn.cluster import AgglomerativeClustering
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=float(tol), linkage="average").fit_predict(X)
    return lab, "agglomerative(average, cut %.2g)" % tol


#: Intensive SHAPE descriptors: what kind of object, not how big. Extensive ones (d_mass, d_amp) vary
#: continuously with parameters and IC amplitude inside one basin and would split it arbitrarily (seen in
#: the first KG demo, 2026-10-04); they are reported but not clustered on unless --columns asks.
SHAPE_COLUMNS = ("d_contrast", "d_n_nodes", "d_node_size_mean", "d_node_size_std", "d_pair_mean",
                 "d_pair_std", "d_pair_min", "d_rgyr", "d_k_mean", "d_aniso")


def run(run_dir, group_by, min_cluster=2, columns=None, match_tol=0.2):
    """Cluster replicates WITHIN each parameter point, then match local clusters ACROSS points by
    nearest centroid (relative-space distance <= match_tol) so one basin keeps one label along a parameter."""
    pts = load_points(run_dir)
    if len(pts) < 2:
        raise SystemExit("need >= 2 points with state_descriptors in %s" % run_dir)
    common = sorted(set.intersection(*[set(p["desc"]) for p in pts]))
    cols = [c for c in (columns or SHAPE_COLUMNS) if c in common]
    X = np.array([[p["desc"][c] for c in cols] for p in pts], dtype=float)
    for j, c in enumerate(cols):
        if c in LOG_COLUMNS:
            X[:, j] = np.log10(np.maximum(X[:, j], 1e-300))
    # RELATIVE scale, not z-scores: z-scoring blows a column that only carries 1% jitter up to unit
    # variance, so noise would dominate the distance (the first synthetic test split 2 basins into 12).
    # Log columns are compared as log10 differences; the others relative to their median magnitude.
    # Distance 0.1 ~ "10% different in one descriptor".
    Z = X.copy()
    for j, c in enumerate(cols):
        if c not in LOG_COLUMNS:
            Z[:, j] = X[:, j] / (np.median(np.abs(X[:, j])) + 1e-12)
    keep = Z.std(0) > 1e-12
    Z = Z[:, keep] if keep.any() else np.zeros((len(pts), 1))

    groups = {}
    for i, p in enumerate(pts):
        key = tuple((g, json.dumps(axis_value(p["spec"], g))) for g in group_by)
        groups.setdefault(key, []).append(i)
    how = "single"
    local = []                                             # (group key, member indices, centroid)
    for key, idx in groups.items():
        if len(idx) >= 2 and keep.any():
            lab, how = cluster(Z[idx], match_tol)
        else:
            lab = np.zeros(len(idx), int)
        for lb in sorted(set(int(x) for x in lab)):
            mem = [idx[k] for k in range(len(idx)) if lab[k] == lb]
            if lb == -1:                                   # noise: each is its own local cluster
                local += [(key, [m], Z[m]) for m in mem]
            else:
                local.append((key, mem, Z[mem].mean(0)))
    # match local clusters across points: greedy, nearest existing global basin within match_tol
    gl_cent, gl_members, assign = [], [], {}
    for key, mem, c in sorted(local, key=lambda t: -len(t[1])):
        d = [float(np.linalg.norm(c - g)) for g in gl_cent]
        if d and min(d) <= match_tol:
            b = int(np.argmin(d))
            n0 = len(gl_members[b])
            gl_cent[b] = (gl_cent[b] * n0 + c * len(mem)) / (n0 + len(mem))
            gl_members[b] += mem
        else:
            b = len(gl_cent)
            gl_cent.append(c)
            gl_members.append(list(mem))
        for m in mem:
            assign[m] = b
    allc = sorted(common)
    basins = {str(b): {"size": len(mem),
                       "centroid": {c: float(np.mean([pts[m]["desc"][c] for m in mem])) for c in allc},
                       "members": [pts[m]["label"] for m in mem]} for b, mem in enumerate(gl_members)}
    table = []
    for key, idx in sorted(groups.items(), key=lambda kv: str(kv[0])):
        counts = {}
        for i in idx:
            counts[str(assign[i])] = counts.get(str(assign[i]), 0) + 1
        table.append({"point": {g: json.loads(v) for g, v in key}, "n": len(idx), "basins": counts})
    out = {"run": os.path.abspath(run_dir), "method": "%s within point; cross-point match tol %.2g" % (how, match_tol),
           "columns": cols, "columns_used": [c for c, k in zip(cols, keep) if k], "group_by": group_by,
           "basins": basins, "by_point": table}
    with open(os.path.join(run_dir, "basins.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    L = ["# Basins: %s" % os.path.basename(os.path.abspath(run_dir)), "",
         "Clustered %d end states (%s) on shape descriptors %s. Generated by `tools/basin_cluster.py`; "
         "labels are descriptive, never ranked." % (len(pts), out["method"], ", ".join(out["columns_used"]) or "(none vary)"),
         "", "## Basins", "",
         "| basin | size | " + " | ".join(SHOW) + " |", "|---|---|" + "---|" * len(SHOW)]
    for b, info in basins.items():
        L.append("| %s | %d | " % (b, info["size"]) + " | ".join(
            "%.4g" % info["centroid"].get(c, float("nan")) for c in SHOW) + " |")
    L += ["", "## By parameter point", "", "| point | replicates | basin counts |", "|---|---|---|"]
    for row in table:
        L.append("| %s | %d | %s |" % (", ".join("%s=%s" % kv for kv in row["point"].items()) or "(all)", row["n"],
                                        ", ".join("%s: %d" % kv for kv in sorted(row["basins"].items()))))
    with open(os.path.join(run_dir, "BASINS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--group-by", default="", help="comma list of parameter names defining a point")
    ap.add_argument("--min-cluster", type=int, default=2)
    ap.add_argument("--match-tol", type=float, default=0.2,
                    help="basin radius in relative shape units (0.1 ~ 10%% different in one descriptor)")
    ap.add_argument("--columns", default="", help="override descriptor columns (comma list)")
    args = ap.parse_args(argv)
    out = run(args.run_dir, [g for g in args.group_by.split(",") if g], args.min_cluster,
              [c for c in args.columns.split(",") if c] or None, args.match_tol)
    print("%d basins (%s) -> %s" % (len(out["basins"]), out["method"], os.path.join(args.run_dir, "BASINS.md")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
