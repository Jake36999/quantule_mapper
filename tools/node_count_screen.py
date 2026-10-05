"""Corpus-wide screen for the C2.8b signature: a node count that changes when it should not.

WHY. The C2 paired-reading queue has 102 entries and reviewing them one at a time is the slow way
to find the thing worth finding. C2.8b -- a hand-rolled peak tracker that failed when two cores
overlapped, producing an elasticity of 3.21 and an apparent energy-conservation violation that was
never in the physics -- announced itself as an unstable node count. That is mechanically detectable
across the whole corpus, so this ranks the queue instead of working it in arbitrary order.

WHAT IT CHECKS. Within one run, packs that differ only by a sample index (`..._sample000.npz`,
`..._sample020.npz`) are the same configuration at different times. For a fixed field, the number of
nodes across those samples should change only when the physics changes it -- a merge, a split, a
decay. A count that flickers up and down is the signature.

WHAT IT IS NOT. It is a SCREEN, not a verdict. A changing count can be real (cores genuinely merge),
and `tools/render_fields.find_centroids` has known false positives on ring-structured fields, both
documented in its docstring. Output is a ranked list of runs to look at, ordered by how badly the
count misbehaves. Every hit still needs a human reading; the point is to put the likely ones first.

Usage:
    python tools/node_count_screen.py
    python tools/node_count_screen.py --limit 40 --out runtime_logs/node_count_screen.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_fields as rf  # noqa: E402

SAMPLE_RE = re.compile(r"^(?P<stem>.+?)_?sample(?P<idx>\d+)\.npz$", re.IGNORECASE)


def series(run_dir):
    """Group a run's packs into {(series stem): [(index, path)]}, for packs that form a time series."""
    out = collections.defaultdict(list)
    for dp, dn, fns in os.walk(run_dir):
        dn[:] = [d for d in dn if d not in ("rendered", "snapshots")]
        for fn in fns:
            m = SAMPLE_RE.match(fn)
            if m:
                key = os.path.join(os.path.relpath(dp, run_dir), m.group("stem"))
                out[key].append((int(m.group("idx")), os.path.join(dp, fn)))
    return {k: sorted(v) for k, v in out.items() if len(v) >= 3}


def counts_for(path):
    """{field: n_nodes} for every cube field in a pack."""
    got = {}
    try:
        with np.load(path, allow_pickle=False) as z:
            for k in z.files:
                a = z[k]
                if not rf.is_cube(a):
                    continue
                got[k] = len(rf.find_centroids(rf.centre_plane(a)))
    except Exception:
        return {}
    return got


def instability(seq):
    """Score a count sequence. Monotone changes are cheap; reversals are the signature.

    A merge (2,2,1,1) or a split (1,1,2,2) is physics and scores 0 reversals. A flicker
    (2,1,2,1) is a tracker failing to decide, and that is what C2.8b looked like.
    """
    d = [b - a for a, b in zip(seq, seq[1:])]
    nz = [x for x in d if x != 0]
    reversals = sum(1 for a, b in zip(nz, nz[1:]) if a * b < 0)
    return reversals, (max(seq) - min(seq)) if seq else 0


def main():
    ap = argparse.ArgumentParser(description="corpus screen for unstable node counts")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many runs")
    ap.add_argument("--out", default="runtime_logs/node_count_screen.json")
    args = ap.parse_args()

    runs = sorted(d for d in os.listdir(rf.SWEEP) if os.path.isdir(os.path.join(rf.SWEEP, d)))
    if args.limit:
        runs = runs[:args.limit]

    findings, n_series, n_runs = [], 0, 0
    for run in runs:
        rd = os.path.join(rf.SWEEP, run)
        ser = series(rd)
        if not ser:
            continue
        n_runs += 1
        for key, items in ser.items():
            n_series += 1
            per_field = collections.defaultdict(list)
            for _idx, path in items:
                for field, n in counts_for(path).items():
                    per_field[field].append(n)
            for field, seq in per_field.items():
                if len(seq) < 3 or len(set(seq)) == 1:
                    continue
                rev, spread = instability(seq)
                findings.append({"run": run, "series": key.replace("\\", "/"), "field": field,
                                 "counts": seq, "reversals": rev, "spread": spread})
        print("  scanned %-52s (%d series)" % (run[:52], len(ser)), flush=True)

    findings.sort(key=lambda f: (-f["reversals"], -f["spread"]))
    flick = [f for f in findings if f["reversals"] > 0]

    print("")
    print("scanned %d run(s), %d time series" % (n_runs, n_series))
    print("%d field-series have a changing node count; %d of those REVERSE direction"
          % (len(findings), len(flick)))
    print("")
    if flick:
        print("ranked -- reversals first, these are the ones to read:")
        for f in flick[:25]:
            print("  rev=%d spread=%d  %-34s %-28s %s"
                  % (f["reversals"], f["spread"], f["run"][:34], f["field"][:28],
                     "/".join(str(c) for c in f["counts"])))
    else:
        print("no reversing series found -- every changing count moves monotonically,")
        print("which is what a genuine merge or split looks like.")

    outp = os.path.join(rf.REPO, args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump({"n_runs": n_runs, "n_series": n_series,
                   "n_changing": len(findings), "n_reversing": len(flick),
                   "findings": findings}, fh, indent=2)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
