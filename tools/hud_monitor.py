"""HUD item 6.4 - live monitor: watch a running simulation without touching it.

WHY THIS SHAPE. The previous HUD died of telemetry problems and button failures: it was coupled to
the simulation, so when the UI misbehaved the run was affected, and when the run misbehaved the UI
lied. This one is a SEPARATE PROCESS with exactly one channel - the filesystem - and it is
strictly reader-only:

  * it never writes into the snapshot directory, only into <run>/rendered/;
  * it holds no lock and opens nothing for writing that the run can see;
  * it has NO buttons. Its entire interface is --run, --interval and Ctrl-C. There is no control
    path from this process to the simulation, so there is nothing for a UI fault to break;
  * a torn or unreadable frame is skipped and retried next tick rather than raising. The writer
    publishes frames by atomic rename (jax_scout/snapshots.py), so this should not happen; the
    tolerance is there because "should not happen" is not a guarantee.

Killing this process at any moment is safe. Starting it against a finished run just renders it
once and exits.

Usage:
    python tools/hud_monitor.py --run sweep_runs/MY_RUN
    python tools/hud_monitor.py --latest --interval 20
    python tools/hud_monitor.py --run sweep_runs/MY_RUN --once
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_fields  # noqa: E402

SWEEPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweep_runs")
SNAP_RE = re.compile(r"snap_\d+\.npz$")


def snapshot_dirs(run):
    """Every directory under <run>/snapshots that actually holds frames."""
    root = os.path.join(run, "snapshots")
    if not os.path.isdir(root):
        return []
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(SNAP_RE.match(f) for f in filenames):
            out.append(dirpath)
    return sorted(out)


def frame_count(d):
    try:
        return sum(1 for f in os.listdir(d) if SNAP_RE.match(f))
    except OSError:
        return 0


def run_is_finished(run):
    """A finished run stops growing. RUN_COMPLETE.json is written by the harnesses on success;
    absence of it is not failure, so the caller also needs an idle timeout."""
    return os.path.exists(os.path.join(run, "RUN_COMPLETE.json"))


def latest_run():
    if not os.path.isdir(SWEEPS):
        return None
    cands = []
    for name in os.listdir(SWEEPS):
        p = os.path.join(SWEEPS, name)
        if os.path.isdir(p) and snapshot_dirs(p):
            cands.append((os.path.getmtime(p), p))
    return max(cands)[1] if cands else None


def tick(run, *, seen, dpi):
    """Re-render any snapshot directory that has grown since the last pass. Returns what it made."""
    made = []
    for d in snapshot_dirs(run):
        n = frame_count(d)
        if n == 0 or n == seen.get(d):
            continue
        try:
            paths, err = render_fields.render_snapshots(d, os.path.join(run, "rendered"), dpi=dpi)
        except Exception as exc:                 # a render fault must not end the watch
            print("  render failed (%s): %s" % (os.path.basename(d), exc))
            continue
        seen[d] = n
        if err:
            print("  %s: %s" % (os.path.basename(d), err))
        for p in paths:
            print("  %3d frames -> %s" % (n, os.path.relpath(p, run)))
        made += paths
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run directory to watch (contains snapshots/)")
    src.add_argument("--latest", action="store_true",
                     help="watch the most recently modified run under sweep_runs/ that has snapshots")
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between passes")
    ap.add_argument("--once", action="store_true", help="render once and exit")
    ap.add_argument("--idle-exit", type=float, default=900.0,
                    help="stop after this many seconds with no new frames (0 = never)")
    ap.add_argument("--dpi", type=int, default=90, help="lower than the offline default; this redraws often")
    args = ap.parse_args()

    run = args.run or latest_run()
    if not run:
        print("no run with snapshots found under %s" % SWEEPS)
        return 2
    run = os.path.abspath(run)
    if not os.path.isdir(run):
        print("no such run: %s" % run)
        return 2
    print("watching %s" % run)
    if not snapshot_dirs(run):
        print("  (no snapshots yet - the harness needs --snapshots; waiting)")

    seen, last_change = {}, time.time()
    while True:
        made = tick(run, seen=seen, dpi=args.dpi)
        if made:
            last_change = time.time()
        if args.once:
            return 0
        idle = time.time() - last_change
        if run_is_finished(run) and not made:
            print("run complete and nothing new to render - done")
            return 0
        if args.idle_exit and idle > args.idle_exit:
            print("no new frames for %.0f s - stopping" % idle)
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("stopped")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
