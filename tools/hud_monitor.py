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

TELEMETRY (2026-10-04). If the run writes <run>/telemetry.jsonl (jax_scout.snapshots.TelemetryWriter),
each pass also redraws <run>/rendered/telemetry.png: every scalar against t, and every DECLARED invariant
as |value| on a log axis with its tolerance drawn as a line. Breaches are printed once each. Still
reader-only: a breach is reported, never acted on.

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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import render_fields  # noqa: E402
from jax_scout.snapshots import read_telemetry, invariant_breaches, TELEMETRY_FILE  # noqa: E402

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


def telemetry_dirs(run):
    """Every directory under <run> holding a telemetry.jsonl (multi-arm harnesses write one per arm)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(run):
        dirnames[:] = [d for d in dirnames if d not in ("rendered", "snapshots")]
        if TELEMETRY_FILE in filenames:
            out.append(dirpath)
    return sorted(out)


def has_telemetry(run):
    return bool(telemetry_dirs(run))


def render_telemetry(run, *, dpi=90, tel_dir=None):
    """Redraw rendered/telemetry[_<arm>].png from tel_dir (default: run). -> (path or None, breaches)."""
    tel_dir = tel_dir or run
    meta, rows = read_telemetry(tel_dir)
    if not rows:
        return None, []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inv = meta.get("invariants") or {}
    keys = [k for k in rows[-1] if k not in ("t", "wall_s") and isinstance(rows[-1][k], (int, float))]
    inv_keys = [k for k in keys if k in inv]
    other = [k for k in keys if k not in inv]
    panels = inv_keys + other[:max(0, 8 - len(inv_keys))]
    if not panels:
        return None, []
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.8 * n), sharex=True, squeeze=False)
    t = [r.get("t") for r in rows]
    for ax, k in zip(axes[:, 0], panels):
        ys = [r.get(k) for r in rows]
        if k in inv:
            ax.semilogy(t, [abs(y) if isinstance(y, (int, float)) and y != 0 else float("nan") for y in ys], lw=1)
            ax.axhline(inv[k], color="crimson", ls="--", lw=1)
            ax.set_ylabel("|%s|" % k, fontsize=7)
        else:
            ax.plot(t, ys, lw=1)
            ax.set_ylabel(k, fontsize=7)
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("t")
    breaches = invariant_breaches(meta, rows)
    rel = os.path.relpath(tel_dir, run)
    parts = [] if rel == "." else [x for x in rel.split(os.sep) if x != "telemetry"]
    tag = "".join("_" + x for x in parts)
    fig.suptitle("%s%s  (%d samples, %d invariant breaches)" % (os.path.basename(run), tag, len(rows),
                                                              len(breaches)), fontsize=8)
    fig.tight_layout()
    out_dir = os.path.join(run, "rendered")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "telemetry%s.png" % tag)
    tmp = path + ".part.png"
    fig.savefig(tmp, dpi=dpi)
    plt.close(fig)
    os.replace(tmp, path)
    return path, breaches


def latest_run():
    if not os.path.isdir(SWEEPS):
        return None
    cands = []
    for name in os.listdir(SWEEPS):
        p = os.path.join(SWEEPS, name)
        if os.path.isdir(p) and (snapshot_dirs(p) or has_telemetry(p)):
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
    for tdir in telemetry_dirs(run):
        tel = os.path.join(tdir, TELEMETRY_FILE)
        size = os.path.getsize(tel)
        if size != seen.get(tel):
            try:
                path, breaches = render_telemetry(run, dpi=dpi, tel_dir=tdir)
            except Exception as exc:             # a render fault must not end the watch
                print("  telemetry render failed: %s" % exc)
                path, breaches = None, []
            seen[tel] = size
            reported = seen.setdefault("_breaches", set())
            for name, t, v, tol in breaches:
                if (tdir, name) not in reported:
                    reported.add((tdir, name))
                    print("  INVARIANT BREACH  %s/%s = %.3e at t=%s (tolerance %.1e)"
                          % (os.path.relpath(tdir, run), name, v, t, tol))
                    note = (read_telemetry(tdir)[0] or {}).get("note")
                    if note:
                        print("      harness note: %s" % note)
            if path:
                made.append(path)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run directory to watch (contains snapshots/)")
    src.add_argument("--latest", action="store_true",
                     help="watch the most recently modified run under sweep_runs/ with snapshots or telemetry")
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
    if not snapshot_dirs(run) and not has_telemetry(run):
        print("  (no snapshots or telemetry yet - waiting)")

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
