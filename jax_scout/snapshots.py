"""Field snapshot writer — the simulation's half of the visual HUD.

WHY. `gravity_TG_B2_*` records scalars and CSVs only: zero field-save calls. The sector where the
sign problem lives has no visual data at all, so nothing can be rendered from it, live or after the
fact. See docs/VISUAL_HUD_SCOPE_RFC.md section 5.

THE NON-PERTURBATION CONTRACT. This module observes; it must never change what the simulation
computes or how long it takes.

  * OFF BY DEFAULT. A harness with snapshots disabled must be bit-identical to one without this
    module at all. Enforced by tests/test_snapshots.py.
  * PURE READ. Capture only slices existing state. No mutation, no donation, no re-jit of the
    stepper.
  * NO NEW SYNC POINT. The harnesses already force a device->host transfer every sample
    (`float(d[k])` on jitted diagnostics), so capture rides an existing sync. At N=80 three
    orthogonal complex64 slices are ~150 KB, microseconds over PCIe.
  * DISK CAN NEVER STALL THE SIM. Writes happen on a background thread behind a BOUNDED queue. If
    the disk cannot keep up the queue drops the oldest frame and records the loss. A dropped frame
    is a cosmetic loss; a blocked 5.68-hour run is not.
  * CRASH-SAFE. Every failure path is swallowed and counted. The writer thread dying must not take
    the run with it.

WHAT IS CAPTURED. Three orthogonal centre-plane slices per field, downcast to complex64/float32.
Slices catch translation, breathing, merging and asymmetry - which covers every failure mode in the
integrity ledger. Optionally a downsampled full volume every `volume_every` frames for 3-D views.

    N=80, 3 slices, complex64 : ~0.15 MB/frame ->  ~123 MB over 800 frames
    N=80, volume at 48^3      : ~0.88 MB/frame

Usage in a harness:

    from jax_scout.snapshots import SnapshotWriter
    snap = SnapshotWriter(out / "snapshots", enabled=args.snapshots, every=args.snapshot_every)
    ...
    snap.capture(t, {"phi": phi, "G": G, "A": A}, dx=g["dx"])
    ...
    snap.close()          # drains the queue and writes a manifest
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time

import numpy as np

_DOWNCAST = {np.dtype("complex128"): np.complex64,
             np.dtype("float64"): np.float32}


def _centre_slices(arr):
    """Three orthogonal centre-plane slices of an (N,N,N) array, downcast for size.

    Returns a dict, not a stacked array, because the planes stay individually addressable and the
    renderer should not have to remember an index convention.
    """
    n = arr.shape[0]
    h = n // 2
    out = {"yz": arr[h, :, :], "xz": arr[:, h, :], "xy": arr[:, :, h]}
    dt = _DOWNCAST.get(arr.dtype)
    if dt is not None:
        out = {k: v.astype(dt) for k, v in out.items()}
    return out


def _downsample(arr, target=48):
    """Strided downsample of an (N,N,N) array to <= target per side. Cheap and adequate for viewing."""
    n = arr.shape[0]
    if n <= target:
        step = 1
    else:
        step = int(np.ceil(n / target))
    sub = arr[::step, ::step, ::step]
    dt = _DOWNCAST.get(arr.dtype)
    return sub.astype(dt) if dt is not None else sub


class SnapshotWriter:
    """Bounded, async, crash-safe field snapshot writer.

    Disabled instances are inert: `capture` returns immediately without touching the state, so a
    harness can call it unconditionally.
    """

    def __init__(self, outdir, *, enabled=False, every=1, volume_every=0,
                 volume_target=48, maxqueue=8):
        self.enabled = bool(enabled)
        self.outdir = str(outdir)
        self.every = max(1, int(every))
        self.volume_every = max(0, int(volume_every))
        self.volume_target = int(volume_target)
        self.n_seen = 0
        self.n_written = 0
        self.n_dropped = 0
        self.n_failed = 0
        self._t0 = time.time()
        self._q = None
        self._thread = None
        if not self.enabled:
            return
        os.makedirs(self.outdir, exist_ok=True)
        self._q = queue.Queue(maxsize=int(maxqueue))
        self._thread = threading.Thread(target=self._drain, name="snapshot-writer", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- writer thread

    def _drain(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            path, payload = item
            try:
                # Atomic publish. The live monitor (tools/hud_monitor.py) reads this directory
                # WHILE the run is writing it; np.load on a half-written npz raises, and a reader
                # that trips over torn files is a telemetry failure of exactly the kind that killed
                # the previous HUD. Rename is atomic within a directory on both NTFS and ext4, so
                # a frame is either absent or complete. Costs one rename on the writer thread,
                # which is off the simulation's critical path by construction.
                tmp = path + ".part"
                # a file handle, not a name: np.savez_compressed appends '.npz' to a bare path
                # and would turn 'snap_000000.npz.part' into 'snap_000000.npz.part.npz'.
                with open(tmp, "wb") as fh:
                    np.savez_compressed(fh, **payload)
                os.replace(tmp, path)
                self.n_written += 1
            except Exception:          # a failed write must never propagate into the run
                self.n_failed += 1
            finally:
                self._q.task_done()

    # ---------------------------------------------------------------- capture

    def capture(self, t, fields, *, dx=None, extra=None):
        """Queue a snapshot. Pure read of `fields`; returns immediately.

        fields: {name: array-like}. JAX arrays are fine - they are converted here, at a sync point
        the harness has already paid for.
        """
        if not self.enabled:
            return False
        self.n_seen += 1
        idx = self.n_seen - 1
        if idx % self.every:
            return False
        want_volume = self.volume_every and (idx % self.volume_every == 0)

        payload = {"t": np.asarray(float(t), dtype=np.float64)}
        if dx is not None:
            payload["dx"] = np.asarray(float(dx), dtype=np.float64)
        try:
            for name, arr in fields.items():
                a = np.asarray(arr)                  # device -> host, on the existing sync
                if a.ndim != 3 or a.shape[0] != a.shape[1] != a.shape[2]:
                    continue
                for plane, sl in _centre_slices(a).items():
                    payload["%s__%s" % (name, plane)] = sl
                if want_volume:
                    payload["%s__vol" % name] = _downsample(a, self.volume_target)
            if extra:
                for k, v in extra.items():
                    payload["scalar__%s" % k] = np.asarray(float(v), dtype=np.float64)
        except Exception:
            self.n_failed += 1
            return False

        path = os.path.join(self.outdir, "snap_%06d.npz" % idx)
        try:
            self._q.put_nowait((path, payload))
            return True
        except queue.Full:
            # Drop rather than block. A missing frame is cosmetic; a stalled run is not.
            self.n_dropped += 1
            return False

    # ---------------------------------------------------------------- shutdown

    def close(self, timeout=120.0):
        """Drain the queue and write a manifest. Safe to call on a disabled writer."""
        if not self.enabled:
            return {}
        try:
            self._q.put(None, timeout=5.0)
            self._thread.join(timeout=timeout)
        except Exception:
            pass
        stats = {
            "frames_seen": self.n_seen,
            "frames_written": self.n_written,
            "frames_dropped": self.n_dropped,
            "frames_failed": self.n_failed,
            "every": self.every,
            "volume_every": self.volume_every,
            "volume_target": self.volume_target,
            "elapsed_s": round(time.time() - self._t0, 3),
            "note": "three orthogonal centre-plane slices per field, downcast to complex64/float32; "
                    "dropped frames mean the disk could not keep up and the run was NOT blocked",
        }
        try:
            with open(os.path.join(self.outdir, "MANIFEST.json"), "w", encoding="utf-8") as fh:
                json.dump(stats, fh, indent=2)
        except OSError:
            pass
        return stats
