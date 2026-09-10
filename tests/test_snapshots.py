"""Snapshot-writer contract tests — the observer must not perturb the observed.

The HUD's whole justification is that it is a second, independent detector. That only holds if
turning it on cannot change what the simulation computes. These tests enforce the four clauses of
the non-perturbation contract in jax_scout/snapshots.py:

  1. a disabled writer is inert
  2. capture is a PURE READ - the state it is handed is unchanged, bit for bit
  3. a full queue DROPS rather than blocks (a stalled 5.68-hour run is worse than a missing frame)
  4. a failing write cannot propagate into the run

Plus the shape/dtype contract the renderer depends on.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from jax_scout.snapshots import SnapshotWriter, _centre_slices, _downsample


def _cube(n=8, dtype=np.complex128, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n, n))
    if np.dtype(dtype).kind == "c":
        a = a + 1j * rng.standard_normal((n, n, n))
    return a.astype(dtype)


# ---------------------------------------------------------------- primitives

def test_centre_slices_are_the_actual_centre_planes():
    a = _cube(8)
    h = 8 // 2
    s = _centre_slices(a)
    assert np.allclose(s["yz"], a[h, :, :])
    assert np.allclose(s["xz"], a[:, h, :])
    assert np.allclose(s["xy"], a[:, :, h])


def test_slices_are_downcast_for_size():
    """complex128 -> complex64 and float64 -> float32; the renderer relies on this."""
    assert _centre_slices(_cube(8, np.complex128))["xy"].dtype == np.complex64
    assert _centre_slices(_cube(8, np.float64))["xy"].dtype == np.float32


def test_downsample_respects_the_target_and_keeps_the_cube():
    out = _downsample(_cube(96, np.complex128), target=48)
    assert out.ndim == 3 and out.shape[0] == out.shape[1] == out.shape[2]
    assert out.shape[0] <= 48
    assert out.dtype == np.complex64


# ---------------------------------------------------------------- the contract

def test_disabled_writer_is_completely_inert(tmp_path):
    """Clause 1. A harness must be able to call capture() unconditionally."""
    w = SnapshotWriter(tmp_path / "snaps", enabled=False)
    assert w.capture(0.0, {"phi": _cube()}) is False
    assert w.close() == {}
    assert not (tmp_path / "snaps").exists()          # not even a directory
    assert w.n_written == 0


def test_capture_does_not_mutate_the_state_it_is_given(tmp_path):
    """Clause 2 -- the load-bearing one. Capture is a pure read."""
    a = _cube(8)
    before = a.copy()
    w = SnapshotWriter(tmp_path / "snaps", enabled=True)
    w.capture(1.0, {"phi": a}, dx=0.25, extra={"F_R": -5e-5})
    w.close()
    assert np.array_equal(a, before), "capture mutated the field it was handed"
    assert a.dtype == before.dtype


def test_full_queue_drops_and_never_blocks(tmp_path):
    """Clause 3. With a size-1 queue and a wedged writer, capture must return promptly."""
    w = SnapshotWriter(tmp_path / "snaps", enabled=True, maxqueue=1)
    w._thread = None                                  # simulate a writer that is not draining
    w._q.put((str(tmp_path / "blocker.npz"), {"t": np.asarray(0.0)}))   # fill it
    t0 = time.time()
    for i in range(20):
        w.capture(float(i), {"phi": _cube(8)})
    elapsed = time.time() - t0
    assert elapsed < 5.0, "capture blocked on a full queue (%.1fs)" % elapsed
    assert w.n_dropped > 0, "a full queue should record drops"


def test_write_failure_is_swallowed_and_counted(tmp_path):
    """Clause 4. An unwritable path must not raise into the run."""
    w = SnapshotWriter(tmp_path / "snaps", enabled=True)
    w._q.put((os.path.join(str(tmp_path), "nonexistent-dir", "x.npz"),
              {"t": np.asarray(0.0)}))
    stats = w.close()
    assert stats["frames_failed"] >= 1
    assert stats["frames_written"] == 0


def test_non_cubic_fields_are_skipped_not_crashed(tmp_path):
    w = SnapshotWriter(tmp_path / "snaps", enabled=True)
    ok = w.capture(0.0, {"good": _cube(8), "scalars": np.zeros(10), "flat": np.zeros((4, 5, 6))})
    w.close()
    assert ok
    with np.load(tmp_path / "snaps" / "snap_000000.npz") as z:
        keys = set(z.files)
    assert {"good__xy", "good__yz", "good__xz"} <= keys
    assert not any(k.startswith(("scalars", "flat")) for k in keys)


# ---------------------------------------------------------------- cadence & payload

def test_every_n_controls_cadence(tmp_path):
    w = SnapshotWriter(tmp_path / "snaps", enabled=True, every=3)
    taken = [w.capture(float(i), {"phi": _cube(8)}) for i in range(9)]
    w.close()
    assert taken == [True, False, False, True, False, False, True, False, False]
    assert len(list((tmp_path / "snaps").glob("snap_*.npz"))) == 3


def test_volume_every_adds_a_downsampled_cube(tmp_path):
    w = SnapshotWriter(tmp_path / "snaps", enabled=True, volume_every=2, volume_target=4)
    w.capture(0.0, {"phi": _cube(8)})
    w.capture(1.0, {"phi": _cube(8)})
    w.close()
    with np.load(tmp_path / "snaps" / "snap_000000.npz") as z:
        assert "phi__vol" in z.files and z["phi__vol"].shape[0] <= 4
    with np.load(tmp_path / "snaps" / "snap_000001.npz") as z:
        assert "phi__vol" not in z.files          # only every 2nd frame


def test_manifest_records_the_loss_accounting(tmp_path):
    w = SnapshotWriter(tmp_path / "snaps", enabled=True)
    w.capture(0.0, {"phi": _cube(8)})
    stats = w.close()
    assert stats["frames_written"] == 1
    with open(tmp_path / "snaps" / "MANIFEST.json", encoding="utf-8") as fh:
        import json
        m = json.load(fh)
    assert m["frames_seen"] == 1 and m["frames_dropped"] == 0


def test_snapshot_payload_matches_what_the_renderer_expects(tmp_path):
    """The renderer discovers fields by the '<name>__<plane>' convention and the cube rule."""
    w = SnapshotWriter(tmp_path / "snaps", enabled=True)
    w.capture(2.5, {"phi": _cube(8), "A_minus_1": _cube(8, np.float64)},
              dx=0.25, extra={"F_R": -5.7e-5})
    w.close()
    with np.load(tmp_path / "snaps" / "snap_000000.npz") as z:
        assert float(z["t"]) == 2.5
        assert float(z["dx"]) == 0.25
        assert float(z["scalar__F_R"]) == pytest.approx(-5.7e-5)
        assert z["phi__xy"].dtype == np.complex64
        assert z["A_minus_1__xy"].dtype == np.float32
