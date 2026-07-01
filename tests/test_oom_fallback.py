"""
test_oom_fallback.py
Vector 5 (P1) gate — validation_pipeline.ArtifactLoader._anti_aliased_downsample
must handle oversized datasets gracefully (no MemoryError raise) and return a
valid numpy ndarray.

Strategy:
  - Set ASTE_ANTI_ALIAS_MAX_SOURCE_ELEMENTS to a tiny value (64) via env var so
    that even a small synthetic array triggers the OOM fallback branch.
  - Create a real h5py Dataset in a temporary file so the function receives an
    actual Dataset object (not a mock), exercising the genuine strided-read path.
  - Assert: result is ndarray, shape is non-empty, no exception raised, and the
    number of elements returned is ≤ the original (downsampled, not inflated).
"""

import importlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _patch_max_elements(monkeypatch):
    """Force the OOM threshold to 64 elements so any normal-sized array triggers
    the fallback path without requiring a genuinely huge array."""
    monkeypatch.setenv("ASTE_ANTI_ALIAS_MAX_SOURCE_ELEMENTS", "64")
    # Re-import the module inside the test so the patched env var takes effect.
    if "validation_pipeline" in sys.modules:
        del sys.modules["validation_pipeline"]
    yield
    # Cleanup: remove module so subsequent tests get a clean import
    if "validation_pipeline" in sys.modules:
        del sys.modules["validation_pipeline"]


def _make_h5_dataset(tmp_path, shape, dtype=np.float32):
    """Write a synthetic array to a temp HDF5 file and return opened file + dataset."""
    import h5py
    h5_path = str(tmp_path / "test_rho.h5")
    data = np.random.default_rng(0).random(shape).astype(dtype)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("rho", data=data)
    return h5_path


def test_anti_alias_downsample_oom_fallback_returns_ndarray(tmp_path):
    """When size > threshold the fallback must return a valid ndarray, not raise."""
    h5_path = _make_h5_dataset(tmp_path, shape=(16, 16, 4))  # 1024 elements > 64 threshold
    import h5py
    import validation_pipeline as vp
    with h5py.File(h5_path, "r") as f:
        dataset = f["rho"]
        result = vp.ArtifactLoader._anti_aliased_downsample(dataset, stride=2, label="test_rho")

    assert isinstance(result, np.ndarray), (
        f"_anti_aliased_downsample returned {type(result)}, expected np.ndarray"
    )
    assert result.size > 0, "Returned array is empty"


def test_anti_alias_downsample_oom_fallback_reduces_size(tmp_path):
    """The fallback must downsample (result is smaller than or equal to input)."""
    h5_path = _make_h5_dataset(tmp_path, shape=(32, 32))  # 1024 > 64 threshold
    import h5py
    import validation_pipeline as vp
    with h5py.File(h5_path, "r") as f:
        dataset = f["rho"]
        original_size = dataset.size
        result = vp.ArtifactLoader._anti_aliased_downsample(dataset, stride=1, label="test_rho")

    assert result.size <= original_size, (
        f"Fallback inflated array: {result.size} > {original_size}"
    )


def test_anti_alias_downsample_oom_fallback_no_exception(tmp_path):
    """No exception of any kind should be raised for an oversized dataset."""
    h5_path = _make_h5_dataset(tmp_path, shape=(8, 8, 2))  # 128 > 64 threshold
    import h5py
    import validation_pipeline as vp
    with h5py.File(h5_path, "r") as f:
        dataset = f["rho"]
        try:
            result = vp.ArtifactLoader._anti_aliased_downsample(dataset, stride=1, label="test_rho")
        except Exception as exc:
            pytest.fail(f"_anti_aliased_downsample raised {type(exc).__name__}: {exc}")


def test_anti_alias_downsample_complex_oom_fallback(tmp_path):
    """The fallback must also handle complex-valued datasets without error."""
    import h5py
    import validation_pipeline as vp
    h5_path = str(tmp_path / "test_complex.h5")
    data = (
        np.random.default_rng(1).random((16, 16)).astype(np.float32)
        + 1j * np.random.default_rng(2).random((16, 16)).astype(np.float32)
    )
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("psi", data=data)
    with h5py.File(h5_path, "r") as f:
        dataset = f["psi"]
        result = vp.ArtifactLoader._anti_aliased_downsample(dataset, stride=2, label="test_psi")

    assert np.iscomplexobj(result), "Expected complex output for complex input"
    assert result.size > 0
