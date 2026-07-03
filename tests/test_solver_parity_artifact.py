"""H10 — tests for the solver-parity artifact (tools/solver_parity_check.py).

These cover the jax-free logic (shared-IC determinism + the compare verdict), runnable on the dev box
(no jax/cupy). The actual backend runs (run_jax on WSL, run_cupy on the CuPy box) are exercised operationally,
not here. See docs/SOLVER_PARITY_ARTIFACT.md.
"""
import os, sys, io, contextlib, argparse
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.solver_parity_check as p


def test_make_ic_deterministic(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    p.make_ic(N=16, seed=42, out=str(a))
    p.make_ic(N=16, seed=42, out=str(b))
    pa, pb = np.load(a)["psi0"], np.load(b)["psi0"]
    assert pa.shape == (16, 16, 16) and pa.dtype == np.complex128
    assert np.array_equal(pa, pb), "same seed must give a byte-identical shared IC"


def test_make_ic_seed_sensitive(tmp_path):
    p.make_ic(N=16, seed=1, out=str(tmp_path / "s1.npz"))
    p.make_ic(N=16, seed=2, out=str(tmp_path / "s2.npz"))
    assert not np.array_equal(np.load(tmp_path / "s1.npz")["psi0"], np.load(tmp_path / "s2.npz")["psi0"])


def _write_ref(path, psi):
    np.savez_compressed(path, psi_fin=psi.astype(np.complex128), backend="test", steps=1, N=psi.shape[0],
                        L=10.0, dt=0.005, params="{}")


def _compare(a, b, tol=1e-6):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        p.cmd_compare(argparse.Namespace(a=str(a), b=str(b), tol=tol))
    return out.getvalue()


def test_compare_identical_is_bit_parity(tmp_path):
    psi = (np.random.default_rng(0).standard_normal((8, 8, 8))
           + 1j * np.random.default_rng(1).standard_normal((8, 8, 8))).astype(np.complex128)
    _write_ref(tmp_path / "a.npz", psi); _write_ref(tmp_path / "b.npz", psi.copy())
    assert "BIT_PARITY" in _compare(tmp_path / "a.npz", tmp_path / "b.npz")


def test_compare_small_perturbation_within_tol(tmp_path):
    psi = np.ones((8, 8, 8), np.complex128)
    _write_ref(tmp_path / "a.npz", psi)
    _write_ref(tmp_path / "b.npz", psi + 1e-9)          # rel-L2 ~1e-9 < 1e-6
    assert "PARITY_WITHIN_TOL" in _compare(tmp_path / "a.npz", tmp_path / "b.npz")


def test_compare_large_diff_fails(tmp_path):
    psi = np.ones((8, 8, 8), np.complex128)
    _write_ref(tmp_path / "a.npz", psi)
    _write_ref(tmp_path / "b.npz", psi * 2.0)            # rel-L2 ~1.0 >> tol
    assert "PARITY_FAIL" in _compare(tmp_path / "a.npz", tmp_path / "b.npz")
