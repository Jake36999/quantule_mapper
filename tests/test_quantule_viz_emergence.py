from __future__ import annotations

import json

import numpy as np

from quantule_viz.renderers.emergence_sequence import (
    derive_omega_sq,
    load_frame_bundle,
    selected_frame_indices,
)


def test_load_frame_bundle_computes_rho_from_complex_psi(tmp_path):
    psi = np.ones((3, 4, 4, 4), dtype=np.complex64)
    psi[1] *= 1 + 2j
    times = np.array([0.0, 2.5, 5.0])
    source = tmp_path / "frames.npz"
    np.savez(source, psi=psi, times=times)

    bundle = load_frame_bundle(source)

    assert bundle.psi.shape == (3, 4, 4, 4)
    assert np.allclose(bundle.times, times)
    assert np.allclose(bundle.rho[0], 1.0)
    assert np.allclose(bundle.rho[1], 5.0)


def test_selected_frame_indices_are_stable_and_unique():
    assert selected_frame_indices(1) == [0]
    assert selected_frame_indices(6) == [0, 1, 2, 3, 4, 5]
    assert selected_frame_indices(41) == [0, 5, 10, 20, 30, 40]


def test_derive_omega_sq_uses_params_when_available(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"params": {"param_rho_vac": 2.0, "param_a_coupling": 1.5}}),
        encoding="utf-8",
    )
    rho = np.array([[[1.0, 2.0], [4.0, 8.0]]], dtype=float)

    omega = derive_omega_sq(rho, summary)

    assert omega is not None
    assert np.allclose(omega, (2.0 / rho) ** 1.5)
