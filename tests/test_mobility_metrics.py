"""H10 — tests for the mobility metrics (drag classifier + kick velocity fit).

The mobility scripts import jax at module load, so these are gated on jax (run on WSL; skipped on the dev box).
They cover the pure-logic discriminators — the drag DRAG/NUCLEATION/NULL taxonomy and the circular-COM velocity
fit — which are what the kick/adiabatic-drag conclusions rest on.
"""
import os, sys
import numpy as np
import pytest

pytest.importorskip("jax")   # skip where the jax_scout stack can't import (dev box)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jax_scout import feb_adiabatic_drag as drag
from jax_scout import feb_kick_inertia as kick

OFFSET = 1.8


def _r(**over):
    base = dict(com0=np.array([1.0, 0, 0]), com_fin=np.array([1.0, 0, 0]),
               node_com0=np.array([1.0, 0, 0]), node_com_fin=np.array([1.0, 0, 0]),
               origin_mass0=0.20, origin_mass_fin=0.20, well_mass0=0.05, well_mass_fin=0.05,
               n_start=4, n_end=4)
    base.update(over)
    return base


def test_classify_null():
    assert drag.classify_offset(_r(), OFFSET)["cell_label"] == "NULL"


def test_classify_weak_local_accretion():
    # well grows a little (>0.02), structure does not move
    assert drag.classify_offset(_r(well_mass_fin=0.10), OFFSET)["cell_label"] == "WEAK_LOCAL_ACCRETION"


def test_classify_relocation_bias():
    # COM biases toward well AND origin depletes, node count held -> genuine relocation
    r = _r(com_fin=np.array([1.5, 0, 0]), node_com_fin=np.array([1.4, 0, 0]), origin_mass_fin=0.15)
    assert drag.classify_offset(r, OFFSET)["cell_label"] == "RELOCATION_BIAS"


def test_classify_new_blob_nucleation_by_nodecount():
    assert drag.classify_offset(_r(n_end=5), OFFSET)["cell_label"] == "NEW_BLOB_NUCLEATION"


def test_classify_new_blob_nucleation_by_wellmass():
    # big well-mass gain while origin does NOT deplete = new structure, not relocation
    r = _r(well_mass_fin=0.30, origin_mass_fin=0.20)
    assert drag.classify_offset(r, OFFSET)["cell_label"] == "NEW_BLOB_NUCLEATION"


def test_classify_morphology_break():
    assert drag.classify_offset(_r(n_end=3), OFFSET)["cell_label"] == "MORPHOLOGY_BREAK"


def test_velocity_recovers_constant_motion():
    # constant velocity -> circular angle rises linearly -> velocity() must recover the slope
    L = 10.0; v_true = 0.5
    t = np.arange(0, 8.0, 0.4)
    angles = (2 * np.pi * (v_true * t) / L)          # small motion, no wrap
    m, r2, pos = kick.velocity(angles, t)
    assert abs(m - v_true) < 1e-6 and r2 > 0.999


def test_velocity_zero_for_static():
    t = np.arange(0, 8.0, 0.4)
    m, r2, _ = kick.velocity(np.zeros_like(t), t)
    assert abs(m) < 1e-9
