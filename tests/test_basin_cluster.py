"""tools/basin_cluster.py (Phase F2) on synthetic ensembles with known basins."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
pytest.importorskip("sklearn")
import basin_cluster as bc  # noqa: E402


def make_run(tmp_path, points):
    """points: [(label, param_a, seed, descriptors)]"""
    for label, pa, seed, desc in points:
        d = tmp_path / label
        d.mkdir()
        (d / "spec.json").write_text(json.dumps({"substrate": {"params": {"param_a": pa}},
                                                 "protocol": {"ic": {"params": {"seed": seed}}}}))
        (d / "summary.json").write_text(json.dumps({"final": desc, "stop_reason": "completed"}))
    return str(tmp_path)


def shape(n_nodes, rg, k=1.0, contrast=50.0, mass=10.0, jitter=0.0, rng=None):
    j = (lambda: rng.normal(0, jitter)) if rng is not None else (lambda: 0.0)
    return {"d_mass": mass * (1 + j()), "d_amp": 1.0, "d_contrast": contrast * (1 + j()), "d_n_nodes": float(n_nodes),
            "d_node_size_mean": 20.0 * (1 + j()), "d_node_size_std": 2.0, "d_pair_mean": 3.0 * n_nodes / 4 * (1 + j()),
            "d_pair_std": 0.5, "d_pair_min": 1.5, "d_rgyr": rg * (1 + j()), "d_k_mean": k * (1 + j()), "d_aniso": 0.5}


def test_two_basins_at_one_point_are_separated_and_matched_across_points(tmp_path):
    rng = np.random.default_rng(0)
    pts = []
    for pa in (0.55, 0.56):
        for s in range(4):          # 4-node basin
            pts.append(("p%.2f_s%d" % (pa, s), pa, s, shape(4, 2.0, jitter=0.01, rng=rng)))
        for s in range(4, 8):       # 6-node basin, same mass (extensive) -- must still separate
            pts.append(("p%.2f_s%d" % (pa, s), pa, s, shape(6, 3.0, jitter=0.01, rng=rng)))
    out = bc.run(make_run(tmp_path, pts), ["param_a"])
    assert len(out["basins"]) == 2
    for row in out["by_point"]:
        assert sorted(row["basins"].values()) == [4, 4]
    assert (tmp_path / "BASINS.md").exists()


def test_extensive_mass_differences_do_not_split_a_basin(tmp_path):
    pts = [("s%d" % s, 0.55, s, shape(4, 2.0, mass=10.0 * (1 + s))) for s in range(5)]   # 1x..5x mass
    out = bc.run(make_run(tmp_path, pts), ["param_a"])
    assert len(out["basins"]) == 1


def test_identical_end_states_are_one_basin(tmp_path):
    pts = [("s%d" % s, 0.55, s, shape(4, 2.0)) for s in range(3)]
    out = bc.run(make_run(tmp_path, pts), ["param_a"])
    assert len(out["basins"]) == 1 and out["by_point"][0]["basins"] == {"0": 3}
