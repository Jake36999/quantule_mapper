"""tools/screen_verify.py: choose fp32 screen points for fp64 re-runs, rebuild them as one spec, compare."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import basin_cluster as bc  # noqa: E402
import irer_specs  # noqa: E402
import screen_verify as sv  # noqa: E402

needs_sklearn = pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="clustering replicates needs sklearn")
BASE = irer_specs.load(os.path.join(ROOT, "specs", "approved", "astar-basin-ensemble.json"))


def shape(n_nodes, rg):
    return {"d_mass": 10.0, "d_amp": 1.0, "d_contrast": 50.0, "d_n_nodes": float(n_nodes), "d_node_size_mean": 20.0,
            "d_node_size_std": 2.0, "d_pair_mean": 3.0 * n_nodes / 4, "d_pair_std": 0.5, "d_pair_min": 1.5,
            "d_rgyr": rg, "d_k_mean": 1.0, "d_aniso": 0.5}


def make_screen(tmp_path, layout):
    """layout: {param_a: [n_nodes per seed]} -> a fake fp32 screen run directory (real specs, fake finals)."""
    root = copy.deepcopy(BASE)
    root["id"] = "screen-test"
    root["protocol"].update(precision="fp32", grid={"N": 32, "L": 10.0})
    seeds = list(range(len(next(iter(layout.values())))))
    root["sweep"] = {"axes": {"substrate.params.param_a": sorted(layout),
                              "protocol.ic.params.seed": seeds}}
    run = tmp_path / "screen"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(root))
    for label, spec in irer_specs.expand(root):
        a, s = spec["substrate"]["params"]["param_a"], spec["protocol"]["ic"]["params"]["seed"]
        n = layout[a][s]
        d = run / label
        d.mkdir()
        (d / "spec.json").write_text(json.dumps(spec))
        fin = shape(n, 2.0 if n == 4 else 3.0) if n else {}
        (d / "summary.json").write_text(json.dumps({"final": fin, "stop_reason": "completed" if n else "nonfinite"}))
    bc.run(str(run), ["param_a"])
    return str(run)


@needs_sklearn
def test_selects_boundaries_splits_outliers_and_failures(tmp_path):
    run = make_screen(tmp_path, {0.50: [4, 4, 4], 0.52: [4, 4, 4], 0.54: [4, 6, 6], 0.56: [6, 6, 6],
                                 0.58: [6, 6, 0]})
    chosen = sv.select(run, ["param_a"])
    pts = {lb: json.load(open(os.path.join(run, lb, "spec.json"))) for lb in chosen}
    a_of = {lb: s["substrate"]["params"]["param_a"] for lb, s in pts.items()}
    reasons = lambda a: {w for lb, ws in chosen.items() if a_of[lb] == a for w in ws}   # noqa: E731
    assert "boundary" in reasons(0.52) and "boundary" in reasons(0.54) and "boundary" in reasons(0.56)
    assert "ic_split" in reasons(0.54)
    assert "screen_failed" in reasons(0.58)
    assert not reasons(0.50) - {"representative"}          # deep inside a basin: at most a representative
    assert sum(1 for ws in chosen.values() if "representative" in ws) == 2   # one per basin
    assert len(chosen) < 15                                # it is a SELECTION, not a re-run of everything


@needs_sklearn
def test_structureless_screen_warns(tmp_path):
    run = make_screen(tmp_path, {0.50: [4, 6], 0.52: [6, 4], 0.54: [4, 6]})   # every point splits
    m = sv.plan(run, ["param_a"], out_dir=str(tmp_path / "proposed"))
    assert m["warnings"] and "little basin structure" in m["warnings"][0]


@needs_sklearn
def test_verify_spec_rebuilds_exactly_the_chosen_points_in_fp64(tmp_path):
    run = make_screen(tmp_path, {0.50: [4, 4], 0.52: [6, 6], 0.54: [6, 6]})
    m = sv.plan(run, ["param_a"], N=96, out_dir=str(tmp_path / "proposed"))
    spec = json.load(open(m["verify_spec"]))
    assert irer_specs.validate(spec) == []
    assert spec["protocol"]["precision"] == "fp64" and spec["protocol"]["grid"]["N"] == 96
    assert "requires" not in spec
    pts = irer_specs.expand(spec)
    assert len(pts) == len(m["points"])
    for (vl, vs), row in zip(pts, m["points"]):
        src = json.load(open(os.path.join(run, row["screen_label"], "spec.json")))
        assert row["verify_label"] == vl
        for path in ("substrate.params.param_a", "protocol.ic.params.seed", "protocol.T", "protocol.dt"):
            get = lambda s: eval("s" + "".join("[%r]" % k for k in path.split(".")))   # noqa: E731
            assert get(vs) == get(src), path


@needs_sklearn
def test_compare_reports_agreement_and_disagreement(tmp_path):
    run = make_screen(tmp_path, {0.50: [4, 4], 0.52: [6, 6]})
    m = sv.plan(run, ["param_a"], out_dir=str(tmp_path / "proposed"))
    vrun = tmp_path / "verify"
    vrun.mkdir()
    flipped = m["points"][0]["screen_label"]
    for row in m["points"]:
        src = json.load(open(os.path.join(run, row["screen_label"], "summary.json")))["final"]
        n = src["d_n_nodes"]
        if row["screen_label"] == flipped:                 # fp64 says the other basin
            n = 6.0 if n == 4.0 else 4.0
        d = vrun / row["verify_label"]
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"final": shape(int(n), 2.0 if n == 4 else 3.0),
                                                    "stop_reason": "completed"}))
    out = sv.compare(run, str(vrun))
    assert out["counts"].get("DISAGREE") == 1
    assert out["counts"].get("AGREE") == len(m["points"]) - 1
    assert (vrun / "VERIFY.md").exists()


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="end-to-end needs JAX")
def test_end_to_end_fp32_screen_then_fp64_verify_agrees(tmp_path):
    """Real tiny runs: fp32 screen -> cluster -> plan -> fp64 verify -> compare. Over a short run fp32 and
    fp64 must put every verified point in the same basin. One seed per point, so clustering needs no
    sklearn (the WSL JAX env has none)."""
    import run_spec
    spec = copy.deepcopy(BASE)
    spec["id"] = "e2e-screen"
    spec.pop("requires")
    spec["protocol"].update(grid={"N": 16, "L": 10.0}, T=0.5, sample_every=0.25, precision="fp32")
    spec["sweep"] = {"axes": {"substrate.params.param_a": [0.40, 0.50, 0.60],
                              "protocol.ic.params.seed": [20260619]}}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(spec))
    screen = tmp_path / "screen"
    assert run_spec.main([str(p), "--out", str(screen), "--sweep-root", str(tmp_path)]) == 0
    bc.run(str(screen), ["param_a"])
    m = sv.plan(str(screen), ["param_a"], out_dir=str(tmp_path / "proposed"))
    verify = tmp_path / "verify"
    assert run_spec.main([m["verify_spec"], "--out", str(verify), "--sweep-root", str(tmp_path)]) == 0
    out = sv.compare(str(screen), str(verify))
    assert out["counts"] == {"AGREE": len(m["points"])}, out["rows"]
