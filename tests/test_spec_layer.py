"""Experiment specs (irer_specs), component registry (jax_scout/registry.py), executor (tools/run_spec.py).

Phase E1 of docs/research_infrastructure/IMPLEMENTATION_PLAN_2026-10.md. The key acceptance test is
EQUIVALENCE: a spec run must reproduce the harness it replaces to round-off, otherwise "experiments as
data" would silently be a different experiment.
"""
from __future__ import annotations

import copy
import glob
import json
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import irer_specs  # noqa: E402

MINI = {
    "spec_version": 1, "id": "kg-mini", "title": "tiny KG check",
    "substrate": {"name": "kg-strang", "version": 1, "params": {"c": 1.0, "m": 1.0, "a": 0.8, "s": -0.5, "f": -0.1}},
    "protocol": {"ic": {"name": "gaussian", "params": {"A": 1.0, "sigma": 1.0, "omega": 0.9}},
                 "grid": {"N": 12, "L": 10.0}, "dt": 0.02, "T": 0.4, "sample_every": 0.1},
    "observers": [{"name": "kg_invariants"}, {"name": "centroid", "params": {"slopes": ["x"]}}],
    "invariants": {"Q_rel_drift": 1e-8},
    "prediction": {"statement": "U(1) charge is conserved by the Strang flow.",
                   "quantities": [{"key": "Q_rel_drift", "expected": 0.0, "tolerance": 1e-8}]},
}


# ------------------------------------------------------------------ schema / validator (no JAX)

def test_schema_file_is_in_sync_with_the_code():
    on_disk = json.load(open(irer_specs.SCHEMA_PATH, encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(irer_specs.SCHEMA)), "run irer_specs.write_schema()"


def test_minimal_spec_is_valid():
    assert irer_specs.validate(MINI) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s.pop("prediction"), "prediction"),
    (lambda s: s.update(id="Bad ID!"), "does not match"),
    (lambda s: s["protocol"].update(dt=0), "dt"),
    (lambda s: s.update(extra=1), "unknown key"),
    (lambda s: s["prediction"]["quantities"][0].update(kind="approx"), "not in"),
    (lambda s: s["protocol"]["grid"].update(N="12"), "expected integer"),
])
def test_invalid_specs_are_reported_not_raised(mutate, needle):
    s = copy.deepcopy(MINI)
    mutate(s)
    errs = irer_specs.validate(s)
    assert any(needle in e for e in errs), errs


def test_sweep_expansion_grid_and_zip():
    s = copy.deepcopy(MINI)
    s["sweep"] = {"axes": {"substrate.params.a": [0.7, 0.8], "protocol.dt": [0.02, 0.01]}}
    pts = irer_specs.expand(s)
    assert len(pts) == 4 and all("sweep" not in p for _, p in pts)
    assert pts[-1][1]["substrate"]["params"]["a"] == 0.8 and pts[-1][1]["protocol"]["dt"] == 0.01
    s["sweep"]["mode"] = "zip"
    assert [p["protocol"]["dt"] for _, p in irer_specs.expand(s)] == [0.02, 0.01]


def test_prediction_scoring_kinds():
    pred = {"quantities": [{"key": "a", "expected": 1.0, "tolerance": 0.01, "kind": "rel"},
                           {"key": "b", "expected": 3.0, "kind": "le"},
                           {"key": "c", "expected": 0.0, "tolerance": 1e-3}]}
    ok = irer_specs.check_prediction(pred, {"a": 1.005, "b": 2.0, "c": 5e-4})
    assert ok["status"] == "PREDICTION_MET"
    bad = irer_specs.check_prediction(pred, {"a": 1.05, "b": 2.0})
    assert bad["status"] == "PREDICTION_MISSED" and not bad["quantities"][2]["ok"]


def test_requires_is_a_warning_not_a_gate(tmp_path):
    import run_spec
    s = dict(MINI, requires=[{"spec_id": "never-run", "verdict": "PASS"}])
    assert run_spec.requires_warnings(s, str(tmp_path))
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "verdict.json").write_text(json.dumps({"spec_id": "never-run", "verdict": "PASS"}))
    assert run_spec.requires_warnings(s, str(tmp_path)) == []


def test_every_spec_in_the_repo_is_schema_valid():
    files = glob.glob(os.path.join(ROOT, "specs", "*", "*.json"))
    assert files
    for f in files:
        spec = irer_specs.load(f)
        assert irer_specs.validate(spec) == [], f
        assert os.path.basename(f)[:-5] == spec["id"], "file name must equal the spec id: %s" % f


# ------------------------------------------------------------------ registry + executor (JAX)

import importlib.util  # noqa: E402

needs_jax = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="registry/executor need JAX")


@needs_jax
def test_registry_rejects_unknown_components_and_version_drift():
    from jax_scout import registry
    s = copy.deepcopy(MINI)
    s["substrate"]["name"] = "no-such-stepper"
    s["observers"][0]["version"] = 99
    errs = registry.check_spec(s)
    assert any("unknown component" in e for e in errs) and any("pins 99" in e for e in errs)
    assert registry.check_spec(MINI) == []


@needs_jax
def test_describe_lists_every_component():
    from jax_scout import registry
    d = registry.describe()
    assert {"etdrk4-sncgl", "kg-strang", "tg-rk4"} <= {r["name"] for r in d["substrates"]}
    assert {"centroid", "mass", "kg_invariants", "energy_ratio"} <= {r["name"] for r in d["observers"]}


@needs_jax
def test_end_to_end_run_writes_the_contracted_files(tmp_path):
    import run_spec
    from jax_scout.snapshots import read_telemetry
    p = tmp_path / "kg-mini.json"
    p.write_text(json.dumps(MINI))
    rc = run_spec.main([str(p), "--out", str(tmp_path / "run"), "--sweep-root", str(tmp_path)])
    assert rc == 0
    run = tmp_path / "run"
    for f in ("spec.json", "summary.json", "verdict.json", "RUN_COMPLETE.json", "telemetry.jsonl"):
        assert (run / f).exists(), f
    summ = json.load(open(run / "summary.json"))
    assert summ["prediction_check"]["status"] == "PREDICTION_MET"
    assert "steppers" in summ["provenance"] and "KG-strang" in summ["provenance"]["steppers"]
    assert "component_hashes" in summ["provenance"]
    assert json.load(open(run / "verdict.json"))["verdict"] == "PENDING_REVIEW"
    meta, rows = read_telemetry(str(run))
    assert meta["invariants"] == {"Q_rel_drift": 1e-8} and len(rows) == 5


@needs_jax
def test_spec_run_reproduces_the_astar_harness_to_round_off(tmp_path):
    """Equivalence: the a* probe as a spec vs core_saturation_search.run_probe, same everything."""
    from jax_scout import core_saturation_search as css, registry
    import numpy as np
    N, T_steps = 16, 40
    params = dict(css.FEB)
    params["param_a"] = float(css.FEB["param_a"]) * 1.15
    ref = css.run_probe(params, N, T_steps, 6, seed=20260619, ic_norm=css.IC_NORM_PER_BLOB_FIXED)
    spec = irer_specs.load(os.path.join(ROOT, "specs", "approved", "astar-probe-pilot.json"))
    spec["substrate"]["params"] = params
    spec["protocol"]["grid"]["N"] = N
    spec["protocol"]["T"] = T_steps * spec["protocol"]["dt"]
    sim, obs = registry.build(spec)
    energy = []
    for _ in range(T_steps):          # run_probe records sum|psi|^2 after EVERY step
        sim.advance(1)
        energy.append(float(np.sum(np.abs(sim.fields()["psi"]) ** 2)))
    er = np.asarray(energy) / ref["ic_e"]
    assert np.max(np.abs(er - np.asarray(ref["er"]))) < 1e-10


@needs_jax
def test_record_window_writes_only_the_window(tmp_path):
    """protocol.record: frames every 0.1 inside [0.1, 0.3] only -> 3 frames, each with slices + a volume."""
    import run_spec
    import numpy as np
    s = copy.deepcopy(MINI)
    s["protocol"]["record"] = {"every": 0.1, "window": [0.1, 0.3], "volume": 8, "fields": ["psi"]}
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(s))
    assert run_spec.main([str(p), "--out", str(tmp_path / "run"), "--sweep-root", str(tmp_path)]) == 0
    frames = sorted((tmp_path / "run" / "history").glob("snap_*.npz"))
    ts = [float(np.load(f)["t"]) for f in frames]
    assert np.allclose(ts, [0.1, 0.2, 0.3])
    z = np.load(frames[0])
    v = z["psi__vol"]
    assert v.ndim == 3 and len(set(v.shape)) == 1 and v.shape[0] <= 8 and "pi__xy" not in z.files
    summ = json.load(open(tmp_path / "run" / "summary.json"))
    assert summ["history"]["frames_written"] == 3 and summ["prediction_check"]["status"] == "PREDICTION_MET"


@needs_jax
def test_batched_sweep_matches_point_by_point(tmp_path):
    """vmap batching is a speed-up, not a different experiment: a param_a x seed sweep run batched must
    reproduce the same sweep run one point at a time, observer by observer."""
    import run_spec
    spec = irer_specs.load(os.path.join(ROOT, "specs", "approved", "astar-probe-pilot.json"))
    spec["id"] = "batch-equivalence"
    spec["protocol"].update(grid={"N": 16, "L": 10.0}, T=0.2, sample_every=0.05)
    spec["sweep"] = {"axes": {"substrate.params.param_a": [0.50, 0.5522, 0.60],
                              "protocol.ic.params.seed": [20260619, 20260620]}}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(spec))
    assert run_spec.main([str(p), "--out", str(tmp_path / "b"), "--sweep-root", str(tmp_path)]) == 0
    assert run_spec.main([str(p), "--out", str(tmp_path / "s"), "--sweep-root", str(tmp_path), "--no-batch"]) == 0
    rb = json.load(open(tmp_path / "b" / "summary.json"))["rows"]
    rs = json.load(open(tmp_path / "s" / "summary.json"))["rows"]
    assert len(rb) == len(rs) == 6
    for a, b in zip(sorted(rb, key=lambda r: r["point"]), sorted(rs, key=lambda r: r["point"])):
        assert a["point"] == b["point"]
        for k in b["final"]:
            assert abs(a["final"][k] - b["final"][k]) <= 1e-10 * max(1.0, abs(b["final"][k])), (a["point"], k)
    one = json.load(open(tmp_path / "b" / rb[0]["dir"] / "summary.json"))
    assert one["batch"]["size"] == 6                              # really ran as one vmapped batch


@needs_jax
def test_batch_planning_separates_incompatible_points():
    import run_spec
    base = irer_specs.load(os.path.join(ROOT, "specs", "approved", "astar-probe-pilot.json"))
    pts = []
    for i, (a, dt, mode) in enumerate([(0.5, 0.005, "dissipative"), (0.6, 0.005, "dissipative"),
                                       (0.5, 0.0025, "dissipative"), (0.5, 0.005, "conservative")]):
        s = copy.deepcopy(base)
        s["substrate"]["params"].update(param_a=a, kinetic_mode=mode)
        s["protocol"]["dt"] = dt
        pts.append(("p%d" % i, s))
    pts.append(("kg", copy.deepcopy(MINI)))                       # different substrate: never batched
    sizes = sorted(len(b) for b in run_spec.plan_batches(pts))
    assert sizes == [1, 1, 1, 2]                                  # only the two a-values share a batch
    assert [len(b) for b in run_spec.plan_batches(pts[:2], batch_size=1)] == [1, 1]


@needs_jax
def test_fp32_is_offered_only_where_registered():
    from jax_scout import registry
    s = copy.deepcopy(MINI)                                       # kg-strang: fp64 only
    s["protocol"]["precision"] = "fp32"
    assert any("does not support fp32" in e for e in registry.check_spec(s))
    with pytest.raises(ValueError):
        registry.build(s)


@needs_jax
def test_fp32_screening_runs_close_to_fp64_and_never_shares_a_batch(tmp_path):
    """protocol.precision=fp32 really steps in complex64, agrees with fp64 to screening accuracy over a short
    run, is stamped in the summary, and is never vmapped together with fp64 points."""
    import run_spec
    from jax_scout import registry
    import jax.numpy as jnp
    spec = irer_specs.load(os.path.join(ROOT, "specs", "approved", "astar-probe-pilot.json"))
    spec["id"] = "precision-screen"
    spec["protocol"].update(grid={"N": 16, "L": 10.0}, T=0.2, sample_every=0.05)
    spec["sweep"] = {"axes": {"protocol.precision": ["fp64", "fp32"],
                              "substrate.params.param_a": [0.50, 0.60]}}
    assert irer_specs.validate(spec) == [] and registry.check_spec(spec) == []
    pts = irer_specs.expand(spec)
    assert sorted(len(b) for b in run_spec.plan_batches(pts)) == [2, 2]
    sim, _ = registry.build(dict(pts[-1][1], protocol=dict(pts[-1][1]["protocol"], precision="fp32")))
    assert sim.psi_k.dtype == jnp.complex64
    p = tmp_path / "s.json"
    p.write_text(json.dumps(spec))
    assert run_spec.main([str(p), "--out", str(tmp_path / "o"), "--sweep-root", str(tmp_path)]) == 0
    finals = {}
    for r in json.load(open(tmp_path / "o" / "summary.json"))["rows"]:
        d = tmp_path / "o" / r["dir"]
        s, sp = json.load(open(d / "summary.json")), json.load(open(d / "spec.json"))
        finals[(s["precision"], sp["substrate"]["params"]["param_a"])] = s["final"]
    assert len(finals) == 4
    for a in (0.50, 0.60):
        f64, f32 = finals[("fp64", a)], finals[("fp32", a)]
        for k, v in f64.items():
            if isinstance(v, float) and math.isfinite(v) and abs(v) > 1e-6:
                assert abs(f32[k] - v) <= 1e-3 * abs(v), k
