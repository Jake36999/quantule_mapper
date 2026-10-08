"""tools/qd_explore.py: parameter box, regime labels, boundary sampler, Chao1, and a real resumable run."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import qd_explore as qd  # noqa: E402

has = lambda m: importlib.util.find_spec(m) is not None   # noqa: E731
needs_sklearn = pytest.mark.skipif(not has("sklearn"), reason="needs sklearn")
needs_all = pytest.mark.skipif(not (has("jax") and has("ribs") and has("sklearn")),
                               reason="a real run needs jax + ribs + sklearn (the WSL env)")


def test_box_maps_lin_log_and_int_both_ways():
    box = qd.Box({"a": {"range": [-1.0, 1.0]}, "D": {"range": [0.1, 10.0], "scale": "log"},
                  "K": {"range": [1, 10], "type": "int"}})
    p = box.to_params([0.5, 0.5, 0.0])
    assert p["a"] == 0.0 and abs(p["D"] - 1.0) < 1e-12 and p["K"] == 1
    assert box.to_params([1.0, 1.0, 1.0])["K"] == 10
    assert {box.to_params([0, 0, u])["K"] for u in np.linspace(0, 1, 101)} == set(range(1, 11))
    for x in ([0.2, 0.7, 0.55], [0.9, 0.05, 0.95]):
        assert box.to_params(box.to_unit(box.to_params(x))) == pytest.approx(box.to_params(x))


def test_regimes_from_energy_series():
    t = np.linspace(0, 72, 101)
    lab = lambda er, contrast=50.0, status="ok": qd.classify(status, er[-1], contrast, *qd._late_stats(t, er))  # noqa: E731
    assert lab(np.ones_like(t)) == "localised/standing"
    assert lab(np.ones_like(t), contrast=3.0) == "dispersed/standing"
    assert lab(np.exp(0.05 * t)) == "localised/growing"
    assert lab(np.exp(-0.05 * t)) == "localised/decaying"
    assert lab(np.exp(0.1 * np.sin(t))) == "localised/breathing"
    assert lab(np.full_like(t, 1e-6)) == "decayed"
    assert lab(np.ones_like(t), status="diverged") == "blowup"


def test_chao1():
    assert qd.chao1([1, 1, 2, 3]) == 6.0          # 4 seen + 2^2/(2*1)
    assert qd.chao1([5, 5, 5]) == 3.0             # nothing rare -> nothing unseen


@needs_sklearn
def test_boundary_sampler_concentrates_on_the_boundary():
    rng = np.random.default_rng(0)
    X = rng.random((300, 2))
    labels = np.where(X[:, 0] > 0.5, "A", "B")
    P = qd.propose_boundary(X, labels, 20, rng)
    assert P.shape == (20, 2)
    assert np.mean(np.abs(P[:, 0] - 0.5) < 0.15) >= 0.7


def tiny_config(tmp_path):
    cfg = {"id": "qd-test", "base_spec": "specs/approved/astar-basin-ensemble.json",
           "protocol": {"grid": {"N": 16, "L": 10.0}, "dt": 0.005, "T": 0.3, "sample_every": 0.05,
                        "precision": "fp64"},
           "box": {"substrate.params.param_a": {"range": [0.3, 0.8]},
                   "substrate.params.param_eta": {"range": [-0.05, 0.2]},
                   "protocol.ic.params.K": {"range": [2, 6], "type": "int"}},
           "anchors": [{"substrate.params.param_a": 0.55223, "substrate.params.param_eta": 0.0704,
                        "protocol.ic.params.K": 6}],
           "batch": 6, "emitters": 2, "boundary_fraction": 0.34, "seed_evals": 6, "rng_seed": 7}
    p = tmp_path / "t.qd.json"
    p.write_text(json.dumps(cfg))
    return str(p)


@needs_all
def test_real_run_resumes_exports_and_reproduces(tmp_path):
    import irer_specs
    import run_spec
    cfg = tiny_config(tmp_path)
    run_dir = str(tmp_path / "QD")
    ex = qd.run(cfg, max_evals=18, run_dir=run_dir, log=lambda *a: None)
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "evals.jsonl"))]
    assert [r["i"] for r in rows] == list(range(18))
    assert rows[0]["source"] == "anchor" and {"seed", "qd", "boundary"} <= {r["source"] for r in rows}
    assert all(r["regime"] for r in rows) and len(ex.best) >= 1
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["n_evals"] == 18 and st["state"] == "stopped"

    # resume: the archive is rebuilt from evals.jsonl and numbering continues
    ex2 = qd.run(cfg, max_evals=6, run_dir=run_dir, log=lambda *a: None)
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "evals.jsonl"))]
    assert [r["i"] for r in rows] == list(range(24)) and rows[-1]["session"] == 1
    assert len(ex2.best) >= len(ex.best)

    # a changed box must not silently mix into the same archive
    c = json.load(open(cfg))
    c["box"]["substrate.params.param_a"]["range"] = [0.0, 1.0]
    open(cfg, "w").write(json.dumps(c))
    with pytest.raises(SystemExit):
        qd.run(cfg, max_evals=1, run_dir=run_dir, log=lambda *a: None)

    # any eval -> a runnable spec that reproduces it
    ok = next(r for r in rows if r["status"] == "ok" and "d_contrast" in r["final"])
    path, n = qd.export(run_dir, [ok["i"]], out_dir=str(tmp_path / "proposed"))
    spec = irer_specs.load(path)
    assert irer_specs.validate(spec) == [] and n == 1
    assert run_spec.main([path, "--out", str(tmp_path / "rerun"), "--sweep-root", str(tmp_path)]) == 0
    fin = json.load(open(tmp_path / "rerun" / "summary.json"))["final"]
    for k in ("d_contrast", "d_n_nodes", "d_rgyr"):
        assert math.isclose(fin[k], ok["final"][k], rel_tol=1e-8, abs_tol=1e-10), k

    # elites -> one zip sweep, one point per filled cell
    path, n = qd.export(run_dir, elites=True, out_dir=str(tmp_path / "proposed"), fp64=True)
    spec = irer_specs.load(path)
    assert irer_specs.validate(spec) == [] and len(irer_specs.expand(spec)) == n == len(ex2.best)


@needs_all
def test_dropping_diverged_members_does_not_change_the_survivors(tmp_path, monkeypatch):
    """Members that diverge are removed from the vmapped batch mid-run; survivors must end exactly where
    they would have alone, and the dead keep the time they died."""
    cfg, base = qd.load_config(tiny_config(tmp_path))
    mk = lambda eta, s: qd.make_spec(base, cfg, {"substrate.params.param_eta": eta, "substrate.params.param_a": 0.55,  # noqa: E731
                                                 "protocol.ic.params.K": 4}, s, "t%d" % s)
    specs = [mk(-0.3, 1), mk(0.3, 2), mk(0.25, 3)]               # linear gain grows er; damping shrinks it
    monkeypatch.setattr(qd, "ER_BLOWUP", 0.1)                     # "diverged" = er above 1: kills only the first
    res = qd.evaluate(specs, cfg)
    assert res[0]["status"] == "diverged" and 0 < res[0]["t_death"] < 0.3
    assert res[1]["status"] == res[2]["status"] == "ok"
    monkeypatch.setattr(qd, "ER_BLOWUP", 1e4)
    alone = qd.evaluate([specs[1]], cfg)[0]
    for k in ("d_contrast", "d_rgyr", "d_mass"):
        assert math.isclose(res[1]["final"][k], alone["final"][k], rel_tol=1e-10), k
    assert res[1]["er_series"][0] == [0.0, 1.0] and len(res[1]["er_series"]) == 7


@needs_all
def test_eval_chunks_do_not_change_results(tmp_path):
    cfg, base = qd.load_config(tiny_config(tmp_path))
    specs = [qd.make_spec(base, cfg, {"substrate.params.param_eta": eta}, 3, "c%d" % j)
             for j, eta in enumerate((0.0, 0.1, 0.2))]
    whole = qd.evaluate(specs, dict(cfg, eval_chunk=None))
    split = qd.evaluate(specs, dict(cfg, eval_chunk=2))
    for a, b in zip(whole, split):
        assert a["regime"] == b["regime"]
        assert math.isclose(a["final"]["d_contrast"], b["final"]["d_contrast"], rel_tol=1e-10)
