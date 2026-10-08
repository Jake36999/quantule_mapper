#!/usr/bin/env python
"""Quality-diversity explorer: search parameter space for DIFFERENT end states, not one best score.

    python tools/qd_explore.py run    specs/qd/wide-net.qd.json [--hours 8] [--max-evals N]
    python tools/qd_explore.py status sweep_runs/QD_<id>
    python tools/qd_explore.py pause|resume|stop sweep_runs/QD_<id>     (control files; never launches)
    python tools/qd_explore.py export sweep_runs/QD_<id> --eval 123 [--fp64]   (one eval -> runnable spec)
    python tools/qd_explore.py export sweep_runs/QD_<id> --elites [--regime localised] [--fp64]

WHY (docs/research_infrastructure/QD_EXPLORER.md). The Hunter's GA chased ONE hand-weighted score and
averaged over seeds, so its population collapsed into one basin (SEARCH_STACK_AUDIT_2026-10). A blind
grid wastes samples where nothing changes. This driver does what the basin question needs:

  * CMA-MAE (pyribs) fills an ARCHIVE OF BEHAVIOURS -- a grid over end-state measures (localisation,
    node count, energy ratio). Its emitters are rewarded for reaching cells nobody has reached, so it
    refines where behaviour changes, and the objective (stationarity: a standing end state scores 0)
    only ranks states WITHIN a cell. Diversity cannot collapse onto one optimum.
  * A boundary sampler (active learning) fits a classifier params -> coarse regime on every evaluation
    so far and spends part of each batch where the classifier is least certain: the boundaries between
    regimes, where new basins are born.
  * The IC seed is drawn fresh for every evaluation, so multistability (two basins at one parameter
    point) shows up as one point landing in two cells instead of being averaged away.

BACKGROUND-SAFE. Every evaluation is one line in <run>/evals.jsonl (fsync'd), and that file is the only
state: a restart rebuilds the archive from it (CMA state restarts; it re-adapts in a few generations), so
a crash, reboot or `stop` loses at most the generation in flight. `pause` / `stop` are files the loop
checks between generations. Each row carries the exact parameters + IC seed, so `export` turns any eval
into a runnable spec (tools/run_spec.py; fp64 for verification).

SCOPE. etdrk4-sncgl only (the batched vmapped path). Screening numbers: fp32 by default, short T, a
modest grid. A cell is a HINT of a behaviour, to be confirmed in fp64 (export --fp64 -> run_spec ->
tools/screen_verify.py-style comparison) before it means anything. Descriptive, never a verdict.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time

import numpy as np

# Before anything can import JAX (provenance.stamp lists devices, which starts the backend): a background
# job sharing the PC must allocate GPU memory as it needs it, not grab 75% of the card up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import irer_specs  # noqa: E402

OBJ_FAIL = -10.0          # objective given to blow-ups; below THRESHOLD_MIN so they never enter the archive
THRESHOLD_MIN = -9.5
CONTRAST_MIN = 20.0       # state_descriptors v2: below this a field is not localised
STANDING_TOL = 0.05       # |change of log er over the late half| below this = standing
BREATHING_TOL = 0.02      # detrended late std of log er above this = breathing
ER_BLOWUP, ER_DECAYED = 1e4, 1e-4

DEFAULT_MEASURES = [  # name, lo, hi, bins
    ["log_contrast", 0.0, 4.0, 16],
    ["n_nodes", 0.0, 24.0, 24],
    ["log_er", -4.0, 2.0, 12],
]


# ============================================================================ config + parameter box

def load_config(path):
    cfg = json.load(open(path, encoding="utf-8"))
    cfg.setdefault("batch", 20)
    cfg.setdefault("emitters", 3)
    cfg.setdefault("boundary_fraction", 0.25)
    cfg.setdefault("seed_evals", 100)
    cfg.setdefault("sigma0", 0.15)
    cfg.setdefault("measures", DEFAULT_MEASURES)
    cfg.setdefault("learning_rate", 0.01)
    cfg.setdefault("rng_seed", 0)
    cfg.setdefault("anchors", [])
    cfg.setdefault("eval_chunk", None)
    base = irer_specs.load(os.path.join(ROOT, cfg["base_spec"]) if not os.path.isabs(cfg["base_spec"])
                           else cfg["base_spec"])
    if base["substrate"]["name"] != "etdrk4-sncgl":
        raise SystemExit("qd_explore supports etdrk4-sncgl only (got %s)" % base["substrate"]["name"])
    names = {o["name"] for o in base["observers"]}
    if "state_descriptors" not in names:
        raise SystemExit("base spec needs the state_descriptors observer")
    return cfg, base


class Box:
    """Normalised [0,1]^d <-> parameter values. scale: lin | log; type: float | int."""

    def __init__(self, box):
        self.paths = list(box)
        self.spec = [box[p] for p in self.paths]
        for p, b in zip(self.paths, self.spec):
            lo, hi = b["range"]
            if not hi > lo:
                raise SystemExit("box %s: empty range" % p)
            if b.get("scale") == "log" and lo <= 0:
                raise SystemExit("box %s: log scale needs a positive range" % p)

    @property
    def dim(self):
        return len(self.paths)

    def to_params(self, x):
        out = {}
        for p, b, u in zip(self.paths, self.spec, np.clip(np.asarray(x, float), 0.0, 1.0)):
            lo, hi = b["range"]
            if b.get("type") == "int":
                v = int(min(hi, max(lo, math.floor(lo + u * (hi - lo + 1)))))
            elif b.get("scale") == "log":
                v = float(lo * (hi / lo) ** u)
            else:
                v = float(lo + u * (hi - lo))
            out[p] = v
        return out

    def to_unit(self, params):
        x = []
        for p, b in zip(self.paths, self.spec):
            lo, hi = b["range"]
            v = float(params[p])
            if b.get("type") == "int":
                u = (v - lo + 0.5) / (hi - lo + 1)
            elif b.get("scale") == "log":
                u = math.log(v / lo) / math.log(hi / lo)
            else:
                u = (v - lo) / (hi - lo)
            x.append(min(1.0, max(0.0, u)))
        return np.array(x)


def make_spec(base, cfg, params, ic_seed, label):
    s = copy.deepcopy(base)
    for k in ("sweep", "requires"):
        s.pop(k, None)
    s["id"] = "%s-%s" % (cfg["id"], label)
    s["protocol"].update(copy.deepcopy(cfg.get("protocol", {})))
    s["protocol"].pop("record", None)
    s["protocol"]["ic"].setdefault("params", {})["seed"] = int(ic_seed)
    for path, v in params.items():
        irer_specs._set(s, path, v)
    return s


# ============================================================================ evaluation

def _late_stats(ts, ers):
    """log-er change over the late half (stationarity) and its detrended std (breathing)."""
    t, y = np.asarray(ts, float), np.log(np.maximum(np.asarray(ers, float), 1e-300))
    m = t >= t[-1] / 2
    if m.sum() < 3:
        return float("nan"), float("nan")
    A = np.vstack([t[m], np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    resid = y[m] - A @ coef
    return float(coef[0] * (t[m][-1] - t[m][0])), float(np.std(resid))


def classify(status, er_final, contrast, late_change, late_std):
    if status != "ok" or not np.isfinite(er_final) or er_final > ER_BLOWUP:
        return "blowup"
    if er_final < ER_DECAYED:
        return "decayed"
    shape = "localised" if contrast >= CONTRAST_MIN else "dispersed"
    if not np.isfinite(late_change):
        return shape + "/unknown"
    if abs(late_change) < STANDING_TOL:
        trend = "breathing" if late_std > BREATHING_TOL else "standing"
    else:
        trend = "growing" if late_change > 0 else "decaying"
    return shape + "/" + trend


def measures_of(cfg, final, er_final):
    vals = {"log_contrast": math.log10(max(final.get("d_contrast", 1.0), 1.0)),
            "n_nodes": float(final.get("d_n_nodes", 0.0)),
            "log_er": math.log10(max(er_final, 1e-300)),
            "log_rgyr": math.log10(max(final.get("d_rgyr", 1e-6), 1e-6)),
            "k_mean": float(final.get("d_k_mean", 0.0)),
            "aniso": float(final.get("d_aniso", 0.0))}
    return [float(np.clip(vals[name], lo, hi - 1e-9)) for name, lo, hi, _ in cfg["measures"]]


def evaluate(specs, cfg, keep_fields=False):
    """Run concrete specs together (vmapped where batch keys agree). -> list of result dicts."""
    import jax.numpy as jnp
    from jax_scout import registry
    pr = specs[0]["protocol"]
    dt, T = float(pr["dt"]), float(pr["T"])
    n_total = int(round(T / dt))
    every = max(1, int(round(float(pr.get("sample_every", T / 100.0)) / dt)))
    groups = {}
    for i, s in enumerate(specs):
        groups.setdefault(registry.batch_key(s), []).append(i)
    # eval_chunk: members stepped together. At N>=48 vmapping is compute-bound (no speed-up), so smaller
    # chunks cost nothing in throughput and bound GPU memory (N=96: ~4.2 GB for 10 members, ~2 GB for 5).
    chunk = int(cfg.get("eval_chunk") or len(specs))
    chunks = [idx[a:a + chunk] for idx in groups.values() for a in range(0, len(idx), chunk)]
    results = [None] * len(specs)
    for idx in chunks:
        t0 = time.time()
        built = [registry.build(specs[i]) for i in idx]
        sims = [b[0] for b in built]
        energy = lambda b: np.asarray(jnp.sum(jnp.abs(b.psi_k) ** 2, axis=(1, 2, 3)), dtype=float)  # noqa: E731
        batch = registry.BatchedETDRK4(sims, [specs[i] for i in idx])
        e0 = energy(batch)
        n_samples = -(-n_total // every)
        E = np.full((n_samples + 1, len(idx)), np.nan)
        E[0] = 1.0
        ts = [0.0]
        alive = np.isfinite(e0)
        t_death = [None if a else 0.0 for a in alive]
        active = [p for p in range(len(idx)) if alive[p]]
        done = k = 0
        while done < n_total and active:
            n = min(every, n_total - done)
            batch.advance(n)
            done += n
            k += 1
            ts.append(done * dt)
            e = energy(batch) / e0[active]
            E[k, active] = e
            died = False
            for p, v in zip(active, e):
                if not (np.isfinite(v) and v < ER_BLOWUP * 10):
                    alive[p], t_death[p], died = False, done * dt, True
            if died:
                # drop the dead from the vmapped batch: a diverged member would otherwise cost a full
                # member's GPU time to the end of the run (45% of a wide-box seed batch diverged). Each new
                # batch size compiles once per process (registry caches the function; jit caches shapes).
                active = [p for p in active if alive[p]]
                if active:
                    batch = registry.BatchedETDRK4([sims[p] for p in active], [specs[idx[p]] for p in active])
        E = E[:k + 1]
        ts = np.array(ts)
        wall = (time.time() - t0) / len(idx)
        for j, i in enumerate(idx):
            er_final = float(E[-1, j])
            status = "ok" if alive[j] else "diverged"
            final, ok_obs = {}, status == "ok"
            if ok_obs:
                for name, fn, params in built[j][1]:
                    if name == "energy_ratio":
                        continue
                    try:
                        final.update(fn(sims[j], **params))
                    except Exception as exc:       # an observer failure is data, not a crash
                        final["observer_error"] = str(exc)[:200]
            series = E[:, j]
            k_ok = np.isfinite(series)
            late_change, late_std = _late_stats(ts[k_ok], series[k_ok]) if (k_ok.sum() >= 3 and ok_obs) else (
                float("nan"), float("nan"))
            # compact er(t) for later re-analysis (re-labelling with other thresholds, transients, ...)
            keep = np.unique(np.linspace(0, len(ts) - 1, min(len(ts), 60)).round().astype(int))
            er_series = [[round(float(ts[q]), 4), (float("%.5g" % series[q]) if np.isfinite(series[q]) else None)]
                         for q in keep]
            regime = classify(status, er_final, final.get("d_contrast", 0.0), late_change, late_std)
            obj = OBJ_FAIL if regime == "blowup" else float(-min(abs(late_change), 9.0)) \
                if np.isfinite(late_change) else OBJ_FAIL
            res = {"status": status, "regime": regime, "objective": obj, "er_final": er_final,
                   "late_log_er_change": late_change, "late_log_er_std": late_std,
                   "final": {k: v for k, v in final.items() if isinstance(v, (int, float, str))},
                   "t_end": float(ts[-1]) if ok_obs else t_death[j], "t_death": t_death[j],
                   "er_series": er_series, "wall_s": round(wall, 2)}
            res["measures"] = measures_of(cfg, final, er_final) if regime != "blowup" else None
            if keep_fields and ok_obs:
                res["_rho"] = _downsample_rho(sims[j])
            results[i] = res
    return results


def _downsample_rho(sim, target=24):
    psi = sim.fields()["psi"]
    s = max(1, int(math.ceil(psi.shape[0] / target)))
    return (np.abs(psi[::s, ::s, ::s]) ** 2).astype(np.float16)


# ============================================================================ boundary sampler

def entropy(P):
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log(P)).sum(1) / math.log(max(P.shape[1], 2))


def propose_boundary(X, labels, k, rng, pool=4000):
    """Active learning on regime boundaries: fit params -> regime, then choose the k pool points where the
    classifier is least certain, plus a distance bonus so it also visits empty regions; greedily spread."""
    X = np.asarray(X, float)
    d = X.shape[1] if X.ndim == 2 and len(X) else None
    if d is None or len(X) < 30 or len(set(labels)) < 2:
        return rng.random((k, d if d else 1)) if d else None
    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, random_state=int(rng.integers(1 << 30)))
    clf.fit(X, labels)
    C = rng.random((pool, d))
    H = entropy(clf.predict_proba(C))
    dmin = np.full(pool, np.inf)
    for a in range(0, len(X), 2048):
        D = ((C[:, None, :] - X[None, a:a + 2048, :]) ** 2).sum(-1)
        dmin = np.minimum(dmin, D.min(1))
    dmin = np.sqrt(dmin)
    score = H + 0.5 * dmin / (np.median(dmin) + 1e-12) * (0.2 if H.max() > 0.3 else 1.0)
    chosen, sep = [], 0.05 * math.sqrt(d)
    for i in np.argsort(-score):
        if all(np.linalg.norm(C[i] - C[j]) > sep for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    while len(chosen) < k:
        chosen.append(int(rng.integers(pool)))
    return C[chosen]


def chao1(hits):
    """Estimated number of reachable cells from per-cell hit counts (ecology's unseen-species estimator)."""
    c = np.asarray([h for h in hits if h > 0])
    s, f1, f2 = len(c), int((c == 1).sum()), int((c == 2).sum())
    return float(s + (f1 * f1 / (2.0 * f2) if f2 > 0 else f1 * (f1 - 1) / 2.0))


# ============================================================================ driver

class Explorer:
    def __init__(self, cfg, base, run_dir):
        from ribs.archives import GridArchive
        self.cfg, self.base, self.dir = cfg, base, run_dir
        self.box = Box(cfg["box"])
        self.rng = np.random.default_rng(cfg["rng_seed"])
        dims = [int(m[3]) for m in cfg["measures"]]
        ranges = [(float(m[1]), float(m[2])) for m in cfg["measures"]]
        kw = dict(solution_dim=self.box.dim, dims=dims, ranges=ranges)
        self.archive = GridArchive(**kw, learning_rate=cfg["learning_rate"], threshold_min=THRESHOLD_MIN,
                                   seed=cfg["rng_seed"])
        self.result = GridArchive(**kw, seed=cfg["rng_seed"])
        self.rows = []
        self.hits = {}
        self.best = {}
        self.scheduler = None
        os.makedirs(os.path.join(run_dir, "elites"), exist_ok=True)

    # ---- persistence
    @property
    def evals_path(self):
        return os.path.join(self.dir, "evals.jsonl")

    def load(self):
        if not os.path.exists(self.evals_path):
            return
        with open(self.evals_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:                 # torn last line after a crash: skip
                    continue
                self._ingest(row, add=True)

    def _ingest(self, row, add):
        self.rows.append(row)
        if row.get("measures") is None:
            return
        cell = int(self.result.index_of_single(row["measures"]))
        row["cell"] = cell
        self.hits[cell] = self.hits.get(cell, 0) + 1
        if row["objective"] > THRESHOLD_MIN and row["objective"] > self.best.get(cell, -np.inf):
            self.best[cell] = row["objective"]
        x, o, m = np.array([row["x"]]), np.array([row["objective"]]), np.array([row["measures"]])
        if add:                                    # qd rows were already added by scheduler.tell
            self.archive.add(x, o, m)
        if row["objective"] > THRESHOLD_MIN:
            self.result.add(x, o, m)

    def _emitters(self):
        from ribs.emitters import EvolutionStrategyEmitter
        from ribs.schedulers import Scheduler
        n_b = int(round(self.cfg["batch"] * self.cfg["boundary_fraction"]))
        n_e = max(1, self.cfg["emitters"])
        per = max(1, (self.cfg["batch"] - n_b) // n_e)
        x0s = [self.box.to_unit(a) for a in self.cfg["anchors"]][:n_e]
        while len(x0s) < n_e:
            x0s.append(self.rng.random(self.box.dim))
        ems = [EvolutionStrategyEmitter(self.archive, x0=x0, sigma0=self.cfg["sigma0"], ranker="imp",
                                        bounds=[(0.0, 1.0)] * self.box.dim, batch_size=per,
                                        seed=int(self.rng.integers(1 << 30))) for x0 in x0s]
        # no result_archive here: the scheduler would insert blow-ups into it at their placeholder
        # measures. self.result is fed by _ingest, valid rows only.
        self.scheduler = Scheduler(self.archive, ems)
        return n_b

    # ---- one generation
    def generation(self, gen, session):
        n_done = len(self.rows)
        if n_done < self.cfg["seed_evals"]:
            import warnings
            from scipy.stats import qmc
            sob = qmc.Sobol(self.box.dim, scramble=True, seed=self.cfg["rng_seed"])
            k = min(self.cfg["batch"], self.cfg["seed_evals"] - n_done)
            with warnings.catch_warnings():            # Sobol balance warning for non-power-of-2 draws
                warnings.simplefilter("ignore")
                if n_done:
                    sob.fast_forward(n_done)
                X = list(sob.random(k))
            src = ["seed"] * k
            if n_done == 0:                        # known points first: the anchors must reappear
                for j, a in enumerate(self.cfg["anchors"][:k]):
                    X[j], src[j] = self.box.to_unit(a), "anchor"
            X_qd = None
        else:
            if self.scheduler is None:
                self.n_b = self._emitters()
            X_qd = self.scheduler.ask()
            X_b = propose_boundary([r["x"] for r in self.rows], [r["regime"] for r in self.rows],
                                   self.n_b, self.rng) if self.n_b else np.zeros((0, self.box.dim))
            X = list(X_qd) + list(X_b)
            src = ["qd"] * len(X_qd) + ["boundary"] * len(X_b)
        seeds = self.rng.integers(1, 2 ** 31 - 1, size=len(X))
        params = [self.box.to_params(x) for x in X]
        labels = ["e%07d" % (n_done + j) for j in range(len(X))]
        specs = [make_spec(self.base, self.cfg, p, s, lb) for p, s, lb in zip(params, seeds, labels)]
        res = evaluate(specs, self.cfg, keep_fields=True)

        if X_qd is not None:                       # emitters must hear about every solution they asked for
            nq = len(X_qd)
            obj = np.array([r["objective"] for r in res[:nq]])
            meas = np.array([r["measures"] if r["measures"] is not None else
                             [m[1] for m in self.cfg["measures"]] for r in res[:nq]])
            self.scheduler.tell(obj, meas)
        rows = []
        with open(self.evals_path, "a", encoding="utf-8") as fh:
            for j, (x, r) in enumerate(zip(X, res)):
                rho = r.pop("_rho", None)
                row = {"i": n_done + j, "gen": gen, "session": session, "source": src[j],
                       "x": [round(float(v), 6) for v in x], "params": params[j], "ic_seed": int(seeds[j]), **r}
                prev_best = None
                if row["measures"] is not None:
                    cell = int(self.result.index_of_single(row["measures"]))
                    prev_best = self.best.get(cell)
                    row["new_cell"] = cell not in self.hits
                # direct inserts for seed/boundary/anchor points (the scheduler already added its own)
                self._ingest(row, add=src[j] != "qd")
                if rho is not None and row.get("cell") is not None and row["objective"] > THRESHOLD_MIN and \
                        (prev_best is None or row["objective"] > prev_best):
                    np.savez_compressed(os.path.join(self.dir, "elites", "cell_%06d.npz" % row["cell"]),
                                        rho=rho, i=row["i"])
                fh.write(json.dumps(row) + "\n")
                rows.append(row)
            fh.flush()
            os.fsync(fh.fileno())
        return rows

    def status(self, started=None, n_session=0):
        regimes = {}
        for r in self.rows:
            regimes[r["regime"]] = regimes.get(r["regime"], 0) + 1
        recent = self.rows[-500:]
        st = {"n_evals": len(self.rows), "cells_filled": len(self.best),
              "cells_total": int(np.prod([m[3] for m in self.cfg["measures"]])),
              "cells_estimated_chao1": round(chao1(self.hits.values()), 1),
              "new_cells_last_500": sum(1 for r in recent if r.get("new_cell")),
              "regimes": dict(sorted(regimes.items(), key=lambda kv: -kv[1])),
              "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        if started and n_session:
            st["evals_per_hour_this_session"] = round(n_session / ((time.time() - started) / 3600.0), 1)
        return st


def _write_status(run_dir, st):
    tmp = os.path.join(run_dir, "status.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=2)
    os.replace(tmp, os.path.join(run_dir, "status.json"))


def run(config_path, hours=None, max_evals=None, run_dir=None, log=print):
    from jax_scout.provenance import stamp, write_json
    cfg, base = load_config(config_path)
    run_dir = run_dir or os.path.join(ROOT, "sweep_runs", "QD_" + cfg["id"].upper().replace("-", "_"))
    os.makedirs(run_dir, exist_ok=True)
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path, encoding="utf-8"))
        same = {k: v for k, v in old.items() if k != "provenance"} == json.loads(json.dumps(cfg))
        if not same:
            raise SystemExit("%s holds a different config; use a new id (or --run-dir) to change the box, "
                             "measures or protocol" % run_dir)
    else:
        write_json(cfg_path, cfg)
        write_json(os.path.join(run_dir, "spec.json"), make_spec(base, cfg, {}, 0, "base"), stamp_metadata=False)
    ex = Explorer(cfg, base, run_dir)
    ex.load()
    sessions = os.path.join(run_dir, "sessions.jsonl")
    session = sum(1 for _ in open(sessions, encoding="utf-8")) if os.path.exists(sessions) else 0
    write_json(os.path.join(run_dir, "session_%03d.json" % session),
               {"session": session, "resumed_at_eval": len(ex.rows), "hours": hours, "max_evals": max_evals,
                "provenance": stamp()})
    with open(sessions, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"session": session, "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "resumed_at_eval": len(ex.rows)}) + "\n")
    for f in ("STOP",):
        if os.path.exists(os.path.join(run_dir, f)):
            os.remove(os.path.join(run_dir, f))
    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: stop.update(flag=True))
        except ValueError:                         # not the main thread (tests)
            pass
    started, n0 = time.time(), len(ex.rows)
    gen = (max(r["gen"] for r in ex.rows) + 1) if ex.rows else 0
    log("QD %s: %d evals on disk, %d cells; box %d-D; batch %d" % (cfg["id"], len(ex.rows), len(ex.best),
                                                                   ex.box.dim, cfg["batch"]))
    while True:
        if stop["flag"] or os.path.exists(os.path.join(run_dir, "STOP")):
            log("stop requested")
            break
        if hours and time.time() - started > hours * 3600:
            log("time budget reached")
            break
        if max_evals and len(ex.rows) - n0 >= max_evals:
            break
        if os.path.exists(os.path.join(run_dir, "PAUSE")):
            _write_status(run_dir, dict(ex.status(started, len(ex.rows) - n0), state="paused"))
            time.sleep(min(30.0, (hours or 1) * 3600))
            continue
        rows = ex.generation(gen, session)
        st = ex.status(started, len(ex.rows) - n0)
        _write_status(run_dir, dict(st, state="running"))
        new = sum(1 for r in rows if r.get("new_cell"))
        log("gen %d: +%d evals (%d new cells) | %d evals, %d cells (chao1 ~%s) | %s" % (
            gen, len(rows), new, st["n_evals"], st["cells_filled"], st["cells_estimated_chao1"],
            ", ".join("%s %d" % kv for kv in list(st["regimes"].items())[:4])))
        gen += 1
    _write_status(run_dir, dict(ex.status(started, len(ex.rows) - n0), state="stopped"))
    return ex


# ============================================================================ export

def export(run_dir, eval_ids=None, elites=False, regime=None, fp64=False, out_dir=None, spec_id=None):
    cfg = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8"))
    cfg.pop("provenance", None)
    _, base = load_config_from(cfg)
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "evals.jsonl"), encoding="utf-8") if line.strip()]
    if elites:
        best = {}
        for r in rows:
            if r.get("measures") is None or r["objective"] <= THRESHOLD_MIN:
                continue
            if regime and not r["regime"].startswith(regime):
                continue
            c = r.get("cell")
            if c is None:
                continue
            if c not in best or r["objective"] > best[c]["objective"]:
                best[c] = r
        chosen = sorted(best.values(), key=lambda r: r["i"])
    else:
        want = set(int(i) for i in eval_ids)
        chosen = [r for r in rows if r["i"] in want]
    if not chosen:
        raise SystemExit("nothing to export")
    specs = [make_spec(base, cfg, r["params"], r["ic_seed"], "e%07d" % r["i"]) for r in chosen]
    v = copy.deepcopy(specs[0])
    v["id"] = spec_id or "%s-%s" % (cfg["id"], "elites" if elites else "e%07d" % chosen[0]["i"])
    v["title"] = "QD %s: %d eval(s) %s" % (cfg["id"], len(chosen), "(fp64 verify)" if fp64 else "")
    v["description"] = ("Exported by tools/qd_explore.py from %s: evals %s." % (
        os.path.basename(os.path.abspath(run_dir)), ", ".join(str(r["i"]) for r in chosen)))[:4000]
    if fp64:
        v["protocol"]["precision"] = "fp64"
    if len(chosen) > 1:
        axes = {"protocol.ic.params.seed": [int(r["ic_seed"]) for r in chosen]}
        for p in cfg["box"]:
            axes[p] = [r["params"][p] for r in chosen]
        v["sweep"] = {"mode": "zip", "axes": axes}
    errs = irer_specs.validate(v)
    if errs:
        raise SystemExit("exported spec invalid: %s" % errs)
    out_dir = out_dir or os.path.join(ROOT, "specs", "proposed")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, v["id"] + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(v, fh, indent=2)
    return path, len(chosen)


def load_config_from(cfg):
    base = irer_specs.load(os.path.join(ROOT, cfg["base_spec"]) if not os.path.isabs(cfg["base_spec"])
                           else cfg["base_spec"])
    return cfg, base


# ============================================================================ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run (or resume) an exploration")
    r.add_argument("config")
    r.add_argument("--hours", type=float, default=None)
    r.add_argument("--max-evals", type=int, default=None)
    r.add_argument("--run-dir", default=None)
    for name in ("status", "pause", "resume", "stop"):
        p = sub.add_parser(name)
        p.add_argument("run_dir")
    e = sub.add_parser("export", help="evals -> a runnable spec in specs/proposed/")
    e.add_argument("run_dir")
    e.add_argument("--eval", type=int, nargs="*", default=None)
    e.add_argument("--elites", action="store_true", help="the best eval in every filled cell")
    e.add_argument("--regime", default=None, help="with --elites: only regimes starting with this")
    e.add_argument("--fp64", action="store_true")
    e.add_argument("--out-dir", default=None)
    e.add_argument("--id", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.config, a.hours, a.max_evals, a.run_dir)
    elif a.cmd == "status":
        p = os.path.join(a.run_dir, "status.json")
        print(open(p, encoding="utf-8").read() if os.path.exists(p) else "no status yet")
    elif a.cmd in ("pause", "stop"):
        open(os.path.join(a.run_dir, a.cmd.upper()), "w").close()
        print("%s requested: takes effect after the generation in flight" % a.cmd)
    elif a.cmd == "resume":
        for f in ("PAUSE",):
            if os.path.exists(os.path.join(a.run_dir, f)):
                os.remove(os.path.join(a.run_dir, f))
        print("resumed (a stopped run is restarted with `run` on the same config)")
    elif a.cmd == "export":
        if not a.elites and not a.eval:
            raise SystemExit("give --eval IDS or --elites")
        path, n = export(a.run_dir, a.eval, a.elites, a.regime, a.fp64, a.out_dir, a.id)
        print("%d eval(s) -> %s\nrun:  python tools/run_spec.py %s" % (n, path, os.path.relpath(path, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
