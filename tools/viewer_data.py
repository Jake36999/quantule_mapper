"""Data layer for the run gallery + viewer (tools/serve_viewer.py). No HTTP here, so it is testable.

Reads, never writes, except `queue_render`, which only drops a job file into specs/queue/ for the
human-started worker (tools/render_queue_worker.py). Same rule as the HUD: the UI has no control path
into a running simulation.

What a run can offer the viewer:
  final fields   any .npz/.h5 array whose last three dims form a cube (the tools/render_fields.py rule)
  time series    1-D arrays (e.g. er(t) in *_probe.npz) and telemetry.jsonl
  history        frames written by protocol.record (history/) or the HUD SnapshotWriter (snapshots/)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "docs", "runs", "_index.sqlite")
SWEEP = os.path.join(ROOT, "sweep_runs")
QUEUE = os.path.join(ROOT, "specs", "queue")
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,200}$")
#: wall-clock per ETDRK4 step at 96^3 on the GTX 1080, measured on the a* replay (64 min / 72000 steps)
SEC_PER_STEP_96 = 64 * 60 / 72000.0


def _rel(path):
    """Repo-relative path for display; absolute when it lives elsewhere (e.g. another drive)."""
    try:
        return os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:
        return path.replace(os.sep, "/")


def safe(name):
    return bool(name) and bool(_SAFE.match(name)) and ".." not in name


def _run_dir(run_id, index_dir=None):
    if not safe(run_id):
        raise ValueError("bad run id")
    for cand in (index_dir, os.path.join(SWEEP, run_id)):
        if cand and os.path.isdir(cand):
            return cand
    raise FileNotFoundError(run_id)


def _inside(path, root):
    return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)


# ----------------------------------------------------------------------------- gallery

def _badge(staleness, steppers):
    sts = [s for s in (staleness or "").split(",") if s]
    if any(s.endswith("STALE_PENDING_REVALIDATION") for s in sts):
        return "stale"
    if any(s.endswith("REVALIDATED") for s in sts):
        return "revalidated"
    if any(s.endswith("CURRENT") for s in sts):
        return "current"
    return "unaffected" if steppers else "unknown"


def _thumb(run_id, run_dir):
    for pat in (os.path.join(run_dir, "rendered", "*.png"),
                os.path.join(ROOT, "docs", "runs", "_figures", run_id, "*.png"),
                os.path.join(ROOT, "docs", "runs", "_plots", run_id, "*.png"),
                os.path.join(run_dir, "*.png")):
        hits = sorted(glob.glob(pat))
        if hits:
            return _rel(hits[0])
    return None


def _scan(run_dir):
    """Cheap capability scan (no array loads)."""
    files = glob.glob(os.path.join(run_dir, "*.npz")) + glob.glob(os.path.join(run_dir, "*", "*.npz")) + \
        glob.glob(os.path.join(run_dir, "*.h5")) + glob.glob(os.path.join(run_dir, "*", "*.h5"))
    files = [f for f in files if not os.path.basename(f).startswith("snap_")]
    hist = glob.glob(os.path.join(run_dir, "history", "snap_*.npz")) or \
        glob.glob(os.path.join(run_dir, "snapshots", "**", "snap_*.npz"), recursive=True)
    return {"n_field_files": len(files), "n_frames": len(hist),
            "has_spec": os.path.exists(os.path.join(run_dir, "spec.json"))}


def list_runs(substrate=None, branch=None, query=None, limit=400):
    if not os.path.exists(INDEX):
        return {"status": "NO_INDEX", "hint": "python tools/build_results_index.py"}
    con = sqlite3.connect("file:%s?mode=ro" % INDEX.replace("\\", "/"), uri=True)
    con.row_factory = sqlite3.Row
    q = ("SELECT r.run_id, r.date, r.substrate, r.family, r.branch, r.harness, r.steppers, r.verdict, r.run_dir, "
         "(SELECT GROUP_CONCAT(s.fix_id || ':' || s.status) FROM run_staleness s WHERE s.run_id = r.run_id) AS st "
         "FROM runs r WHERE 1=1")
    a = []
    for col, v in (("r.substrate", substrate), ("r.branch", branch)):
        if v:
            q += " AND %s = ?" % col
            a.append(v)
    if query:
        q += " AND r.run_id LIKE ?"
        a.append("%" + query + "%")
    q += " ORDER BY r.date DESC, r.run_id DESC LIMIT ?"
    a.append(int(limit))
    out = []
    for r in con.execute(q, a):
        d = dict(r)
        rd = d.pop("run_dir") or os.path.join(SWEEP, d["run_id"])
        if not os.path.isdir(rd):
            rd = os.path.join(SWEEP, d["run_id"])
        d["badge"] = _badge(d.pop("st"), d["steppers"])
        d.update(_scan(rd) if os.path.isdir(rd) else {"n_field_files": 0, "n_frames": 0, "has_spec": False})
        d["thumb"] = _thumb(d["run_id"], rd) if os.path.isdir(rd) else None
        out.append(d)
    # runs written since the index was last rebuilt (e.g. a fresh re-run with visuals): show them too,
    # marked "unindexed", so a user never has to rebuild the index to see what the worker just made
    known = {x["run_id"] for x in out}
    if not (substrate or branch):
        for d in sorted(glob.glob(os.path.join(SWEEP, "*")), key=os.path.getmtime, reverse=True)[:200]:
            rid = os.path.basename(d)
            if rid in known or not os.path.isdir(d) or (query and query not in rid):
                continue
            if not (os.path.exists(os.path.join(d, "spec.json")) or os.path.isdir(os.path.join(d, "history"))):
                continue
            card = {"run_id": rid, "date": time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(d))),
                    "substrate": None, "family": None, "branch": None, "harness": "run_spec.py",
                    "steppers": None, "verdict": None, "badge": "unindexed"}
            card.update(_scan(d))
            card["thumb"] = _thumb(rid, d)
            out.insert(0, card)
    facets = {k: sorted({x[k] for x in out if x[k]}) for k in ("substrate", "branch")}
    con.close()
    return {"status": "OK", "n": len(out), "runs": out, "facets": facets}


# ----------------------------------------------------------------------------- one run

def _npz_shapes(path):
    """{key: (shape, dtype)} from npz headers only -- never loads the arrays."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".npy"):
                continue
            with z.open(name) as fh:
                ver = np.lib.format.read_magic(fh)
                read = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = read(fh)
            out[name[:-4]] = (tuple(int(s) for s in shape), str(dtype))
    return out


def _h5_shapes(path):
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        def v(n, o):
            if isinstance(o, h5py.Dataset):
                out[n] = (tuple(int(s) for s in o.shape), str(o.dtype))
        f.visititems(v)
    return out


def _is_cube(shape):
    return len(shape) >= 3 and shape[-1] == shape[-2] == shape[-3] and shape[-1] >= 8


def _dt_T(run_dir):
    for name in ("summary.json", "config.json") + tuple(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*_summary.json"))):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            try:
                s = json.load(open(p, encoding="utf-8"))
            except ValueError:
                continue
            pr = s.get("protocol") or {}
            dt = pr.get("dt") or s.get("dt") or (s.get("config") or {}).get("dt")
            if dt:
                return float(dt), s
            return None, s
    return None, {}


def run_detail(run_id):
    rd = _run_dir(run_id)
    fields, series = [], []
    files = sorted(set(glob.glob(os.path.join(rd, "*.npz")) + glob.glob(os.path.join(rd, "*", "*.npz")) +
                       glob.glob(os.path.join(rd, "*.h5")) + glob.glob(os.path.join(rd, "*", "*.h5"))))
    for f in files:
        if os.path.basename(f).startswith("snap_"):
            continue
        rel = os.path.relpath(f, rd).replace(os.sep, "/")
        try:
            shapes = _npz_shapes(f) if f.endswith(".npz") else _h5_shapes(f)
        except Exception as exc:                     # a broken file must not break the page
            fields.append({"file": rel, "error": str(exc)[:200]})
            continue
        for k, (shp, dt) in shapes.items():
            if _is_cube(shp):
                fields.append({"file": rel, "key": k, "shape": shp, "dtype": dt,
                               "complex": dt.startswith("complex") or dt.startswith("<c"),
                               "n_time": shp[0] if len(shp) == 4 else 1})
            elif len(shp) == 1 and shp[0] >= 10 and "int" not in dt and "S" not in dt:
                series.append({"file": rel, "key": k, "length": shp[0]})
    if os.path.exists(os.path.join(rd, "telemetry.jsonl")) or glob.glob(os.path.join(rd, "telemetry", "*", "telemetry.jsonl")):
        series.append({"file": "telemetry", "key": "*", "length": None})
    dt, summ = _dt_T(rd)
    frames = sorted(glob.glob(os.path.join(rd, "history", "snap_*.npz"))) or \
        sorted(glob.glob(os.path.join(rd, "snapshots", "**", "snap_*.npz"), recursive=True))
    frame_fields = []
    if frames:
        frame_fields = sorted({k.split("__")[0] for k in _npz_shapes(frames[0]) if "__" in k})
    return {"status": "OK", "run_id": run_id, "fields": fields, "series": series, "dt": dt,
            "n_frames": len(frames), "frame_fields": frame_fields,
            "rerunnable": rerun_spec(run_id, None, dry=True).get("status") == "OK",
            "summary_keys": sorted(summ)[:40]}


QUANTITIES = ("abs2", "abs", "phase", "real", "imag")


def _quantity(a, q):
    if np.iscomplexobj(a):
        return {"abs2": np.abs(a) ** 2, "abs": np.abs(a), "phase": np.angle(a),
                "real": a.real, "imag": a.imag}[q]
    return a if q in ("real", "abs2", "abs") else np.zeros_like(a)


def _down(a, target):
    n = a.shape[-1]
    step = 1 if n <= target else int(np.ceil(n / target))
    return a[..., ::step, ::step, ::step]


def field_volume(run_id, file, key, quantity="abs2", target=48, t_index=-1):
    """-> (header dict, float32 bytes) for a cube field, downsampled and reduced to a real quantity."""
    rd = _run_dir(run_id)
    path = os.path.join(rd, file)
    if not _inside(path, rd) or not os.path.exists(path):
        raise FileNotFoundError(file)
    if quantity not in QUANTITIES:
        raise ValueError("quantity")
    if path.endswith(".npz"):
        with np.load(path, mmap_mode="r") as z:
            a = z[key]
            if a.ndim == 4:
                a = a[t_index]
            full_n = int(a.shape[-1])
            a = np.asarray(_down(a, int(target)))
    else:
        import h5py
        with h5py.File(path, "r") as f:
            ds = f[key]
            n = full_n = int(ds.shape[-1])
            step = 1 if n <= target else int(np.ceil(n / target))
            a = ds[t_index, ::step, ::step, ::step] if ds.ndim == 4 else ds[::step, ::step, ::step]
    v = np.ascontiguousarray(_quantity(a, quantity), dtype=np.float32)
    hdr = {"shape": list(v.shape), "min": float(np.nanmin(v)), "max": float(np.nanmax(v)), "quantity": quantity,
           "full_n": full_n}
    return hdr, v.tobytes()


def _downsample_series(t, y, n=3000):
    if len(y) <= n:
        return t.tolist(), y.tolist()
    k = int(np.ceil(len(y) / n))
    m = len(y) // k * k
    yy = y[:m].reshape(-1, k)
    tt = t[:m].reshape(-1, k)[:, 0]
    return tt.tolist(), yy.mean(1).tolist()


def time_series(run_id):
    """Every 1-D series (time axis in physical units when dt is known) + telemetry, downsampled for plotting,
    plus an ACTIVITY trace and suggested windows where the dynamics change fastest."""
    rd = _run_dir(run_id)
    dt, _ = _dt_T(rd)
    out = []
    for item in run_detail(run_id)["series"]:
        if item["file"] == "telemetry":
            continue
        with np.load(os.path.join(rd, item["file"]), mmap_mode="r") as z:
            y = np.asarray(z[item["key"]], dtype=np.float64)
        t = np.arange(len(y)) * (dt or 1.0)
        tt, yy = _downsample_series(t, y)
        out.append({"name": "%s:%s" % (item["file"], item["key"]), "t": tt, "y": yy,
                    "t_unit": "time" if dt else "step", "suggest": activity_windows(t, y)})
    from jax_scout.snapshots import read_telemetry
    tdirs = [rd] + sorted(glob.glob(os.path.join(rd, "telemetry", "*")))
    for d in tdirs:
        meta, rows = read_telemetry(d)
        if not rows:
            continue
        keys = [k for k in rows[-1] if k not in ("t", "wall_s") and isinstance(rows[-1][k], (int, float))]
        t = np.array([r.get("t", np.nan) for r in rows], dtype=float)
        for k in keys[:12]:
            y = np.array([r.get(k, np.nan) for r in rows], dtype=float)
            tt, yy = _downsample_series(t, y)
            out.append({"name": "%s%s" % ("" if d == rd else os.path.basename(d) + ":", k), "t": tt, "y": yy,
                        "t_unit": "time", "tolerance": (meta.get("invariants") or {}).get(k),
                        "suggest": activity_windows(t, y)})
    return {"status": "OK", "dt": dt, "series": out}


def activity_windows(t, y, n=3, frac=0.08, min_score=0.1):
    """Where does the BEHAVIOUR change? Uses |d2y/dt2| (curvature) smoothed over ~1% of the run, not
    |dy/dt|: a steady drift has a large slope everywhere but nothing worth a close-up, whereas transients,
    bends and oscillation (breathing) all show up as curvature. Returns up to n non-overlapping windows
    (each `frac` of the run) scoring >= min_score of the peak; an empty list means steady behaviour, for
    which a low-frame-rate recording of the whole run is the right choice. (The first version used
    |dy/dt| and suggested arbitrary points along a smooth decay.)"""
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) < 20 or t[-1] <= t[0]:
        return []
    if len(y) > 20000:                                    # curvature on a coarse grid: cheap and less noisy
        k = len(y) // 10000
        t, y = t[::k], y[::k]
    g = np.abs(np.gradient(np.gradient(y, t), t))
    w = max(3, len(g) // 100)
    g = np.convolve(g, np.ones(w) / w, mode="same")
    edge = max(1, w // 2)
    g[:edge] = g[edge]
    g[-edge:] = g[-edge - 1]                              # 'same' convolution under-weights the ends
    duration = t[-1] - t[0]
    # absolute floor: curvature that bends the signal by < 0.1% of its range over the whole run is
    # round-off on a straight line, not an episode (normalising it to 1 invented windows on a pure drift)
    if g.max() * duration ** 2 < 1e-3 * (np.ptp(y) + 1e-300):
        return []
    g = g / g.max()
    # one window per CONTIGUOUS above-threshold region (an oscillating episode is one window, not two at
    # its edges), padded by 1% of the run, at least frac/2 wide, ranked by peak score
    hot = g >= min_score
    regions, i = [], 0
    while i < len(g):
        if hot[i]:
            j = i
            while j + 1 < len(g) and hot[j + 1]:
                j += 1
            regions.append((i, j, float(g[i:j + 1].max())))
            i = j + 1
        else:
            i += 1
    pad, min_w = 0.01 * duration, 0.5 * frac * duration
    out = []
    for a, b, s in sorted(regions, key=lambda r: -r[2])[:n]:
        t0, t1 = t[a] - pad, t[b] + pad
        if t1 - t0 < min_w:
            c = 0.5 * (t0 + t1)
            t0, t1 = c - min_w / 2, c + min_w / 2
        out.append({"t0": float(max(t[0], t0)), "t1": float(min(t[-1], t1)), "score": round(s, 3)})
    return sorted(out, key=lambda w: w["t0"])


# ----------------------------------------------------------------------------- recorded history

def _frames(rd):
    return sorted(glob.glob(os.path.join(rd, "history", "snap_*.npz"))) or \
        sorted(glob.glob(os.path.join(rd, "snapshots", "**", "snap_*.npz"), recursive=True))


def history_index(run_id):
    rd = _run_dir(run_id)
    fr = _frames(rd)
    ts = []
    for f in fr:
        with np.load(f) as z:
            ts.append(float(z["t"]))
    return {"status": "OK", "n": len(fr), "t": ts, "fields": run_detail(run_id)["frame_fields"]}


def history_frame(run_id, i, field, quantity="abs2", kind="vol"):
    rd = _run_dir(run_id)
    fr = _frames(rd)
    i = int(i)
    if not 0 <= i < len(fr):
        raise IndexError(i)
    key = "%s__%s" % (field, kind)
    with np.load(fr[i]) as z:
        if key not in z.files:
            raise KeyError(key)
        a = z[key]
        t = float(z["t"])
    v = np.ascontiguousarray(_quantity(a, quantity), dtype=np.float32)
    return {"shape": list(v.shape), "min": float(np.nanmin(v)), "max": float(np.nanmax(v)), "t": t}, v.tobytes()


# ----------------------------------------------------------------------------- re-run with visuals

_FEB = {"param_D": 2.7329, "param_eta": 0.0704, "param_rho_vac": 1.1866, "param_omega0": 0.0,
        "param_a_coupling": 2.3098, "param_s": 0.0129, "param_f": -0.4861}
_FEB_A = 0.4802


def _probe_cell_spec(run_id, rd, cell):
    """Reconstruct a spec for one cell of the a* probe harnesses (feb_gain_ladder_longt / feb_astar_confirm),
    whose rows record a_factor, T (steps), seed; N, K in the summary; dt = 0.005 unless recorded."""
    summ = None
    for p in glob.glob(os.path.join(rd, "*_summary.json")):
        summ = json.load(open(p, encoding="utf-8"))
    if not summ or "rows" not in summ:
        return None
    row = next((r for r in summ["rows"] if r.get("key") == cell or (cell and r.get("key", "").startswith(cell))), None)
    if row is None:
        return None
    dt = float(summ.get("dt") or 0.005)
    steps = int(float(row.get("T") or summ.get("T")))
    params = dict(_FEB)
    params["param_a"] = _FEB_A * float(row["a_factor"])
    return {"spec_version": 1, "id": ("rerun-%s-%s" % (run_id, cell)).lower().replace("_", "-")[:80],
            "title": "Re-run with visuals: %s / %s" % (run_id, cell),
            "description": "Reconstructed by tools/viewer_data.py from the harness summary (FEB params, param_a x a_factor, "
                           "per-blob multiseed IC). Same physics as the original cell, on the current solver.",
            "substrate": {"name": "etdrk4-sncgl", "version": 1, "params": params},
            "protocol": {"ic": {"name": "multiseed", "version": 1,
                                "params": {"K": int(summ.get("K", 6)), "seed": int(float(row.get("seed") or summ.get("seed") or 20260619))}},
                         "grid": {"N": int(summ.get("N", 96)), "L": 10.0}, "dt": dt, "T": steps * dt},
            "observers": [{"name": "energy_ratio", "version": 1}, {"name": "nodes", "version": 1}],
            "prediction": {"statement": "Visual re-run of a recorded cell: the energy ratio stays below the 3.0 growth "
                                        "bound, as in the original run (er_max %s)." % row.get("er_max"),
                           "quantities": [{"key": "er", "expected": 3.0, "kind": "le"}]}}


def rerun_spec(run_id, cell=None, *, record=None, dry=False):
    """Build a spec that re-runs `run_id` (or one cell of it) with protocol.record switched on."""
    rd = _run_dir(run_id)
    p = os.path.join(rd, "spec.json")
    if os.path.exists(p):
        spec = json.load(open(p, encoding="utf-8"))
        spec.pop("sweep", None)
        spec["id"] = ("rerun-" + spec["id"])[:80]
    else:
        cells = [os.path.basename(f)[:-len("_probe.npz")] for f in glob.glob(os.path.join(rd, "*_probe.npz"))]
        cell = cell or (cells[0] if cells else None)
        spec = _probe_cell_spec(run_id, rd, cell) if cell else None
    if spec is None:
        return {"status": "UNSUPPORTED", "reason": "no spec.json and no known harness mapping for this run"}
    if dry:
        return {"status": "OK"}
    if record:
        spec["protocol"]["record"] = record
        win = record.get("window")
        if win:
            # a VISUAL re-run only needs to reach the end of its window: stop there instead of integrating
            # the rest of the original run (an early window costs a few % of the full run)
            pr = spec["protocol"]
            steps = int(np.ceil(float(win[1]) / pr["dt"] - 1e-9))
            pr["T"] = min(pr["T"], steps * pr["dt"])
            spec["description"] = (spec.get("description", "") + " Visual re-run: integration stops at the end "
                                   "of the recorded window (t = %g)." % pr["T"]).strip()
    return {"status": "OK", "spec": spec}


def estimate(spec):
    """Rough wall time and disk for a (recorded) spec, from the measured a* replay cost."""
    pr = spec["protocol"]
    steps = int(round(pr["T"] / pr["dt"]))
    n = pr["grid"]["N"]
    wall = steps * SEC_PER_STEP_96 * (n / 96.0) ** 3
    rec = pr.get("record")
    frames, mb = 0, 0.0
    if rec:
        lo, hi = rec.get("window", [0, pr["T"]])
        every = max(1, int(round(float(rec["every"]) / pr["dt"]))) * pr["dt"]   # what run_spec actually uses
        frames = int(round((min(hi, pr["T"]) - max(lo, 0)) / every)) + 1
        v = min(int(rec.get("volume", 48)), n)
        mb = frames * (v ** 3 * 8 + 3 * n * n * 8) / 1e6
    return {"steps": steps, "wall_min": round(wall / 60, 1), "frames": frames, "disk_mb": round(mb, 1),
            "note": "integration always starts at t=0 (it cannot jump ahead) and stops at the end of the window"}


def queue_render(spec, requested_by="viewer"):
    """Validate and drop the job in specs/queue/. Never runs anything: a human starts
    tools/render_queue_worker.py, which executes queued jobs one at a time."""
    import sys
    sys.path.insert(0, ROOT)
    import irer_specs
    spec = irer_specs.prune_empty(spec)
    errs = irer_specs.validate(spec)
    if errs:
        return {"status": "INVALID", "errors": errs}
    os.makedirs(QUEUE, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(QUEUE, "%s_%s.json" % (stamp, spec["id"]))
    with open(path, "x", encoding="utf-8") as fh:
        json.dump({"requested_by": requested_by, "requested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "estimate": estimate(spec), "spec": spec}, fh, indent=2)
    return {"status": "QUEUED", "path": _rel(path), "estimate": estimate(spec),
            "next": "start the worker if it is not running: python tools/render_queue_worker.py (in the JAX env)"}


def queue_status():
    jobs = []
    for sub in ("", "running", "done", "failed"):
        for p in sorted(glob.glob(os.path.join(QUEUE, sub, "*.json"))):
            try:
                j = json.load(open(p, encoding="utf-8"))
            except ValueError:
                continue
            jobs.append({"state": sub or "queued", "file": os.path.basename(p), "id": j["spec"]["id"],
                         "estimate": j.get("estimate"), "out": j.get("out"), "error": j.get("error")})
    return {"status": "OK", "jobs": jobs}
