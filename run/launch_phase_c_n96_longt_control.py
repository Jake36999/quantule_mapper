#!/usr/bin/env python3
"""Phase C N96 longer-T control batch — the decisive control for the overnight result.

feb56dc7 (reference bound state), K4 and K2 (the other Stage-1 "supported" branches) at T=24000
with trace. Answers: (a) is long-T integration numerically trustworthy (does feb stay a stable
bound state?), (b) is ANY of these a genuine long-time attractor, (c) do K4/K2 survive long T like
or unlike K6-mid (which blew up by T=24000)?

Analysis/replay only — exact rows, existing replay path; no PDE/solver/classifier/geometry/search
change. Continue-on-error (3-run control; we want all results). Generated bundles not committed.
"""
import csv as _csv
import datetime
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "sweep_runs"
N, T, SNAPS = 96, 24000, 40
PER_RUN_TIMEOUT_S = 3 * 3600

TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT = SWEEP / f"PHASE_C_N96_LONGT_CONTROL_{TS}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_ROOT / "control_orchestrator.log"
MANIFEST_PATH = OUT_ROOT / "control_manifest.json"

# key, ref?, csv, idx, seed, mass (None for ref)
CASES = [
    ("feb56dc7_T24000", "feb56dc7", None, None, None, None),
    ("k4_T24000", None, "CORE_SAT_HUNT_20260624_124444", 25, 20260620, 9600.0),
    ("k2_T24000", None, "CORE_SAT_HUNT_20260624_152029", 10, 20260621, 16402.349616),
]


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def wsl_cmd(key, ref, csv_run, idx, seed, mass):
    out_wsl = f"sweep_runs/{OUT_ROOT.name}/{key}"
    args = ["python", "jax_scout/core_saturation_replay.py", "--N", str(N), "--T", str(T),
            "--trace-snaps", str(SNAPS), "--out", out_wsl]
    if ref:
        args += ["--ref", ref]
    else:
        args += ["--csv", f"sweep_runs/{csv_run}/all_evals.csv", "--idx", str(idx),
                 "--ic-seed-override", str(seed), "--target-initial-mass-override", f"{mass}"]
    inner = "source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && " + " ".join(args)
    return ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", inner]


def held_mass(p):
    try:
        z = np.load(p); pf = z["psi_fin"]
        if not np.all(np.isfinite(pf)):
            return float("inf"), False
        return float(np.sum(np.abs(pf) ** 2)), True
    except Exception:
        return None, None


manifest = {"output_root": str(OUT_ROOT), "timestamp": TS, "N": N, "T": T, "trace_snaps": SNAPS, "runs": []}


def save_manifest():
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


log(f"=== Phase C N96 longer-T CONTROL batch START (T={T}) | out={OUT_ROOT.name} ===")
save_manifest()
t_start = time.time()

for i, (key, ref, csv_run, idx, seed, mass) in enumerate(CASES, 1):
    out_dir = OUT_ROOT / key
    rec = {"i": i, "key": key, "ref": ref, "idx": idx, "seed": seed, "input_mass": mass}
    log(f"[{i}/3] {key} START (ref={ref} mass={mass})")
    t0 = time.time()
    try:
        proc = subprocess.run(wsl_cmd(key, ref, csv_run, idx, seed, mass),
                              capture_output=True, text=True, timeout=PER_RUN_TIMEOUT_S)
        rc = proc.returncode
        stderr_tail = (proc.stderr or "").strip()[-600:]
    except subprocess.TimeoutExpired:
        rc, stderr_tail = -999, "timeout"
    rec["wallclock_min"] = round((time.time() - t0) / 60.0, 1)
    rec["returncode"] = rc
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "replay_stderr.log").write_text(stderr_tail, encoding="utf-8")
    except Exception:
        pass
    sj = out_dir / "summary.json"
    if sj.exists():
        s = json.loads(sj.read_text(encoding="utf-8")); m = s.get("metrics") or {}
        rec["class"] = s.get("klass"); rec["er_fin"] = m.get("er_fin"); rec["er_max"] = m.get("er_max")
        rec["n_fin"] = m.get("n_fin"); rec["late_slope"] = m.get("late_slope")
        rec["mass_scaling_mode"] = s.get("mass_scaling_mode")
        hm, fin = held_mass(out_dir / "probe_data.npz")
        rec["held_mass_raw"] = hm; rec["psi_fin_finite"] = fin
        rec["frames_written"] = (out_dir / "frames.npz").exists()
        rec["status"] = "ok" if rc == 0 else "ok_rc_nonzero"
        log(f"    -> {rec['class']} er_fin={rec['er_fin']} er_max={rec['er_max']} held={hm} rc={rc} | {rec['wallclock_min']}m")
    else:
        rec["status"] = "error_no_summary"; rec["stderr_tail"] = stderr_tail
        log(f"    !! no summary.json rc={rc} {stderr_tail[-160:]}")
    manifest["runs"].append(rec)
    save_manifest()

manifest["total_hours"] = round((time.time() - t_start) / 3600.0, 3)
save_manifest()

# quick summary
with (OUT_ROOT / "control_results.csv").open("w", newline="", encoding="utf-8") as fh:
    cols = ["key", "ref", "input_mass", "class", "er_fin", "er_max", "n_fin", "held_mass_raw", "mass_scaling_mode", "wallclock_min", "status"]
    w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in manifest["runs"]:
        w.writerow(r)

log(f"=== CONTROL DONE: {len(manifest['runs'])} runs, total={manifest['total_hours']}h ===")
# bonus dynamics figures
try:
    if any((OUT_ROOT / r["key"] / "frames.npz").exists() for r in manifest["runs"]):
        subprocess.run(["python", "-m", "quantule_viz", "phase-c-current-closure-dynamics",
                        str(OUT_ROOT), "--outdir", str(OUT_ROOT / "closure_dynamics")],
                       capture_output=True, text=True, timeout=1800)
        log("dynamics figures written")
except Exception as exc:
    log(f"dynamics post-proc skipped: {exc}")
print("CONTROL_ORCHESTRATOR_EXIT")
