#!/usr/bin/env python3
"""Phase C Option B — focused-4 N96/T6000 trace-snap replay orchestrator.

Captures time-resolved frames (frames.npz + diagnostic_summary.json) for the 4 cases where
the static single-snapshot closure analysis was inconclusive and dynamics are needed:
  K6 high-mass (spin-down) vs K6 mid-mass (survivor)  -> time-averaged current leak-vs-closed
  K6 near-threshold (rising)                          -> settling vs still-rising
  K1 low-mass (blowup, ordered LAST)                  -> pre-blowup morphology onset

Exact shortlisted rows, FP64, N=96, T=6000, --trace-snaps 40. Read-only w.r.t. physics:
uses the existing core_saturation_replay.py unchanged. No PDE/solver/classifier/geometry/search change.

Hard stops for the finite cases (non-zero exit, no summary.json, bad mass-resolution metadata).
K1 is last and known to blow up: a trace-capture failure there is recorded, not treated as a batch abort.
"""
import json
import subprocess
import time
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "sweep_runs"
N, T, SNAPS = 96, 6000, 40
PER_RUN_TIMEOUT_S = 6 * 3600

TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT = SWEEP / f"PHASE_C_OPTION_B_N96_TRACE_{TS}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_ROOT / "trace_orchestrator.log"
MANIFEST_PATH = OUT_ROOT / "trace_manifest.json"

# priority, key, csv_run, idx, K, seed, override, is_blowup_case
CANDIDATES = [
    ("k6_high_mass_true",     "CORE_SAT_HUNT_20260624_112918", 32, 6, 20260619, 16402.349616, False),
    ("k6_mid_mass_true",      "CORE_SAT_HUNT_20260624_142152", 34, 6, 20260621,  8000.0,       False),
    ("k6_near_threshold_near","CORE_SAT_HUNT_20260624_102149", 33, 6, 20260619,  8000.0,       False),
    ("k1_low_mass_true",      "CORE_SAT_HUNT_20260624_142152",  4, 1, 20260621,  8000.0,       True),
]


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def wsl_cmd(key, csv_run, idx, seed, override):
    out_wsl = f"sweep_runs/PHASE_C_OPTION_B_N96_TRACE_{TS}/{key}"
    args = ["python", "jax_scout/core_saturation_replay.py",
            "--csv", f"sweep_runs/{csv_run}/all_evals.csv", "--idx", str(idx),
            "--N", str(N), "--T", str(T), "--ic-seed-override", str(seed),
            "--target-initial-mass-override", f"{override}", "--trace-snaps", str(SNAPS),
            "--out", out_wsl]
    inner = "source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && " + " ".join(args)
    return ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", inner]


manifest = {
    "output_root": str(OUT_ROOT), "timestamp": TS, "N": N, "T": T, "trace_snaps": SNAPS,
    "dtype": "FP64 (complex128)", "order": [c[0] for c in CANDIDATES],
    "runs": [], "stopped_early": None, "total_wallclock_sec": None,
}


def save_manifest():
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


log(f"=== Phase C Option B focused-4 N96 TRACE batch START | out={OUT_ROOT.name} ===")
save_manifest()
batch_t0 = time.time()

for prio, (key, csv_run, idx, K, seed, override, is_blowup) in enumerate(CANDIDATES, 1):
    out_dir = OUT_ROOT / key
    rec = {"priority": prio, "key": key, "K": K, "idx": idx, "ic_seed": seed,
           "n96_raw_override": override, "is_blowup_case": is_blowup, "out_dir": str(out_dir)}
    log(f"--- [{prio}/4] {key} START (trace-snaps={SNAPS}) ---")
    t0 = time.time()
    try:
        proc = subprocess.run(wsl_cmd(key, csv_run, idx, seed, override),
                              capture_output=True, text=True, timeout=PER_RUN_TIMEOUT_S)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rec["wallclock_sec"] = round(time.time() - t0, 1)
        rec["status"] = "ERROR_TIMEOUT"
        manifest["runs"].append(rec); manifest["stopped_early"] = f"{key}: timeout"; save_manifest()
        log(f"!! STOP: {key} timeout"); break
    rec["wallclock_sec"] = round(time.time() - t0, 1)
    rec["returncode"] = rc
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "replay_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (out_dir / "replay_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    except Exception:
        pass
    rec["stderr_tail"] = (proc.stderr or "").strip()[-800:]

    sj = out_dir / "summary.json"
    fz = out_dir / "frames.npz"
    ds = out_dir / "diagnostic_summary.json"
    rec["frames_written"] = fz.exists()
    rec["diag_written"] = ds.exists()

    if rc != 0:
        if is_blowup:
            rec["status"] = "BLOWUP_TRACE_PARTIAL"  # expected: capture may stop at non-finite
            manifest["runs"].append(rec); save_manifest()
            log(f"[{prio}/4] {key} returncode={rc} (blowup case) frames={fz.exists()} diag={ds.exists()} -- recorded, continuing")
            continue
        rec["status"] = "ERROR_NONZERO_EXIT"
        manifest["runs"].append(rec); manifest["stopped_early"] = f"{key}: exit {rc}"; save_manifest()
        log(f"!! STOP: {key} exit {rc}"); break

    if not sj.exists():
        rec["status"] = "ERROR_NO_SUMMARY"
        manifest["runs"].append(rec); manifest["stopped_early"] = f"{key}: no summary.json"; save_manifest()
        log(f"!! STOP: {key} no summary.json"); break

    s = json.loads(sj.read_text(encoding="utf-8"))
    m = s.get("metrics") or {}
    rec["n96_class"] = s.get("klass")
    rec["metrics"] = {k: m.get(k) for k in ("er_fin", "er_max", "n_mid", "n_fin", "core_fin", "late_slope")}
    rec["replay_resolution_N"] = s.get("replay_resolution_N")
    rec["mass_scaling_mode"] = s.get("mass_scaling_mode")
    rec["target_raw_mass_replay"] = s.get("target_raw_mass_replay")

    meta_ok = (rec["replay_resolution_N"] == 96 and rec["mass_scaling_mode"] == "resolution_scaled_raw_target"
               and rec["target_raw_mass_replay"] is not None
               and abs(float(rec["target_raw_mass_replay"]) - float(override)) <= 1e-3)
    rec["metadata_ok"] = bool(meta_ok)
    if not meta_ok and not is_blowup:
        rec["status"] = "ERROR_BAD_METADATA"
        manifest["runs"].append(rec); manifest["stopped_early"] = f"{key}: bad metadata"; save_manifest()
        log(f"!! STOP: {key} bad metadata"); break

    rec["status"] = "OK"
    manifest["runs"].append(rec); save_manifest()
    log(f"[{prio}/4] {key} -> {rec['n96_class']} | frames={fz.exists()} diag={ds.exists()} "
        f"| n_fin={rec['metrics']['n_fin']} slope={rec['metrics']['late_slope']} | {rec['wallclock_sec']/60:.1f} min")

manifest["total_wallclock_sec"] = round(time.time() - batch_t0, 1)
save_manifest()
log(f"=== TRACE BATCH DONE: total={manifest['total_wallclock_sec']/3600:.2f}h stopped_early={manifest['stopped_early']} ===")
print("TRACE_ORCHESTRATOR_EXIT")
