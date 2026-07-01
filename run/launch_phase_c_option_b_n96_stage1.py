#!/usr/bin/env python3
"""Phase C Option B — Stage 1 N96/T6000 verdict-first replay orchestrator.

Windows-side sequential controller. For each of the 8 shortlist candidates it invokes
jax_scout/core_saturation_replay.py inside the WSL jax venv (FP64, N=96, T=6000),
verdict-first (NO --trace-snaps), validates the stamped mass-resolution metadata, and
enforces the agreed hard-stop conditions. Read-only w.r.t. physics: the replay path,
solver, classifier, and gravity/unified_omega.py are untouched.

Hard stops (abort remaining runs):
  1. replay process exits non-zero (code/env error, not a scientific failure)
  2. no summary.json written
  3. mass-resolution metadata absent or wrong
  4. feb56dc7 control does not reproduce (not TRUE/NEAR)
A scientific failure (e.g. BLOWUP) is an OUTCOME, not a stop.

Outputs (never auto-committed):
  sweep_runs/PHASE_C_OPTION_B_N96_STAGE1_<ts>/
    <key>/                      per-candidate replay dir (summary.json, probe_data.npz)
    stage1_manifest.json        machine-readable progress + verdicts
    stage1_orchestrator.log     human log
"""
import json
import subprocess
import time
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # F:\quantule_mapper
SWEEP = ROOT / "sweep_runs"
N, T = 96, 6000
PER_RUN_TIMEOUT_S = 6 * 3600                           # safety net per single config

TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT = SWEEP / f"PHASE_C_OPTION_B_N96_STAGE1_{TS}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_ROOT / "stage1_orchestrator.log"
MANIFEST_PATH = OUT_ROOT / "stage1_manifest.json"

# priority, key, csv_run (None=ref), idx, K, seed, n96_raw_override (None=ref), ref
CANDIDATES = [
    ("k6_high_mass_true",     "CORE_SAT_HUNT_20260624_112918", 32, 6, 20260619, 16402.349616, None),
    ("k6_mid_mass_true",      "CORE_SAT_HUNT_20260624_142152", 34, 6, 20260621,  8000.0,       None),
    ("k4_intermediate_true",  "CORE_SAT_HUNT_20260624_124444", 25, 4, 20260620,  9600.0,       None),
    ("feb56dc7_control",      None,                            None, 6, None,    None,          "feb56dc7"),
    ("k2_intermediate_true",  "CORE_SAT_HUNT_20260624_152029", 10, 2, 20260621, 16402.349616, None),
    ("k1_low_mass_true",      "CORE_SAT_HUNT_20260624_142152",  4, 1, 20260621,  8000.0,       None),
    ("k1_high_mass_failure",  "CORE_SAT_HUNT_20260624_132302",  0, 1, 20260620, 16402.349616, None),
    ("k6_near_threshold_near","CORE_SAT_HUNT_20260624_102149", 33, 6, 20260619,  8000.0,       None),
]


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def wsl_replay_cmd(key, csv_run, idx, seed, override, ref):
    out_wsl = f"sweep_runs/PHASE_C_OPTION_B_N96_STAGE1_{TS}/{key}"
    args = ["python", "jax_scout/core_saturation_replay.py",
            "--N", str(N), "--T", str(T), "--out", out_wsl]
    if ref:
        args += ["--ref", ref]
    else:
        args += ["--csv", f"sweep_runs/{csv_run}/all_evals.csv", "--idx", str(idx),
                 "--ic-seed-override", str(seed),
                 "--target-initial-mass-override", f"{override}"]
    inner = "source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && " + " ".join(args)
    return ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", inner], args


manifest = {
    "output_root": str(OUT_ROOT),
    "timestamp": TS,
    "N": N, "T": T, "dtype": "FP64 (complex128)",
    "trace_policy": "verdict-first (no --trace-snaps)",
    "priority_order": [c[0] for c in CANDIDATES],
    "runs": [],
    "stopped_early": None,
    "completed_ok": 0,
    "total_wallclock_sec": None,
}


def save_manifest():
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def stop(rec, status, reason):
    rec["status"] = status
    manifest["runs"].append(rec)
    manifest["stopped_early"] = reason
    save_manifest()
    log(f"!! HARD STOP after {rec['key']}: {reason}")


log(f"=== Phase C Option B Stage 1 N96/T6000 verdict-first batch START | out={OUT_ROOT.name} ===")
save_manifest()
batch_t0 = time.time()
stopped = False

for prio, (key, csv_run, idx, K, seed, override, ref) in enumerate(CANDIDATES, 1):
    out_dir = OUT_ROOT / key
    cmd, args = wsl_replay_cmd(key, csv_run, idx, seed, override, ref)
    rec = {"priority": prio, "key": key, "K": K, "idx": idx, "ic_seed": seed,
           "n96_raw_override": override, "ref": ref, "out_dir": str(out_dir)}
    log(f"--- [{prio}/8] {key} START (K={K}, idx={idx}, override={override}, ref={ref}) ---")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=PER_RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        rec["wallclock_sec"] = round(time.time() - t0, 1)
        stop(rec, "ERROR_TIMEOUT", f"{key}: exceeded {PER_RUN_TIMEOUT_S}s per-run timeout")
        stopped = True
        break
    rec["wallclock_sec"] = round(time.time() - t0, 1)
    rec["returncode"] = proc.returncode
    # forensic dumps
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "replay_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (out_dir / "replay_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    except Exception:
        pass
    rec["stdout_tail"] = (proc.stdout or "").strip()[-400:]
    rec["stderr_tail"] = (proc.stderr or "").strip()[-800:]

    if proc.returncode != 0:
        stop(rec, "ERROR_NONZERO_EXIT", f"{key}: replay exited {proc.returncode} (code/env error)")
        stopped = True
        break

    sj = out_dir / "summary.json"
    if not sj.exists():
        stop(rec, "ERROR_NO_SUMMARY", f"{key}: no summary.json written")
        stopped = True
        break

    s = json.loads(sj.read_text(encoding="utf-8"))
    m = s.get("metrics") or {}
    rec["n96_class"] = s.get("klass")
    rec["metrics"] = {k: m.get(k) for k in ("er_fin", "er_max", "n_mid", "n_fin", "core_fin", "late_slope")}
    rec["replay_resolution_N"] = s.get("replay_resolution_N")
    rec["source_resolution_N"] = s.get("source_resolution_N")
    rec["mass_scaling_mode"] = s.get("mass_scaling_mode")
    rec["replay_kind"] = s.get("replay_kind")
    rec["target_raw_mass_replay"] = s.get("target_raw_mass_replay")
    rec["replay_target_initial_mass"] = s.get("replay_target_initial_mass")
    rec["target_integral_mass"] = s.get("target_integral_mass")
    rec["initial_mass"] = (s.get("ic_stats") or {}).get("initial_mass")

    # metadata validation
    meta_ok = (rec["replay_resolution_N"] == 96)
    if ref:
        meta_ok = meta_ok and (rec["mass_scaling_mode"] == "exact_saved_raw_target")
    else:
        meta_ok = meta_ok and (rec["mass_scaling_mode"] == "resolution_scaled_raw_target")
        trm = rec["target_raw_mass_replay"]
        meta_ok = meta_ok and (trm is not None) and (abs(float(trm) - float(override)) <= 1e-3)
    rec["metadata_ok"] = bool(meta_ok)
    if not meta_ok:
        stop(rec, "ERROR_BAD_METADATA",
             f"{key}: mass-resolution metadata absent/wrong "
             f"(N={rec['replay_resolution_N']}, mode={rec['mass_scaling_mode']}, raw={rec['target_raw_mass_replay']})")
        stopped = True
        break

    # feb reproduction gate
    if ref == "feb56dc7" and rec["n96_class"] not in ("TRUE_SATURATED_BOUND_STATE", "NEAR_SATURATED_BOUND_STATE"):
        stop(rec, "ERROR_FEB_NOT_REPRODUCED", f"feb56dc7 did not reproduce (class={rec['n96_class']})")
        stopped = True
        break

    rec["status"] = "OK"
    manifest["runs"].append(rec)
    manifest["completed_ok"] = sum(1 for r in manifest["runs"] if r.get("status") == "OK")
    save_manifest()
    log(f"[{prio}/8] {key} -> {rec['n96_class']} | n_fin={rec['metrics']['n_fin']} "
        f"slope={rec['metrics']['late_slope']} er_fin={rec['metrics']['er_fin']} "
        f"| meta={rec['mass_scaling_mode']} | {rec['wallclock_sec']/60:.1f} min")

manifest["total_wallclock_sec"] = round(time.time() - batch_t0, 1)
manifest["completed_ok"] = sum(1 for r in manifest["runs"] if r.get("status") == "OK")
save_manifest()
log(f"=== BATCH DONE: completed_ok={manifest['completed_ok']}/8, "
    f"total={manifest['total_wallclock_sec']/3600:.2f}h, stopped_early={manifest['stopped_early']} ===")
print("STAGE1_ORCHESTRATOR_EXIT")
