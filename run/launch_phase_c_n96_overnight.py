#!/usr/bin/env python3
"""Phase C N96 overnight batch — deadline-bounded, priority-ordered replay queue.

Fills ~7 h of GPU time with the most decisive follow-ups to the trace-dynamics finding
(current closure is NOT the discriminator; a mass-holding capacity is), then auto-analyzes
and writes a morning review summary. Analysis/replay only — uses the existing
core_saturation_replay.py unchanged; NO PDE/solver/classifier/geometry/search change.

Priority queue (decisive first, so a deadline cut still leaves the important results done):
  P1  K6 high-mass longer-T ladder  -> is "spin-down" a real failure or a settling overshoot?
  P2  K6 near-threshold longer-T    -> does the still-rising case saturate or run to blowup?
  P3  K6 mid-mass longer-T control  -> does the survivor stay stable at long T?
  P4  seed expansion (x2 seeds)     -> is the Stage-1 mass-trajectory pattern seed-robust?
  P5  K6 mass-capacity ladder       -> map held-mass(input-mass) for the 6-node config (exploratory)

Each run is verdict-first unless trace=True (adds frames.npz + diagnostic_summary.json, ~2x time).
held-mass = sum|psi_fin|^2 is recovered post-hoc from probe_data.npz (free), so the longer-T
convergence test needs no trace. Robust to single-run failures (continue; abort only on 3 in a row).
Stops cleanly before DEADLINE_HOURS; writes OVERNIGHT_SUMMARY.md + results CSV in the output root
(generated output — NOT auto-committed; review and promote in the morning).
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
N = 96
DEADLINE_HOURS = 7.0
PER_RUN_TIMEOUT_S = 3 * 3600
RES_SCALED_TAG = "resolution_scaled_raw_target"

TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT = SWEEP / f"PHASE_C_N96_OVERNIGHT_{TS}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_ROOT / "overnight_orchestrator.log"
MANIFEST_PATH = OUT_ROOT / "overnight_manifest.json"

# Source rows (exact shortlisted Stage-1 rows)
ROW = {
    "k6_high": ("CORE_SAT_HUNT_20260624_112918", 32, 20260619, 16402.349616),  # csv, idx, row_seed, mass(=raw48x8)
    "k6_mid":  ("CORE_SAT_HUNT_20260624_142152", 34, 20260621, 8000.0),
    "k6_near": ("CORE_SAT_HUNT_20260624_102149", 33, 20260619, 8000.0),
    "k1_low":  ("CORE_SAT_HUNT_20260624_142152",  4, 20260621, 8000.0),
}


def spec(key, base, *, seed=None, mass=None, T, trace=False, scaled=True, block):
    csv_run, idx, row_seed, row_mass = ROW[base]
    return {
        "key": key, "block": block, "csv": csv_run, "idx": idx,
        "seed": seed if seed is not None else row_seed,
        "mass": mass if mass is not None else row_mass,
        "T": T, "trace": trace, "scaled": scaled,
    }


QUEUE = []
# P1 — K6 high-mass longer-T ladder (decisive)
for T, tr in [(9000, False), (12000, False), (18000, False), (24000, True), (36000, False), (48000, False)]:
    QUEUE.append(spec(f"k6high_T{T}{'_trace' if tr else ''}", "k6_high", T=T, trace=tr, block="P1_k6high_longT"))
# P2 — K6 near-threshold longer-T
for T, tr in [(12000, False), (24000, True), (36000, False)]:
    QUEUE.append(spec(f"k6near_T{T}{'_trace' if tr else ''}", "k6_near", T=T, trace=tr, block="P2_k6near_longT"))
# P3 — K6 mid-mass longer-T control
for T, tr in [(12000, False), (24000, True)]:
    QUEUE.append(spec(f"k6mid_T{T}{'_trace' if tr else ''}", "k6_mid", T=T, trace=tr, block="P3_k6mid_longT"))
# P4 — seed expansion (T=6000 verdict)
EXTRA_SEEDS = {
    "k6_high": [20260620, 20260621], "k6_mid": [20260619, 20260620],
    "k6_near": [20260620, 20260621], "k1_low": [20260619, 20260620],
}
for base, seeds in EXTRA_SEEDS.items():
    for sd in seeds:
        QUEUE.append(spec(f"{base}_seed{sd}", base, seed=sd, T=6000, trace=False, block="P4_seed_expansion"))
# P5 — K6 capacity mass-ladder (k6_mid IC/params/seed; explicit mass overrides; exploratory)
for m in [4000.0, 6000.0, 10000.0, 12000.0, 14000.0, 20000.0]:
    QUEUE.append(spec(f"k6cap_m{int(m)}", "k6_mid", mass=m, T=6000, trace=False, scaled=False, block="P5_capacity_ladder"))


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def est_minutes(s):
    return (s["T"] / 1000.0) * (2.0 if s["trace"] else 1.0) + 1.5


def wsl_cmd(s):
    out_wsl = f"sweep_runs/{OUT_ROOT.name}/{s['key']}"
    args = ["python", "jax_scout/core_saturation_replay.py",
            "--csv", f"sweep_runs/{s['csv']}/all_evals.csv", "--idx", str(s["idx"]),
            "--N", str(N), "--T", str(s["T"]), "--ic-seed-override", str(s["seed"]),
            "--target-initial-mass-override", f"{s['mass']}", "--out", out_wsl]
    if s["trace"]:
        args += ["--trace-snaps", "40"]
    inner = "source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && " + " ".join(args)
    return ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", inner]


def held_mass(probe_path):
    try:
        z = np.load(probe_path)
        pf = z["psi_fin"]
        if not np.all(np.isfinite(pf)):
            return float("inf"), False
        return float(np.sum(np.abs(pf) ** 2)), True
    except Exception:
        return None, None


manifest = {
    "output_root": str(OUT_ROOT), "timestamp": TS, "N": N, "deadline_hours": DEADLINE_HOURS,
    "queue_len": len(QUEUE), "runs": [], "completed": 0, "skipped_deadline": 0, "aborted": None,
}


def save_manifest():
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


log(f"=== Phase C N96 OVERNIGHT batch START | out={OUT_ROOT.name} | queue={len(QUEUE)} | deadline={DEADLINE_HOURS}h ===")
save_manifest()
start = time.time()
deadline = start + DEADLINE_HOURS * 3600
consec_fail = 0

for i, s in enumerate(QUEUE, 1):
    est = est_minutes(s)
    if time.time() + est * 60 > deadline:
        log(f"[{i}/{len(QUEUE)}] SKIP {s['key']} (est {est:.0f} min would pass deadline)")
        manifest["skipped_deadline"] += 1
        continue
    out_dir = OUT_ROOT / s["key"]
    rec = {"i": i, "key": s["key"], "block": s["block"], "T": s["T"], "seed": s["seed"],
           "input_mass": s["mass"], "trace": s["trace"], "scaled_expected": s["scaled"]}
    log(f"[{i}/{len(QUEUE)}] {s['block']} {s['key']} START (T={s['T']} seed={s['seed']} mass={s['mass']} trace={s['trace']} est~{est:.0f}m)")
    t0 = time.time()
    try:
        proc = subprocess.run(wsl_cmd(s), capture_output=True, text=True, timeout=PER_RUN_TIMEOUT_S)
        rc = proc.returncode
        stderr_tail = (proc.stderr or "").strip()[-600:]
    except subprocess.TimeoutExpired:
        rc, stderr_tail = -999, "timeout"
    rec["wallclock_min"] = round((time.time() - t0) / 60.0, 1)
    rec["returncode"] = rc
    rec["stderr_tail"] = stderr_tail
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "replay_stderr.log").write_text(stderr_tail, encoding="utf-8")
    except Exception:
        pass

    sj = out_dir / "summary.json"
    if rc == 0 and sj.exists():
        consec_fail = 0
        s_json = json.loads(sj.read_text(encoding="utf-8"))
        m = s_json.get("metrics") or {}
        rec["n96_class"] = s_json.get("klass")
        rec["er_fin"] = m.get("er_fin")
        rec["er_max"] = m.get("er_max")
        rec["n_fin"] = m.get("n_fin")
        rec["late_slope"] = m.get("late_slope")
        rec["mass_scaling_mode"] = s_json.get("mass_scaling_mode")
        rec["replay_resolution_N"] = s_json.get("replay_resolution_N")
        hm, fin = held_mass(out_dir / "probe_data.npz")
        rec["held_mass_raw"] = hm
        rec["held_over_input"] = (hm / s["mass"]) if (hm not in (None, float("inf")) and s["mass"]) else (float("inf") if hm == float("inf") else None)
        rec["psi_fin_finite"] = fin
        rec["frames_written"] = (out_dir / "frames.npz").exists()
        rec["status"] = "ok"
        log(f"    -> {rec['n96_class']} | er_fin={rec['er_fin']} held/in={rec['held_over_input']} | {rec['wallclock_min']}m")
    else:
        consec_fail += 1
        rec["status"] = "error"
        log(f"    !! ERROR rc={rc} (consec_fail={consec_fail}) {stderr_tail[-180:]}")
    manifest["runs"].append(rec)
    manifest["completed"] = sum(1 for r in manifest["runs"] if r.get("status") == "ok")
    save_manifest()
    if consec_fail >= 3:
        manifest["aborted"] = "3 consecutive failures (systematic error)"
        save_manifest(); log("!! ABORT: 3 consecutive failures"); break

manifest["total_hours"] = round((time.time() - start) / 3600.0, 3)
save_manifest()
log(f"=== RUNS DONE: completed={manifest['completed']} skipped={manifest['skipped_deadline']} total={manifest['total_hours']}h ===")

# ---------------------------------------------------------------- post-processing / morning summary
def write_summary():
    rows = manifest["runs"]
    ok = [r for r in rows if r.get("status") == "ok"]
    cols = ["key", "block", "T", "seed", "input_mass", "held_mass_raw", "held_over_input",
            "er_fin", "er_max", "n_fin", "late_slope", "n96_class", "mass_scaling_mode", "wallclock_min"]
    with (OUT_ROOT / "overnight_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def fmt(x, n=3):
        if x is None:
            return "—"
        if x == float("inf"):
            return "BLOWUP(inf)"
        try:
            return f"{float(x):.{n}f}"
        except Exception:
            return str(x)

    L = []
    L.append("# Phase C N96 — Overnight batch summary (auto-generated, for morning review)\n")
    L.append(f"Output root: `{OUT_ROOT.name}` · completed {manifest['completed']}/{manifest['queue_len']} "
             f"· skipped-by-deadline {manifest['skipped_deadline']} · total {manifest.get('total_hours')} h "
             f"· aborted: {manifest['aborted']}\n")
    L.append("Analysis/replay only — no PDE/solver/classifier/geometry/search change. This file lives in the\n"
             "generated output area and is NOT committed; promote to docs/ after review.\n")

    # P1: K6 high-mass longer-T convergence
    L.append("\n## P1 — K6 high-mass longer-T: failure, or settling overshoot?\n")
    L.append("Does held-mass flatten and `er_fin` recover above the 0.5 TRUE floor as T grows?\n\n")
    L.append("| T | class | er_fin | held_mass_raw | held/input |\n|---|---|---|---|---|\n")
    p1 = sorted([r for r in ok if r["block"] == "P1_k6high_longT"], key=lambda r: r["T"])
    # include the known T=6000 Stage-1 point for reference
    L.append(f"| 6000 (Stage1) | SPIN_DOWN_REJECT | 0.438 | 7180.3 | 0.438 |\n")
    for r in p1:
        L.append(f"| {r['T']} | {r.get('n96_class')} | {fmt(r.get('er_fin'))} | {fmt(r.get('held_mass_raw'),1)} | {fmt(r.get('held_over_input'))} |\n")
    L.append("\nRead: if class flips to TRUE/NEAR and held-mass plateaus → spin-down was a T-window artifact of "
             "a high-mass overshoot. If it stays SPIN_DOWN with held-mass still falling → genuine decay.\n")

    # P2 / P3 longer-T
    for blk, name in [("P2_k6near_longT", "P2 — K6 near-threshold longer-T (saturate or run away?)"),
                      ("P3_k6mid_longT", "P3 — K6 mid-mass longer-T (survivor stability control)")]:
        L.append(f"\n## {name}\n\n| T | class | er_fin | held_mass_raw | held/input |\n|---|---|---|---|---|\n")
        for r in sorted([x for x in ok if x["block"] == blk], key=lambda r: r["T"]):
            L.append(f"| {r['T']} | {r.get('n96_class')} | {fmt(r.get('er_fin'))} | {fmt(r.get('held_mass_raw'),1)} | {fmt(r.get('held_over_input'))} |\n")

    # P4 seed expansion
    L.append("\n## P4 — Seed expansion (is the Stage-1 pattern seed-robust?)\n\n")
    L.append("Stage-1 reference: k6_high=SPIN_DOWN, k6_mid=TRUE, k6_near=TRUE(rising), k1_low=BLOWUP.\n\n")
    L.append("| case | seed | class | er_fin | held/input |\n|---|---|---|---|---|\n")
    for r in [x for x in ok if x["block"] == "P4_seed_expansion"]:
        L.append(f"| {r['key'].rsplit('_seed',1)[0]} | {r['seed']} | {r.get('n96_class')} | {fmt(r.get('er_fin'))} | {fmt(r.get('held_over_input'))} |\n")

    # P5 capacity ladder
    L.append("\n## P5 — K6 mass-capacity ladder (held-mass vs input-mass, k6_mid IC, T=6000, exploratory)\n\n")
    L.append("Tests the holding-capacity reading: above capacity → shed (held<input); below → hold/grow.\n\n")
    L.append("| input_mass | class | er_fin | held_mass_raw | held/input |\n|---|---|---|---|---|\n")
    for r in sorted([x for x in ok if x["block"] == "P5_capacity_ladder"], key=lambda r: r["input_mass"]):
        L.append(f"| {fmt(r['input_mass'],0)} | {r.get('n96_class')} | {fmt(r.get('er_fin'))} | {fmt(r.get('held_mass_raw'),1)} | {fmt(r.get('held_over_input'))} |\n")

    errs = [r for r in rows if r.get("status") != "ok"]
    if errs:
        L.append("\n## Errors / skips\n")
        for r in errs:
            L.append(f"- {r['key']}: rc={r.get('returncode')} {str(r.get('stderr_tail'))[-160:]}\n")

    L.append("\n## Suggested morning read\n")
    L.append("- P1 is the headline: K6 high-mass failure-vs-overshoot.\n")
    L.append("- P4 tells whether to trust the Stage-1 single-seed verdicts.\n")
    L.append("- P5 sketches the capacity curve; trace bundles (T=24000 runs) feed phase-c-current-closure-dynamics.\n")
    (OUT_ROOT / "OVERNIGHT_SUMMARY.md").write_text("".join(L), encoding="utf-8")
    log(f"wrote {OUT_ROOT/'OVERNIGHT_SUMMARY.md'}")


try:
    write_summary()
except Exception as exc:  # never lose run data to a summary bug
    log(f"!! summary generation failed: {exc}")

# bonus: dynamics figures on any trace bundles produced (best-effort)
try:
    if any((OUT_ROOT / r["key"] / "frames.npz").exists() for r in manifest["runs"]):
        dyn_out = OUT_ROOT / "closure_dynamics"
        subprocess.run(["python", "-m", "quantule_viz", "phase-c-current-closure-dynamics",
                        str(OUT_ROOT), "--outdir", str(dyn_out)],
                       capture_output=True, text=True, timeout=1800)
        log(f"dynamics figures -> {dyn_out}")
except Exception as exc:
    log(f"dynamics post-proc skipped: {exc}")

log("=== OVERNIGHT BATCH COMPLETE ===")
print("OVERNIGHT_ORCHESTRATOR_EXIT")
