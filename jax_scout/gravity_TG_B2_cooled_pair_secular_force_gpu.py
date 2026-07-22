"""TG-B2 cooled-pair secular-force campaign wrapper.

This runner reuses `gravity_TG_B2_definitive_force.py` for the actual GPU
field evolutions. It adds a preregistered cooled-pair matrix and post-processes
the returned scalar series for loop-induced body force, right-half momentum
impulse, separation drift, and feedback-off breathing.

No frozen TG-B1S equations are modified here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(outdir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.csv":
            rows.append({"path": str(path.relative_to(outdir)).replace("\\", "/"), "sha256": sha256(path)})
    return rows


def git_state() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception as exc:
        return f"git status failed: {exc}\n"


def preflight() -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    backend = jax.default_backend()
    ok = backend == "gpu" and any(device.platform == "gpu" for device in devices)
    return {
        "backend": backend,
        "devices": [str(device) for device in devices],
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "gpu_ok": bool(ok),
    }


def series(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    out: dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        try:
            out[key] = np.asarray([float(row[key]) for row in rows], dtype=float)
        except Exception:
            pass
    return out


def slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 3:
        return float("nan")
    return float(np.polyfit(t, y, 1)[0])


def analyze_subrun(subdir: Path, discard: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    well_files = sorted(subdir.glob("scalars_sep*_well.csv"))
    for well_path in well_files:
        stem = well_path.stem
        sep_tag = stem.replace("scalars_", "").replace("_well", "")
        off_path = subdir / f"scalars_{sep_tag}_off.csv"
        hill_path = subdir / f"scalars_{sep_tag}_hill.csv"
        if not off_path.exists() or not hill_path.exists():
            rows.append({"subrun": subdir.name, "sep_tag": sep_tag, "status": "MISSING_ARM"})
            continue
        well = series(well_path)
        off = series(off_path)
        hill = series(hill_path)
        n = min(len(well["t"]), len(off["t"]), len(hill["t"]))
        t = well["t"][:n]
        mask = t >= discard * t[-1]
        if np.count_nonzero(mask) < 4:
            mask = np.arange(n) >= max(0, n // 2)
        p_loop = well["P_R"][:n] - off["P_R"][:n]
        sep_loop = well["sep"][:n] - off["sep"][:n]
        body_force = well["F_R"][:n]
        hill_force = hill["F_R"][:n]
        off_force = off["F_R"][:n]
        off_sep = off["sep"][:n]
        off_amp = off["amp"][:n]
        rows.append(
            {
                "subrun": subdir.name,
                "sep_tag": sep_tag,
                "t_end": float(t[-1]),
                "settled_samples": int(np.count_nonzero(mask)),
                "F_R_well_mean": float(np.mean(body_force[mask])),
                "F_R_hill_mean": float(np.mean(hill_force[mask])),
                "F_R_off_mean": float(np.mean(off_force[mask])),
                "body_force_direction": "ATTRACT" if float(np.mean(body_force[mask])) < 0 else "REPEL_OR_NULL",
                "body_force_sign_reverses": bool(np.mean(body_force[mask]) * np.mean(hill_force[mask]) < 0),
                "P_loop_slope": slope(t[mask], p_loop[mask]),
                "P_loop_direction": "INWARD" if slope(t[mask], p_loop[mask]) < 0 else "OUTWARD_OR_NULL",
                "sep_loop_slope": slope(t[mask], sep_loop[mask]),
                "sep_loop_direction": "CLOSING" if slope(t[mask], sep_loop[mask]) < 0 else "OPENING_OR_NULL",
                "off_sep_peak_to_peak": float(np.max(off_sep[mask]) - np.min(off_sep[mask])),
                "off_amp_peak_to_peak": float(np.max(off_amp[mask]) - np.min(off_amp[mask])),
                "cooling_reduced_breathing_candidate": "",
                "status": "ANALYZED",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "sweep_runs" / f"TG_B2_COOLED_PAIR_SECULAR_FORCE_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--seps", default="3.0,4.0")
    parser.add_argument("--cool-T-values", default="0,40")
    parser.add_argument("--T", type=float, default=180.0)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=16.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--sample-dt", type=float, default=0.5)
    parser.add_argument("--discard", type=float, default=0.4)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.out)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "git_state_before.txt").write_text(git_state(), encoding="utf-8")

    cool_values = [float(value.strip()) for value in args.cool_T_values.split(",") if value.strip()]
    matrix = [
        {
            "run_id": f"coolT_{cool:g}".replace(".", "p"),
            "cool_T": cool,
            "seps": args.seps,
            "T": args.T,
            "N": args.N,
            "L": args.L,
            "dt": args.dt,
            "sample_dt": args.sample_dt,
        }
        for cool in cool_values
    ]
    write_json(outdir / "preregistered_matrix.json", matrix)
    write_json(
        outdir / "model_boundary.json",
        {
            "source": "Wrapper around gravity_TG_B2_definitive_force.py",
            "frozen_model_edits": "none",
            "primary_question": "Does quieter/cooler two-node preparation reduce bare breathing enough for body-force and secular impulse/separation signs to agree?",
            "bounded_claims_only": True,
        },
    )

    if args.dry_run:
        write_json(outdir / "RUN_COMPLETE.json", {"status": "DRY_RUN_ONLY", "outdir": str(outdir)})
        print(json.dumps({"status": "DRY_RUN_ONLY", "outdir": str(outdir)}, indent=2))
        return

    pf = preflight()
    write_json(outdir / "gpu_preflight.json", pf)
    if not pf["gpu_ok"]:
        write_json(outdir / "RUN_FAILED.json", {"status": "GPU_PREFLIGHT_FAILED", **pf})
        raise SystemExit("GPU preflight failed")

    all_metrics: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for row in matrix:
        subdir = outdir / row["run_id"]
        cmd = [
            args.python,
            "-u",
            str(ROOT / "jax_scout" / "gravity_TG_B2_definitive_force.py"),
            "--out",
            str(subdir),
            "--seps",
            args.seps,
            "--T",
            str(args.T),
            "--N",
            str(args.N),
            "--L",
            str(args.L),
            "--dt",
            str(args.dt),
            "--sample-dt",
            str(args.sample_dt),
            "--cool-T",
            str(row["cool_T"]),
        ]
        log_path = outdir / f"{row['run_id']}.log"
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - started
        run_rows.append({"run_id": row["run_id"], "returncode": proc.returncode, "elapsed_s": elapsed, "log": str(log_path), "subdir": str(subdir), "command": " ".join(cmd)})
        if proc.returncode != 0:
            all_metrics.append({"subrun": row["run_id"], "status": "SUBPROCESS_FAILED", "returncode": proc.returncode})
            continue
        metrics = analyze_subrun(subdir, args.discard)
        for metric in metrics:
            metric["cool_T"] = row["cool_T"]
        all_metrics.extend(metrics)
        write_csv(outdir / "cooled_pair_metrics_partial.csv", all_metrics)

    write_csv(outdir / "run_manifest.csv", run_rows)
    write_csv(outdir / "cooled_pair_metrics.csv", all_metrics)
    valid = [row for row in all_metrics if row.get("status") == "ANALYZED"]
    body_ok = valid and all(row["body_force_direction"] == "ATTRACT" and row["body_force_sign_reverses"] for row in valid)
    secular_matches = valid and all(row["P_loop_direction"] == "INWARD" and row["sep_loop_direction"] == "CLOSING" for row in valid)
    verdict = "TG_B2_COOLED_PAIR_BODY_FORCE_AND_SECULAR_SIGN_MATCH" if body_ok and secular_matches else "TG_B2_COOLED_PAIR_SECULAR_SIGN_UNRESOLVED"
    write_json(
        outdir / "secular_force_assessment.json",
        {
            "verdict": verdict,
            "body_force_all_attractive": bool(body_ok),
            "secular_impulse_and_separation_match": bool(secular_matches),
            "note": "A negative/partial verdict does not undo TG-R body-force robustness; it only reports the secular proxy state.",
        },
    )
    (outdir / "TECHNICAL_HANDOFF.md").write_text(
        "# TG-B2 Cooled Pair Secular Force Handoff\n\n"
        f"Verdict: `{verdict}`.\n\n"
        "This wrapper reused `gravity_TG_B2_definitive_force.py` for all GPU field evolution and added secular post-processing only.\n",
        encoding="utf-8",
    )
    (outdir / "OPEN_QUESTIONS.md").write_text(
        "# Open Questions\n\n"
        "- Did cooling materially reduce feedback-off breathing?\n"
        "- Do loop impulse and separation signs agree with the robust live-field body-force sign?\n",
        encoding="utf-8",
    )
    (outdir / "git_state_after.txt").write_text(git_state(), encoding="utf-8")
    write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
    write_json(outdir / "RUN_COMPLETE.json", {"status": verdict, "outdir": str(outdir)})
    print(json.dumps({"status": verdict, "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
