"""G1-derived clock migration characterization.

This script reuses the G1 supplied-objective-lapse clock machinery, but changes
the question: not "is the clock calibrated?", rather "does the clock mode
migrate toward the source in a reproducible way under trap/mass/width changes?"

Sparse eigen solves are CPU work. Optional time-domain rows use JAX GPU via the
G1 runner functions. No production or TG feedback model is touched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_scout import gravity_G1_clock_calibration_gpu as g1  # noqa: E402


from jax_scout.provenance import write_json  # noqa: E402  (shared: stamps run metadata)

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
    return [{"path": str(p.relative_to(outdir)).replace("\\", "/"), "sha256": sha256(p)} for p in sorted(outdir.rglob("*")) if p.is_file() and p.name != "artifact_hashes.csv"]


def cfg(run_id: str, case: str, pos: tuple[float, float, float], nmin: float, mass: float, width: float, trap_k: float, grid: int = 64) -> dict[str, Any]:
    out = g1.base_cfg(run_id, grid, 30.0, 0.001, case, pos, nmin, mass, width, 1.0, "trapped_eigenmode")
    out["trap_k"] = trap_k
    out["T"] = 2.0
    return out


def build_matrix(kind: str) -> list[dict[str, Any]]:
    near = (1.2, 0.0, 0.0)
    far = (4.0, 0.0, 0.0)
    rows: list[dict[str, Any]] = []
    traps = [0.03, 0.05, 0.10] if kind == "pilot" else [0.02, 0.03, 0.05, 0.10, 0.20]
    masses = [12.0] if kind == "pilot" else [8.0, 12.0, 20.0]
    widths = [1.5] if kind == "pilot" else [1.0, 1.5, 2.0]
    for trap in traps:
        for case, pos in [("flat", near), ("near", near), ("far", far)]:
            rows.append(cfg(f"trap{trap:g}_{case}_m12_w15", case, pos, 0.5, 12.0, 1.5, trap))
    for mass in masses:
        rows.append(cfg(f"mass{mass:g}_near_trap005", "near", near, 0.5, mass, 1.5, 0.05))
    for width in widths:
        rows.append(cfg(f"width{width:g}_near_trap005", "near", near, 0.5, 12.0, width, 0.05))
    if kind != "pilot":
        for grid in [96]:
            rows.append(cfg(f"refine_N{grid}_near_trap005", "near", near, 0.5, 12.0, 1.5, 0.05, grid=grid))
            rows[-1]["dt"] = 0.0005
    seen = set()
    unique = []
    for row in rows:
        if row["run_id"] not in seen:
            seen.add(row["run_id"])
            unique.append(row)
    return unique


def migration_row(erow: dict[str, Any]) -> dict[str, Any]:
    intended = np.asarray(erow["clock_position"], dtype=float)
    mode_com = np.asarray(erow["mode_com"], dtype=float)
    displacement = mode_com - intended
    radius = float(np.linalg.norm(intended))
    toward = -intended / max(radius, 1e-30)
    toward_projection = float(np.dot(displacement, toward))
    transverse = float(np.linalg.norm(displacement - toward_projection * toward))
    return {
        "run_id": erow["run_id"],
        "field_case": erow["field_case"],
        "grid": erow["grid"],
        "N_min": erow["N_min"],
        "mass": erow["mass"],
        "clock_width": erow["clock_width"],
        "trap_k": erow.get("trap_k", ""),
        "intended_x": intended[0],
        "mode_com_x": mode_com[0],
        "migration_distance": float(np.linalg.norm(displacement)),
        "toward_source_projection": toward_projection,
        "transverse_migration": transverse,
        "omega_eigen": erow["omega_eigen"],
        "local_N_center": erow["local_N_center"],
        "operator_symmetry_residual": erow["operator_symmetry_residual"],
        "migration_direction": "TOWARD_SOURCE" if toward_projection > 0 else "AWAY_OR_NULL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "sweep_runs" / f"TG_CLOCK_MIGRATION_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--matrix", choices=["pilot", "bounded"], default="pilot")
    parser.add_argument("--time-domain", choices=["none", "selected", "all"], default="selected")
    args = parser.parse_args()

    outdir = Path(args.out)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories").mkdir(exist_ok=True)
    (outdir / "git_state_before.txt").write_text(g1.git_state(), encoding="utf-8")
    pf = g1.preflight()
    write_json(outdir / "gpu_preflight.json", pf)
    write_json(outdir / "environment_versions.json", g1.environment_record(pf))
    matrix = build_matrix(args.matrix)
    write_json(outdir / "preregistered_matrix.json", matrix)

    eigen_rows = []
    time_rows = []
    migration_rows = []
    for row in matrix:
        print(f"[clock-migration] {row['run_id']}", flush=True)
        erow, mode, Nt, V = g1.solve_eigen(row)
        erow["trap_k"] = row["trap_k"]
        eigen_rows.append(erow)
        migration_rows.append(migration_row(erow))
        run_td = args.time_domain == "all" or (args.time_domain == "selected" and row["run_id"] in {"trap0.05_flat_m12_w15", "trap0.05_near_m12_w15", "trap0.05_far_m12_w15", "trap0.1_near_m12_w15"})
        if run_td:
            time_rows.append(g1.run_time_domain(row, mode, Nt, V, outdir))

    write_csv(outdir / "eigenfrequency_metrics.csv", eigen_rows)
    write_csv(outdir / "time_domain_clock_metrics.csv", time_rows)
    write_csv(outdir / "clock_migration_metrics.csv", migration_rows)
    near_rows = [row for row in migration_rows if row["field_case"] == "near"]
    flat_rows = [row for row in migration_rows if row["field_case"] == "flat"]
    toward_count = sum(row["migration_direction"] == "TOWARD_SOURCE" for row in near_rows)
    flat_max = max((abs(float(row["toward_source_projection"])) for row in flat_rows), default=0.0)
    verdict = "TG_CLOCK_MIGRATION_RESPONSE_DETECTED" if near_rows and toward_count >= max(1, len(near_rows) // 2) and flat_max < max(abs(float(row["toward_source_projection"])) for row in near_rows) else "TG_CLOCK_MIGRATION_CHARACTERIZATION_INCONCLUSIVE"
    write_json(outdir / "migration_summary.json", {"verdict": verdict, "near_toward_count": toward_count, "near_rows": len(near_rows), "flat_max_abs_projection": flat_max})
    (outdir / "TECHNICAL_HANDOFF.md").write_text(
        "# TG Clock Migration Characterization\n\n"
        f"Verdict: `{verdict}`.\n\n"
        "This is a clock-instrument response characterization. It does not restore G1 clock calibration or claim objective time dilation.\n",
        encoding="utf-8",
    )
    (outdir / "OPEN_QUESTIONS.md").write_text("# Open Questions\n\n- Does migration vanish in stronger traps or grow with temporal gradient?\n", encoding="utf-8")
    (outdir / "git_state_after.txt").write_text(g1.git_state(), encoding="utf-8")
    write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
    write_json(outdir / "RUN_COMPLETE.json", {"status": verdict, "outdir": str(outdir)})
    print(json.dumps({"status": verdict, "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
