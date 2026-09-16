"""Gravity-D load-capacity / yield-map scout.

This is an instantaneous-force atlas around the already characterized spatial
effective-medium operator. It asks whether force response saturates, turns
over, or becomes numerically stiff as coefficient deformation strengthens.

No production geometry is touched. Full field evolution is optional and off by
default; the default run uses the exact instantaneous operator-force contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_scout import gravity_D_dynamics_characterization_gpu as gd  # noqa: E402


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


def build_matrix(kind: str) -> list[dict[str, object]]:
    nmins = [1.0, 0.95, 0.90, 0.80, 0.65, 0.50, 0.35, 0.25] if kind == "bounded" else [1.0, 0.80, 0.50, 0.35]
    widths = [0.75, 1.0, 1.5] if kind == "bounded" else [1.0]
    distances = [2.0, 3.0, 4.0, 5.0] if kind == "bounded" else [4.0]
    rows = []
    base = gd.base_cfg()
    for nmin in nmins:
        for width in widths:
            for radius in distances:
                cfg = gd.with_position({**base, "N_min": nmin, "probe_width": width, "run_id": "tmp"}, radius)
                cfg["run_id"] = f"yield_n{nmin:g}_w{width:g}_r{radius:g}".replace(".", "p")
                cfg["stage"] = "YIELD"
                rows.append(cfg)
    return rows


def classify(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["probe_width"]), str(row["radial_distance"])), []).append(row)
    out = []
    for (width, radius), vals in groups.items():
        vals = sorted(vals, key=lambda row: float(row["N_min"]), reverse=True)
        forces = [abs(float(row["radial_force"])) for row in vals]
        eps = [1.0 - float(row["N_min"]) for row in vals]
        ratios = [forces[i] / max(eps[i], 1e-12) for i in range(len(vals))]
        max_idx = max(range(len(vals)), key=lambda i: forces[i])
        turnover = max_idx < len(vals) - 1 and forces[-1] < 0.8 * forces[max_idx]
        out.append(
            {
                "probe_width": width,
                "radial_distance": radius,
                "max_force": max(forces),
                "max_force_N_min": vals[max_idx]["N_min"],
                "strongest_force": forces[-1],
                "weak_field_force_over_epsilon_mean": sum(ratios[1:4]) / max(len(ratios[1:4]), 1),
                "turnover_or_saturation_candidate": bool(turnover),
                "classification": "YIELD_OR_SATURATION_CANDIDATE" if turnover else "MONOTONE_OR_UNRESOLVED",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "sweep_runs" / f"GRAVITY_D_LOAD_CAPACITY_YIELD_GPU_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--matrix", choices=["pilot", "bounded"], default="pilot")
    args = parser.parse_args()

    outdir = Path(args.out)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    env = gd.save_environment(outdir, command)
    matrix = build_matrix(args.matrix)
    write_json(outdir / "preregistered_matrix.json", matrix)
    rows = []
    manifest = []
    for cfg in matrix:
        print(f"[yield-map] {cfg['run_id']}", flush=True)
        row = gd.run_instantaneous(cfg)
        rows.append(row)
        manifest.append({"run_id": row["run_id"], "config_hash": row["config_hash"], "status": "completed", "gpu_device": row["gpu_device"]})
    cls = classify(rows)
    write_csv(outdir / "instantaneous_force_atlas.csv", rows)
    write_csv(outdir / "yield_classification.csv", cls)
    write_csv(outdir / "run_manifest.csv", manifest)
    verdict = "GRAVITY_D_LOAD_CAPACITY_YIELD_CANDIDATES_FOUND" if any(row["classification"] == "YIELD_OR_SATURATION_CANDIDATE" for row in cls) else "GRAVITY_D_LOAD_CAPACITY_MONOTONE_OR_INCONCLUSIVE"
    write_json(outdir / "yield_summary.json", {"verdict": verdict, "environment": env})
    (outdir / "TECHNICAL_HANDOFF.md").write_text(
        "# Gravity-D Load Capacity / Yield Map\n\n"
        f"Verdict: `{verdict}`.\n\n"
        "This is an instantaneous-force atlas. It does not promote gravity, universal free fall, or a production claim.\n",
        encoding="utf-8",
    )
    (outdir / "OPEN_QUESTIONS.md").write_text("# Open Questions\n\n- If saturation appears, does it survive short trajectory and refinement checks?\n", encoding="utf-8")
    write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
    write_json(outdir / "RUN_COMPLETE.json", {"status": verdict, "outdir": str(outdir)})
    print(json.dumps({"status": verdict, "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
