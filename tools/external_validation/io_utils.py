from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_STATUSES = {"RAN", "RAN_WEAK", "SKIPPED_WITH_REASON", "FAILED"}
FINAL_DECISIONS = {
    "VALIDATION_HARNESS_READY_WITH_RESULTS",
    "VALIDATION_HARNESS_READY_WITH_GAPS",
    "VALIDATION_HARNESS_BLOCKED",
}
BEHAVIOURAL_DECISIONS = {
    "BEHAVIOURAL_VALIDATION_READY_FOR_CLAUDE_REVIEW",
    "BEHAVIOURAL_VALIDATION_READY_WITH_GAPS",
    "BEHAVIOURAL_VALIDATION_BLOCKED",
}
VALIDATION_DEPTHS = {
    "FORMALISM_ONLY",
    "INTERNAL_DATA_ANALYTIC_FIT",
    "INTERNAL_DATA_BEHAVIOUR_CHECK",
    "EXTERNAL_DATA_CORRELATION",
    "PLACEHOLDER_OR_WEAK",
}
PROTECTED_FILES = [
    "solver/core.py",
    "solver/run.py",
    "worker_cupy.py",
    "aste_hunter.py",
    "validation_pipeline.py",
    "config_utils.py",
    "gravity/unified_omega.py",
    "jax_scout/physics.py",
    "jax_scout/phase_d_c2_transport.py",
    "jax_scout/phase_d_c2_soliton_scout.py",
    "jax_scout/phase_d_c2_2_loss_source.py",
    "metrics/tensor_validation.py",
    "metrics/collapse_dynamics.py",
]


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def relpath(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


@dataclass
class FileTracker:
    files_read: list[str] = field(default_factory=list)

    def record(self, path: str | Path) -> None:
        rp = relpath(path)
        if rp not in self.files_read:
            self.files_read.append(rp)

    def read_json(self, path: str | Path) -> Any:
        p = repo_path(path)
        self.record(p)
        return json.loads(p.read_text(encoding="utf-8"))

    def read_csv(self, path: str | Path) -> list[dict[str, str]]:
        p = repo_path(path)
        self.record(p)
        with p.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def load_npz_arrays(self, path: str | Path) -> dict[str, np.ndarray]:
        p = repo_path(path)
        self.record(p)
        with np.load(p, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    default_path = Path(__file__).with_name("validation_config.yaml")
    p = repo_path(path) if path else default_path
    if not p.exists():
        return default_config()
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else default_config()
    except Exception:
        return _parse_simple_yaml(text) or default_config()


def default_config() -> dict[str, Any]:
    return {
        "output_root": "docs/external_validation/generated_metrics",
        "metrics": {
            "v1": {"paths": ["sweep_runs/C29_ROBUST", "sweep_runs/C29_VALIDATE", "sweep_runs/C3_TWOQBALL_FULL"]},
            "v2": {"paths": ["sweep_runs/C27_REDERIVE"]},
            "v3": {"paths": ["sweep_runs/C3_EXACT_VK2"]},
            "v5": {"paths": ["sweep_runs/C3_COLLISION_LADDER_FULL", "sweep_runs/C3_ANTIPHASE_LADDER", "sweep_runs/C3_PHASE_pi2", "sweep_runs/C3_PHASE_3pi4", "sweep_runs/C3_PHASE_7pi8"]},
            "v6": {"paths": ["sweep_runs/C23_N96", "sweep_runs/C24_LOCAL_N96", "sweep_runs/C27_REDERIVE", "sweep_runs/PHASE_D_C2_6_CODEX_AUDIT_20260709", "sweep_runs/PHASE_D_CODEX_REPRODUCTION_20260709_233433"]},
            "v7": {"paths": ["sweep_runs/GRAVITY_AD_bg0", "sweep_runs/GRAVITY_DESAT_PILOT", "sweep_runs/PHASE_D_CODEX_REPRODUCTION_20260709_233433/gravity_geometry"]},
        },
    }


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Tiny parser for this repository's fixed config shape; avoids adding a YAML dependency."""
    out: dict[str, Any] = {"metrics": {}}
    current_metric: str | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if not raw.startswith(" ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            if key == "output_root":
                out[key] = value.strip()
            continue
        if raw.startswith("  ") and stripped.endswith(":") and not raw.startswith("    "):
            key = stripped[:-1]
            if key != "paths":
                current_metric = key
                out["metrics"].setdefault(key, {"paths": []})
            continue
        if stripped.startswith("- ") and current_metric:
            out["metrics"][current_metric].setdefault("paths", []).append(stripped[2:].strip())
    return out


def discover_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def protected_diff() -> str:
    try:
        return subprocess.check_output(["git", "diff", "--", *PROTECTED_FILES], cwd=REPO_ROOT, text=True)
    except Exception as exc:
        return f"PROTECTED_DIFF_ERROR: {exc}"


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or ["empty"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_safe(row.get(k, "")) for k in fieldnames})


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def result_template(metric_id: str, formula_target: str) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "status": "SKIPPED_WITH_REASON",
        "data_sources": [],
        "files_read": [],
        "formula_target": formula_target,
        "fit_parameters": {},
        "fit_quality": {},
        "caveats": [],
        "match_level_candidate": "UNVERIFIED",
        "validation_depth": "PLACEHOLDER_OR_WEAK",
        "behavioural_result": "INCONCLUSIVE",
        "no_new_simulation": True,
        "provisional_until_claude_review": True,
        "created_utc": now_iso(),
    }


def metric_paths(config: dict[str, Any], metric_id: str, override: str | None = None) -> list[Path]:
    if override:
        return [repo_path(override)]
    paths = config.get("metrics", {}).get(metric_id, {}).get("paths", [])
    return [repo_path(p) for p in paths]


def ensure_empty_guardrail_text(text: str) -> list[str]:
    forbidden = [
        "IRER is validated experimentally",
        "gravity is shown",
        "matter is shown",
        "unification is shown",
        "external dataset proves IRER",
    ]
    return [phrase for phrase in forbidden if phrase in text]
