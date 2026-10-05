from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import repo_path, relpath, write_json, write_text


DEPTH_ORDER = {
    "FORMALISM_ONLY": 0,
    "PLACEHOLDER_OR_WEAK": 0,
    "INTERNAL_DATA_BEHAVIOUR_CHECK": 1,
    "INTERNAL_DATA_ANALYTIC_FIT": 2,
    "EXTERNAL_DATA_CORRELATION": 3,
}


def classify_existing_result(result: dict[str, Any]) -> str:
    depth = result.get("validation_depth")
    if depth:
        return str(depth)
    if result.get("status") != "RAN":
        return "PLACEHOLDER_OR_WEAK"
    if result.get("fit_parameters") or result.get("fit_quality"):
        return "INTERNAL_DATA_BEHAVIOUR_CHECK"
    return "FORMALISM_ONLY"


def audit_generated_metrics(root: str | Path = "docs/external_validation/generated_metrics") -> list[dict[str, Any]]:
    base = repo_path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(base.glob("v*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        metric_dir = path.parent
        read_files = []
        read_path = metric_dir / "read_files.txt"
        if read_path.exists():
            read_files = [line.strip() for line in read_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        numeric_fields = sorted((result.get("fit_quality") or {}).keys())
        depth = classify_existing_result(result)
        rows.append(
            {
                "metric_id": result.get("metric_id"),
                "status": result.get("status"),
                "validation_depth": depth,
                "behavioural_result": result.get("behavioural_result", "INCONCLUSIVE"),
                "files_read_count": len(read_files),
                "numeric_fields_used": ", ".join(numeric_fields),
                "claude_reviewable": depth not in {"FORMALISM_ONLY", "PLACEHOLDER_OR_WEAK"} and len(read_files) > 0,
                "result_path": relpath(path),
            }
        )
    return rows


def write_depth_audit(root: str | Path = "docs/external_validation/generated_metrics") -> list[dict[str, Any]]:
    rows = audit_generated_metrics(root)
    lines = [
        "# Harness Depth Audit",
        "",
        "Status: provisional until Claude review.",
        "",
        "This audit separates formalism alignment from behavioural comparison. IRER formulations remain primary; external models are comparison instruments.",
        "",
        "| Metric | Status | Validation depth | Behavioural result | Files read | Claude-reviewable? | Numeric fields used |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['metric_id']}` | `{row['status']}` | `{row['validation_depth']}` | "
            f"`{row['behavioural_result']}` | {row['files_read_count']} | {row['claude_reviewable']} | "
            f"{row['numeric_fields_used']} |"
        )
    weak = [r for r in rows if r["validation_depth"] in {"FORMALISM_ONLY", "PLACEHOLDER_OR_WEAK"}]
    behavioural = [r for r in rows if r["validation_depth"].startswith("INTERNAL_DATA")]
    external = [r for r in rows if r["validation_depth"] == "EXTERNAL_DATA_CORRELATION"]
    lines += [
        "",
        "## Summary",
        "",
        f"- Behavioural/internal-data metrics: {len(behavioural)}",
        f"- External-data correlations: {len(external)}",
        f"- Formalism-only or weak metrics: {len(weak)}",
        "",
        "No metric is treated as physical correspondence. Metrics marked weak should not be used as scientific results.",
    ]
    write_text("docs/external_validation/HARNESS_DEPTH_AUDIT.md", "\n".join(lines) + "\n")
    write_json(repo_path(root) / "depth_audit.json", {"rows": rows})
    return rows

