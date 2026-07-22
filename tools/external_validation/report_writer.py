from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import relpath, write_text


def metric_report(metric_id: str, title: str, result: dict[str, Any], rows: list[dict[str, Any]], warnings: list[str]) -> str:
    lines = [
        f"# {title}",
        "",
        "Status: provisional until Claude review.",
        "",
        f"- Metric ID: `{metric_id}`",
        f"- Status: `{result.get('status')}`",
        f"- Match-level candidate: `{result.get('match_level_candidate')}`",
        f"- Validation depth: `{result.get('validation_depth')}`",
        f"- Behavioural result: `{result.get('behavioural_result')}`",
        f"- Formula target: `{result.get('formula_target')}`",
        f"- No new simulation: `{result.get('no_new_simulation')}`",
        "",
        "## Fit Parameters",
        "",
        "```json",
        _compact_json(result.get("fit_parameters", {})),
        "```",
        "",
        "## Fit Quality",
        "",
        "```json",
        _compact_json(result.get("fit_quality", {})),
        "```",
        "",
        "## Rows",
        "",
    ]
    if rows:
        keys = list(rows[0].keys())
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        for row in rows[:40]:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    else:
        lines.append("No summary rows were produced.")
    lines += ["", "## Warnings", ""]
    lines += [f"- {w}" for w in warnings] if warnings else ["- None"]
    lines += [
        "",
        "Guardrail: this report is a first-pass metric preparation artifact, not a verdict change or a physical-correspondence claim.",
    ]
    return "\n".join(lines) + "\n"


def write_metric_outputs(out_dir: str | Path, result: dict[str, Any], rows: list[dict[str, Any]], report: str, warnings: list[str]) -> None:
    from .io_utils import write_csv, write_json

    out = Path(out_dir)
    write_json(out / "result.json", result)
    write_csv(out / "summary.csv", rows)
    write_text(out / "report.md", report)
    write_text(out / "read_files.txt", "\n".join(result.get("files_read", [])) + ("\n" if result.get("files_read") else ""))
    write_text(out / "warnings.txt", "\n".join(warnings) + ("\n" if warnings else ""))


def suite_report(output_root: str | Path, metric_results: list[dict[str, Any]], commands: list[str], protected_diff: str, decision: str) -> str:
    ran = [r for r in metric_results if r.get("status") == "RAN"]
    skipped = [r for r in metric_results if r.get("status") == "SKIPPED_WITH_REASON"]
    failed = [r for r in metric_results if r.get("status") == "FAILED"]
    files = sorted({f for r in metric_results for f in r.get("files_read", [])})
    lines = [
        "# External Validation Automation Report",
        "",
        "Status: provisional until Claude review.",
        "",
        f"Final decision: `{decision}`",
        "",
        "## Summary",
        "",
        f"- Metrics run successfully: {len(ran)}",
        f"- Metrics skipped with reason: {len(skipped)}",
        f"- Metrics failed: {len(failed)}",
        "- No simulations run.",
        "- No production physics changed.",
        "- No verdicts changed.",
        "- No physical correspondence claims added.",
        "",
        "## Metric Outcomes",
        "",
        "| Metric | Status | Depth | Behavioural result | Match-level candidate | Main caveat |",
        "|---|---|---|---|---|---|",
    ]
    for r in metric_results:
        caveats = r.get("caveats") or [""]
        lines.append(
            f"| `{r.get('metric_id')}` | `{r.get('status')}` | `{r.get('validation_depth')}` | "
            f"`{r.get('behavioural_result')}` | `{r.get('match_level_candidate')}` | {caveats[0]} |"
        )
    lines += [
        "",
        "## Files Read",
        "",
    ]
    lines += [f"- `{f}`" for f in files] if files else ["- None"]
    lines += [
        "",
        "## Commands Run",
        "",
    ]
    lines += [f"- `{cmd}`" for cmd in commands] if commands else ["- Not recorded"]
    lines += [
        "",
        "## Protected Diff Status",
        "",
        "Expected protected diff is empty.",
        "",
        "```text",
        protected_diff.strip() or "<empty>",
        "```",
        "",
        "## Output Root",
        "",
        f"`{relpath(output_root)}`",
        "",
        "## Guardrails",
        "",
        "All results are first-pass numerical summaries for external legibility. They do not validate IRER experimentally, do not change project verdicts, and do not claim matter, gravity, or unification.",
    ]
    return "\n".join(lines) + "\n"


def _compact_json(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)
