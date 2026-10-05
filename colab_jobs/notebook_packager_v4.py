"""
Colab Capsule Notebook Packager v4.

Builds a deterministic, self-extracting notebook from an explicit job manifest.
This project-local copy is the active Quantule Mapper "shop lane" packager:
plan a targeted run, package a compact notebook, upload/run it manually in
Google Colab, copy the result archive from Drive into colab_jobs/results, then
resume the normal local review workflow.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
from typing import Any


PACKAGER_VERSION = "4.6"
DEFAULT_JOB_ROOT = "/content/qm_job"
DEFAULT_MAX_FILE_BYTES = 1_500_000
DEFAULT_MAX_TOTAL_BYTES = 10_000_000
DEFAULT_MAX_NOTEBOOK_BYTES = 25_000_000
DEFAULT_IDEAL_RESULT_BYTES = 1_000_000_000
DEFAULT_HARD_RESULT_BYTES = 5_000_000_000
DEFAULT_LOCAL_RESULTS_DIR = r"F:\quantule_mapper\colab_jobs\results"
SUPPORTED_WORKFLOW_MODE = "manual_colab_upload"
CONTROL_PANEL_SCHEMA = "qm-colab-controls/1.0"
RESERVED_TEST_SLOTS = ("T1", "T2", "T3")
TRUSTED_HANDLERS = {
    "runtime_preflight",
    "numeric_smoke_test",
    "persistence_test",
    "run_registered_job",
}

DEFAULT_RESOURCE_PROFILE: dict[str, Any] = {
    "class": "standard",
    "push_level": "balanced",
    "xla_mem_fraction": "0.90",
    "xla_preallocate": "false",
    "autotune_before_j1": True,
    "autotune_target_mb": 1024,
    "autotune_max_seconds": 20,
    "warn_if_gpu_memory_used_fraction_below": 0.15,
    "notes": (
        "Resource profile controls Colab runtime policy and profiling only. "
        "Scientific scale changes such as larger T, grid, rows, or cadence must be explicit "
        "in the run queue and manifest arguments."
    ),
}


SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(r"-----BEGIN[A-Z0-9 ]+PRIVATE KEY-----.*?-----END[A-Z0-9 ]+PRIVATE KEY-----", re.DOTALL),
    ),
    (
        "certificate",
        re.compile(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.DOTALL),
    ),
    (
        "named_secret",
        re.compile(
            r"\b(api_key|secret_key|auth_token|access_token|password|passwd|pwd)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]",
            re.IGNORECASE,
        ),
    ),
]


def load_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit("YAML manifests require PyYAML. Use JSON or install PyYAML.") from exc
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise SystemExit("Manifest root must be an object.")
    return data


def notebook_cell(kind: str, source: str) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": kind,
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bundle_hash(files: list[dict[str, Any]]) -> str:
    payload = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in sorted(files, key=lambda entry: entry["path"])
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def normalized_rel_path(path_value: str) -> Path:
    rel = Path(path_value.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise SystemExit(f"Manifest path must be relative and stay inside the project: {path_value}")
    return rel


def assert_relative_child(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"{label} escaped its allowed root: {path}") from exc


def likely_binary(path: Path, data: bytes) -> bool:
    if b"\x00" in data[:4096]:
        return True
    guess, _ = mimetypes.guess_type(str(path))
    if guess and not (guess.startswith("text/") or guess in {"application/json", "application/xml"}):
        return True
    try:
        data[:4096].decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def secret_findings(rel_path: str, data: bytes) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []

    findings: list[dict[str, Any]] = []
    for name, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            findings.append({"path": rel_path, "line": line_no, "kind": name})
    return findings


def manifest_file_list(manifest: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("source_files", "config_files", "embedded_files"):
        value = manifest.get(key, [])
        if not isinstance(value, list):
            raise SystemExit(f"Manifest field {key!r} must be a list.")
        paths.extend(str(item) for item in value)

    def add_entrypoint(value: Any) -> None:
        if isinstance(value, dict):
            entry_path = value.get("path")
            if entry_path:
                paths.append(str(entry_path))

    def add_stages(value: Any, label: str) -> None:
        if value is None:
            return
        if not isinstance(value, list):
            raise SystemExit(f"Manifest field {label!r} must be a list when present.")
        for idx, stage in enumerate(value):
            if not isinstance(stage, dict):
                raise SystemExit(f"{label}[{idx}] must be an object.")
            add_entrypoint(stage.get("entrypoint"))
            add_entrypoint(stage)

    requirements_colab = manifest.get("requirements_colab")
    if requirements_colab:
        paths.append(str(requirements_colab))

    entrypoint = manifest.get("entrypoint", {})
    add_entrypoint(entrypoint)
    add_entrypoint(manifest.get("simulation_entrypoint"))
    add_entrypoint(manifest.get("analysis_entrypoint"))
    add_stages(manifest.get("stages"), "stages")
    rows = manifest.get("rows", [])
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        raise SystemExit("Manifest field 'rows' must be a list when present.")
    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"rows[{row_idx}] must be an object.")
        add_stages(row.get("stages"), f"rows[{row_idx}].stages")

    seen: set[str] = set()
    ordered: list[str] = []
    for item in paths:
        rel = normalized_rel_path(item).as_posix()
        if rel not in seen:
            seen.add(rel)
            ordered.append(rel)
    return ordered


def workflow_config(manifest: dict[str, Any]) -> dict[str, Any]:
    workflow = manifest.get("workflow", {})
    if workflow is None:
        workflow = {}
    if not isinstance(workflow, dict):
        raise SystemExit("Manifest field 'workflow' must be an object when present.")

    mode = workflow.get("mode", SUPPORTED_WORKFLOW_MODE)
    if mode != SUPPORTED_WORKFLOW_MODE:
        raise SystemExit(
            f"Unsupported workflow.mode {mode!r}; expected {SUPPORTED_WORKFLOW_MODE!r}."
        )

    result_policy = manifest.get("result_policy", {})
    if result_policy is None:
        result_policy = {}
    if not isinstance(result_policy, dict):
        raise SystemExit("Manifest field 'result_policy' must be an object when present.")

    ideal_bytes = int(result_policy.get("ideal_archive_bytes", DEFAULT_IDEAL_RESULT_BYTES))
    hard_bytes = int(result_policy.get("hard_archive_bytes", DEFAULT_HARD_RESULT_BYTES))
    if ideal_bytes <= 0 or hard_bytes <= 0:
        raise SystemExit("Result archive limits must be positive integers.")
    if ideal_bytes > hard_bytes:
        raise SystemExit("result_policy.ideal_archive_bytes must not exceed hard_archive_bytes.")
    if hard_bytes > DEFAULT_HARD_RESULT_BYTES:
        raise SystemExit(
            f"result_policy.hard_archive_bytes exceeds the project hard cap "
            f"({hard_bytes} > {DEFAULT_HARD_RESULT_BYTES})."
        )

    return {
        "mode": mode,
        "agent_plans_run": bool(workflow.get("agent_plans_run", True)),
        "manual_upload_to_colab": bool(workflow.get("manual_upload_to_colab", True)),
        "manual_colab_execution": bool(workflow.get("manual_colab_execution", True)),
        "drive_persistence_required": bool(workflow.get("drive_persistence_required", True)),
        "local_results_dir": workflow.get("local_results_dir", DEFAULT_LOCAL_RESULTS_DIR),
        "review_prompt_after_download": bool(workflow.get("review_prompt_after_download", True)),
        "intended_use": workflow.get(
            "intended_use",
            "Targeted burst-compute runs; not broad hunts or long unattended campaigns.",
        ),
        "analogy": workflow.get(
            "analogy",
            "Sometimes it is quicker to send the car to the shop than fix it yourself.",
        ),
        "result_policy": {
            "ideal_archive_bytes": ideal_bytes,
            "hard_archive_bytes": hard_bytes,
        },
    }


def resource_profile_config(manifest: dict[str, Any]) -> dict[str, Any]:
    profile = dict(DEFAULT_RESOURCE_PROFILE)
    declared = manifest.get("resource_profile", {})
    if declared is None:
        declared = {}
    if not isinstance(declared, dict):
        raise SystemExit("Manifest field 'resource_profile' must be an object when present.")
    profile.update(declared)

    push_level = str(profile.get("push_level", "balanced"))
    if push_level not in {"conservative", "balanced", "aggressive", "max-safe"}:
        raise SystemExit(
            "resource_profile.push_level must be one of: "
            "conservative, balanced, aggressive, max-safe."
        )

    try:
        mem_fraction = float(profile.get("xla_mem_fraction", "0.90"))
    except (TypeError, ValueError) as exc:
        raise SystemExit("resource_profile.xla_mem_fraction must be numeric.") from exc
    if not 0.10 <= mem_fraction <= 0.95:
        raise SystemExit("resource_profile.xla_mem_fraction must be between 0.10 and 0.95.")
    profile["xla_mem_fraction"] = f"{mem_fraction:.2f}"

    preallocate = str(profile.get("xla_preallocate", "false")).lower()
    if preallocate not in {"true", "false"}:
        raise SystemExit("resource_profile.xla_preallocate must be true or false.")
    profile["xla_preallocate"] = preallocate

    profile["autotune_before_j1"] = bool(profile.get("autotune_before_j1", True))
    profile["autotune_target_mb"] = int(profile.get("autotune_target_mb", 1024))
    profile["autotune_max_seconds"] = int(profile.get("autotune_max_seconds", 20))
    if profile["autotune_target_mb"] < 64:
        raise SystemExit("resource_profile.autotune_target_mb must be at least 64.")
    if not 1 <= profile["autotune_max_seconds"] <= 120:
        raise SystemExit("resource_profile.autotune_max_seconds must be between 1 and 120.")

    warn_fraction = float(profile.get("warn_if_gpu_memory_used_fraction_below", 0.15))
    if not 0.0 <= warn_fraction <= 1.0:
        raise SystemExit("resource_profile.warn_if_gpu_memory_used_fraction_below must be between 0 and 1.")
    profile["warn_if_gpu_memory_used_fraction_below"] = warn_fraction

    launch_env = profile.get("launch_env", {})
    if launch_env is None:
        launch_env = {}
    if not isinstance(launch_env, dict):
        raise SystemExit("resource_profile.launch_env must be an object when present.")
    profile["launch_env"] = {str(key): str(value) for key, value in launch_env.items()}
    return profile


def validate_external_inputs_contract(manifest: dict[str, Any]) -> None:
    external_inputs = manifest.get("external_inputs", [])
    if not isinstance(external_inputs, list):
        raise SystemExit("Manifest field 'external_inputs' must be a list when present.")

    for idx, item in enumerate(external_inputs):
        if not isinstance(item, dict):
            raise SystemExit(f"external_inputs[{idx}] must be an object.")
        if not item.get("path"):
            raise SystemExit(f"external_inputs[{idx}] must define path.")
        if item.get("required", True) and not item.get("stage_to"):
            raise SystemExit(f"external_inputs[{idx}] is required and must define stage_to.")

        required_files = item.get("required_files", [])
        if required_files is None:
            required_files = []
        if not isinstance(required_files, list):
            raise SystemExit(f"external_inputs[{idx}].required_files must be a list.")
        if item.get("required", True) and not required_files and not item.get("sha256"):
            raise SystemExit(
                f"external_inputs[{idx}] must define sha256 for a file input or "
                "required_files with sha256 values for a directory input."
            )
        for req_idx, req in enumerate(required_files):
            if not isinstance(req, dict):
                raise SystemExit(f"external_inputs[{idx}].required_files[{req_idx}] must be an object.")
            if not req.get("path"):
                raise SystemExit(f"external_inputs[{idx}].required_files[{req_idx}] must define path.")
            normalized_rel_path(str(req["path"]))
            if item.get("required", True) and not req.get("sha256"):
                raise SystemExit(
                    f"external_inputs[{idx}].required_files[{req_idx}] must define sha256."
                )


def default_control_panel(manifest: dict[str, Any]) -> dict[str, Any]:
    actions: list[dict[str, Any]] = [
        {
            "action_id": "T1",
            "kind": "system_test",
            "label": "Run Preflight",
            "handler": "runtime_preflight",
            "output_channel": "terminal_T1",
            "required": True,
            "fixed": True,
        },
        {
            "action_id": "T2",
            "kind": "system_test",
            "label": "Run Smoke Test",
            "handler": "numeric_smoke_test",
            "output_channel": "terminal_T2",
            "requires_pass": ["T1"],
            "required": True,
            "fixed": True,
        },
        {
            "action_id": "T3",
            "kind": "system_test",
            "label": "Test Persistence",
            "handler": "persistence_test",
            "output_channel": "terminal_T3",
            "requires_pass": ["T1", "T2"],
            "required": True,
            "fixed": True,
        },
    ]

    rows = manifest.get("rows") or [{"row_id": "main"}]
    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("row_id", f"row_{idx}"))
        actions.append(
            {
                "action_id": f"J{idx}",
                "kind": "job",
                "label": f"Run {row_id}",
                "handler": "run_registered_job",
                "job_id": manifest.get("job_id", "colab_capsule_job"),
                "row_id": row_id,
                "output_channel": f"terminal_J{idx}",
                "requires_pass": ["T1", "T2", "T3"],
                "confirmation_required": True,
            }
        )

    return {
        "schema_version": CONTROL_PANEL_SCHEMA,
        "reserved_test_slots": list(RESERVED_TEST_SLOTS),
        "actions": actions,
    }


def control_panel_config(manifest: dict[str, Any]) -> dict[str, Any]:
    control_panel = manifest.get("control_panel") or default_control_panel(manifest)
    if not isinstance(control_panel, dict):
        raise SystemExit("Manifest field 'control_panel' must be an object when present.")
    if control_panel.get("schema_version") != CONTROL_PANEL_SCHEMA:
        raise SystemExit(
            f"control_panel.schema_version must be {CONTROL_PANEL_SCHEMA!r}."
        )
    if control_panel.get("reserved_test_slots") != list(RESERVED_TEST_SLOTS):
        raise SystemExit("control_panel.reserved_test_slots must be ['T1', 'T2', 'T3'].")

    actions = control_panel.get("actions", [])
    if not isinstance(actions, list):
        raise SystemExit("control_panel.actions must be a list.")

    fixed = {
        "T1": {"kind": "system_test", "handler": "runtime_preflight", "output_channel": "terminal_T1"},
        "T2": {"kind": "system_test", "handler": "numeric_smoke_test", "output_channel": "terminal_T2"},
        "T3": {"kind": "system_test", "handler": "persistence_test", "output_channel": "terminal_T3"},
    }
    action_ids: set[str] = set()
    output_channels: set[str] = set()
    row_ids = {str(row.get("row_id", "main")) for row in manifest.get("rows", [{"row_id": "main"}])}

    for action in actions:
        if not isinstance(action, dict):
            raise SystemExit("Each control_panel action must be an object.")
        action_id = str(action.get("action_id", ""))
        output_channel = str(action.get("output_channel", ""))
        handler = str(action.get("handler", ""))
        kind = str(action.get("kind", ""))

        if not action_id:
            raise SystemExit("Every control_panel action must define action_id.")
        if action_id in action_ids:
            raise SystemExit(f"Duplicate control_panel action_id: {action_id}")
        action_ids.add(action_id)

        if not output_channel:
            raise SystemExit(f"Action {action_id} must define output_channel.")
        if output_channel in output_channels:
            raise SystemExit(f"Duplicate control_panel output_channel: {output_channel}")
        output_channels.add(output_channel)

        if handler not in TRUSTED_HANDLERS:
            raise SystemExit(f"Action {action_id} uses unsupported handler: {handler}")

        if action_id in fixed:
            expected = fixed[action_id]
            for key, value in expected.items():
                if action.get(key) != value:
                    raise SystemExit(f"Reserved action {action_id} must keep {key}={value!r}.")
            if not action.get("fixed", False):
                raise SystemExit(f"Reserved action {action_id} must set fixed=true.")
        elif action_id.startswith("T"):
            raise SystemExit("Only T1, T2 and T3 are valid test-slot action IDs.")
        elif not action_id.startswith("J"):
            raise SystemExit(f"Non-test action IDs must begin with J: {action_id}")

        if kind == "job":
            if action.get("handler") != "run_registered_job":
                raise SystemExit(f"Job action {action_id} must use run_registered_job.")
            row_id = str(action.get("row_id", "main"))
            if row_id not in row_ids:
                raise SystemExit(f"Job action {action_id} references undeclared row_id: {row_id}")
        elif kind != "system_test":
            raise SystemExit(f"Action {action_id} has unsupported kind: {kind}")

        requires = action.get("requires_pass", [])
        if requires is None:
            requires = []
        if not isinstance(requires, list):
            raise SystemExit(f"Action {action_id}.requires_pass must be a list.")
        missing = [required for required in requires if str(required) not in action_ids and str(required) not in fixed]
        if missing:
            raise SystemExit(f"Action {action_id} has unknown requires_pass entries: {missing}")

    for slot in RESERVED_TEST_SLOTS:
        if slot not in action_ids:
            raise SystemExit(f"control_panel must include reserved action {slot}.")

    return {
        "schema_version": CONTROL_PANEL_SCHEMA,
        "reserved_test_slots": list(RESERVED_TEST_SLOTS),
        "actions": actions,
        "log_root": f"/content/qm_colab/jobs/{manifest.get('job_id', 'colab_capsule_job')}/run/logs",
        "status_root": f"/content/qm_colab/jobs/{manifest.get('job_id', 'colab_capsule_job')}/run/status",
        "tail_lines": int(control_panel.get("tail_lines", 500)),
    }


class CapsulePackager:
    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        *,
        max_file_bytes: int,
        max_total_bytes: int,
        max_notebook_bytes: int,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.manifest = load_manifest(self.manifest_path)
        self.workflow = workflow_config(self.manifest)
        self.resource_profile = resource_profile_config(self.manifest)
        validate_external_inputs_contract(self.manifest)
        self.control_panel = control_panel_config(self.manifest)
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.max_notebook_bytes = max_notebook_bytes

    def collect_files(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        total_size = 0

        for rel_path in manifest_file_list(self.manifest):
            abs_path = (self.project_root / rel_path).resolve()
            if not abs_path.is_file():
                raise SystemExit(f"Manifest file not found: {rel_path}")
            assert_relative_child(abs_path, self.project_root, f"Manifest file {rel_path}")

            data = abs_path.read_bytes()
            size = len(data)
            total_size += size
            if size > self.max_file_bytes:
                raise SystemExit(
                    f"File exceeds per-file capsule limit: {rel_path} "
                    f"({size} bytes > {self.max_file_bytes} bytes)"
                )
            if total_size > self.max_total_bytes:
                raise SystemExit(
                    f"Capsule exceeds total embedded limit "
                    f"({total_size} bytes > {self.max_total_bytes} bytes)"
                )

            is_binary = likely_binary(abs_path, data)
            if is_binary and rel_path not in set(self.manifest.get("binary_files", [])):
                raise SystemExit(
                    f"Refusing to embed likely binary file {rel_path}. "
                    "Declare it as an external input or add it to binary_files."
                )

            findings.extend(secret_findings(rel_path, data))
            records.append(
                {
                    "path": rel_path,
                    "size_bytes": size,
                    "sha256": sha256_bytes(data),
                    "base64": base64.b64encode(data).decode("ascii"),
                }
            )

        return records, findings

    def source_manifest(self, file_records: list[dict[str, Any]]) -> dict[str, Any]:
        slim_files = [
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in sorted(file_records, key=lambda entry: entry["path"])
        ]
        return {
            "packager_version": PACKAGER_VERSION,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "project_root_name": self.project_root.name,
            "manifest_path": self.manifest_path.name,
            "job_id": self.manifest.get("job_id", "colab_capsule_job"),
            "workflow": self.workflow,
            "resource_profile": self.resource_profile,
            "control_panel": self.control_panel,
            "files": slim_files,
            "bundle_sha256": canonical_bundle_hash(file_records),
        }

    def generate_notebook(self) -> str:
        file_records, findings = self.collect_files()
        if findings:
            print("Suspected secrets found. Packaging aborted; source was not modified.", file=sys.stderr)
            for finding in findings:
                print(
                    f"  {finding['path']}:{finding['line']} ({finding['kind']})",
                    file=sys.stderr,
                )
            raise SystemExit(2)

        source_manifest = self.source_manifest(file_records)
        job_root = self.manifest.get("job_root", DEFAULT_JOB_ROOT)
        entrypoint = self.manifest.get("entrypoint", {})
        rows = self.manifest.get("rows") or []
        has_stages = bool(self.manifest.get("stages")) or any(
            isinstance(row, dict) and row.get("stages") for row in rows
        )
        if (not isinstance(entrypoint, dict) or not entrypoint.get("path")) and not has_stages:
            raise SystemExit("Manifest must provide entrypoint.path or declared stages.")

        cells: list[dict[str, Any]] = []
        title = self.manifest.get("job_name", self.manifest.get("job_id", "Colab Capsule Job"))
        cells.append(
            notebook_cell(
                "markdown",
                f"# {title}\n\n"
                "Self-extracting Colab capsule generated by `notebook_packager_v4.py`.\n\n"
                "Run cells top to bottom. Source files are hash-verified before execution.\n\n"
                "**Workflow:** upload this notebook to Google Colab manually, run it there, "
                "let it archive results to Drive, then download the archive into "
                f"`{self.workflow['local_results_dir']}` for local review.",
            )
        )

        constants = (
            "# Capsule constants\n"
            "import json as _capsule_json\n"
            f"JOB_ROOT = {json.dumps(job_root)}\n"
            f"MANIFEST = _capsule_json.loads({json.dumps(json.dumps(self.manifest, indent=2, sort_keys=True))})\n"
            f"WORKFLOW = _capsule_json.loads({json.dumps(json.dumps(self.workflow, indent=2, sort_keys=True))})\n"
            f"RESOURCE_PROFILE = _capsule_json.loads({json.dumps(json.dumps(self.resource_profile, indent=2, sort_keys=True))})\n"
            f"CONTROL_PANEL = _capsule_json.loads({json.dumps(json.dumps(self.control_panel, indent=2, sort_keys=True))})\n"
            f"SOURCE_MANIFEST = _capsule_json.loads({json.dumps(json.dumps(source_manifest, indent=2, sort_keys=True))})\n"
            f"EMBEDDED_FILES = _capsule_json.loads({json.dumps(json.dumps(file_records, sort_keys=True))})\n"
        )
        cells.append(notebook_cell("code", constants))
        cells.append(notebook_cell("markdown", "## 0. Runtime Setup And Drive Mount"))
        cells.append(notebook_cell("code", self._runtime_setup_cell()))
        cells.append(notebook_cell("markdown", "## 1. Extract And Verify"))
        cells.append(notebook_cell("code", self._extract_cell()))
        cells.append(notebook_cell("markdown", "## Control Panel"))
        cells.append(notebook_cell("markdown", self._control_panel_markdown()))
        cells.append(notebook_cell("code", self._control_panel_cell()))
        cells.append(notebook_cell("markdown", "## 2. Environment Preflight"))
        cells.append(notebook_cell("code", self._preflight_cell()))
        cells.append(notebook_cell("markdown", "## 3. Optional Dependencies"))
        cells.append(notebook_cell("code", self._dependencies_cell()))
        cells.append(notebook_cell("markdown", "## 4. External Inputs And Smoke Test"))
        cells.append(notebook_cell("code", self._external_and_smoke_cell()))
        cells.append(notebook_cell("markdown", "## 5. Run Job"))
        cells.append(notebook_cell("code", self._run_cell()))
        cells.append(notebook_cell("markdown", "## 6. Archive Results"))
        cells.append(notebook_cell("code", self._archive_cell()))

        notebook = {
            "cells": cells,
            "metadata": {
                "accelerator": "GPU",
                "colab": {"gpuType": "A100"},
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        raw = json.dumps(notebook, indent=2)
        size = len(raw.encode("utf-8"))
        if size > self.max_notebook_bytes:
            raise SystemExit(
                f"Generated notebook exceeds size limit "
                f"({size} bytes > {self.max_notebook_bytes} bytes)"
            )
        return raw

    def _control_panel_markdown(self) -> str:
        def md(value: Any) -> str:
            text = str(value)
            return text.replace("|", "\\|").replace("\n", " ")

        rows = [
            "| Slot | Label | Handler | Output | Requires |",
            "| --- | --- | --- | --- | --- |",
        ]
        for action in self.control_panel["actions"]:
            requires = ", ".join(str(item) for item in action.get("requires_pass", [])) or "-"
            rows.append(
                "| {slot} | {label} | `{handler}` | `{output}` | {requires} |".format(
                    slot=md(action["action_id"]),
                    label=md(action.get("label", action["action_id"])),
                    handler=md(action["handler"]),
                    output=md(action["output_channel"]),
                    requires=md(requires),
                )
            )
        return (
            "Use this panel as the notebook's fixed dashboard. Run `T1`, `T2`, and `T3` first; "
            "job buttons remain blocked until their declared prerequisites pass. Full logs are "
            "written under the manifest log root, while the widget only shows a bounded tail.\n\n"
            + "\n".join(rows)
        )

    @staticmethod
    def _runtime_setup_cell() -> str:
        return r'''
import json
import os
from pathlib import Path

def capsule_references_drive(value):
    if isinstance(value, str):
        return value.startswith("/content/drive")
    if isinstance(value, list):
        return any(capsule_references_drive(item) for item in value)
    if isinstance(value, dict):
        return any(capsule_references_drive(item) for item in value.values())
    return False

def capsule_runtime_environment():
    merged = {}
    resource_profile = globals().get("RESOURCE_PROFILE", {})
    if isinstance(resource_profile, dict):
        if resource_profile.get("xla_preallocate") is not None:
            merged["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(resource_profile.get("xla_preallocate")).lower()
        if resource_profile.get("xla_mem_fraction") is not None:
            merged["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(resource_profile.get("xla_mem_fraction"))
        for key, value in resource_profile.get("launch_env", {}).items():
            merged[str(key)] = None if value is None else str(value)
    runtime = MANIFEST.get("runtime_environment", {})
    policy = MANIFEST.get("environment_policy", {})
    for source in (
        runtime.get("env") if isinstance(runtime, dict) else None,
        policy.get("env") if isinstance(policy, dict) else None,
        policy.get("env_overrides") if isinstance(policy, dict) else None,
    ):
        if source:
            if not isinstance(source, dict):
                raise RuntimeError("runtime environment overrides must be objects.")
            merged.update({str(key): None if value is None else str(value) for key, value in source.items()})
    return merged

Path(JOB_ROOT).mkdir(parents=True, exist_ok=True)
runtime_env = capsule_runtime_environment()
for key, value in runtime_env.items():
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
(Path(JOB_ROOT) / "runtime_environment_overrides.json").write_text(
    json.dumps(
        {
            "applied": runtime_env,
            "effective": {key: os.environ.get(key) for key in runtime_env},
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
(Path(JOB_ROOT) / "resource_profile_resolved.json").write_text(
    json.dumps(RESOURCE_PROFILE, indent=2, sort_keys=True),
    encoding="utf-8",
)
if runtime_env:
    print("Applied runtime environment overrides before JAX imports:")
    print(json.dumps({key: os.environ.get(key) for key in runtime_env}, indent=2, sort_keys=True))
else:
    print("No runtime environment overrides declared.")

needs_drive = capsule_references_drive(MANIFEST)
if needs_drive:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("Google Drive mounted at /content/drive")
    except ModuleNotFoundError:
        print("google.colab is unavailable in this runtime; Drive paths will only work in Colab.")
    except Exception as exc:
        raise RuntimeError(f"Drive mount failed before input staging: {exc}") from exc
else:
    print("Manifest does not reference /content/drive; skipping Drive mount.")
print("Resolved Colab resource profile:")
print(json.dumps(RESOURCE_PROFILE, indent=2, sort_keys=True))
'''

    @staticmethod
    def _extract_cell() -> str:
        return r'''
import base64
import hashlib
import json
import os
from pathlib import Path

root = Path(JOB_ROOT)
root.mkdir(parents=True, exist_ok=True)
root_resolved = root.resolve()

def safe_dest(rel_path):
    dest = (root / rel_path).resolve()
    try:
        dest.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to write outside job root: {rel_path}")
    return dest

def safe_existing_child(base, rel_path):
    base_resolved = Path(base).resolve()
    dest = (base_resolved / rel_path).resolve()
    try:
        dest.relative_to(base_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to read outside external input root: {rel_path}") from exc
    return dest

for item in EMBEDDED_FILES:
    dest = safe_dest(item["path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = base64.b64decode(item["base64"])
    dest.write_bytes(data)

(root / "source_manifest.json").write_text(
    json.dumps(SOURCE_MANIFEST, indent=2, sort_keys=True),
    encoding="utf-8",
)
(root / "job_manifest.json").write_text(
    json.dumps(MANIFEST, indent=2, sort_keys=True),
    encoding="utf-8",
)
(root / "workflow_instructions.json").write_text(
    json.dumps(WORKFLOW, indent=2, sort_keys=True),
    encoding="utf-8",
)
(root / "control_panel_registry.json").write_text(
    json.dumps(CONTROL_PANEL, indent=2, sort_keys=True),
    encoding="utf-8",
)

verified = []
for item in SOURCE_MANIFEST["files"]:
    path = safe_dest(item["path"])
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != item["sha256"]:
        raise RuntimeError(f"Hash mismatch for {item['path']}: {digest} != {item['sha256']}")
    verified.append({"path": item["path"], "sha256": digest, "size_bytes": len(data)})

(root / "artifact_hashes_extracted.json").write_text(
    json.dumps({"verified": verified, "bundle_sha256": SOURCE_MANIFEST["bundle_sha256"]}, indent=2),
    encoding="utf-8",
)
os.chdir(root)
print(f"Extracted and verified {len(verified)} files in {root}")
'''

    @staticmethod
    def _control_panel_cell() -> str:
        return r'''
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

try:
    import ipywidgets as widgets
    from IPython.display import display
except Exception as exc:
    widgets = None
    display = None
    print(f"ipywidgets unavailable; use the manual cells below. Details: {exc}")

root = Path(JOB_ROOT)
log_root = Path(CONTROL_PANEL["log_root"])
status_root = Path(CONTROL_PANEL["status_root"])
log_root.mkdir(parents=True, exist_ok=True)
status_root.mkdir(parents=True, exist_ok=True)
tail_lines = int(CONTROL_PANEL.get("tail_lines", 500))
trusted_handlers = {
    "runtime_preflight",
    "numeric_smoke_test",
    "persistence_test",
    "run_registered_job",
}
pass_states = {"PASS", "LOCAL_COMMITTED", "DRIVE_COMMITTED"}
actions = {action["action_id"]: action for action in CONTROL_PANEL["actions"]}
tail_buffers = {action_id: collections.deque(maxlen=tail_lines) for action_id in actions}
output_widgets = {}
status_widgets = {}
button_widgets = {}
job_contexts = {}


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def action_log_path(action_id):
    return log_root / f"{action_id}.log"


def action_status_path(action_id):
    return status_root / f"{action_id}.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status_color(state):
    return {
        "NOT_RUN": "#666",
        "READY": "#2563eb",
        "BLOCKED": "#7c3aed",
        "RUNNING": "#d97706",
        "PASS": "#15803d",
        "FAILED": "#b91c1c",
        "INTERRUPTED": "#b45309",
        "LOCAL_COMMITTED": "#047857",
        "DRIVE_COMMITTED": "#047857",
    }.get(state, "#333")


def status_html(state):
    color = status_color(state)
    return f"<div style='color:{color}; font-weight:700;'>Status: {state}</div>"


def write_status(action_id, state, **payload):
    data = {
        "action_id": action_id,
        "state": state,
        "updated_utc": now_iso(),
        **payload,
    }
    action_status_path(action_id).write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if action_id in status_widgets:
        status_widgets[action_id].value = status_html(state)
        try:
            status_widgets[action_id].layout.border = f"2px solid {status_color(state)}"
            if action_id in output_widgets:
                output_widgets[action_id].layout.border = f"2px solid {status_color(state)}"
        except Exception:
            pass
    return data


def read_status(action_id):
    path = action_status_path(action_id)
    if not path.exists():
        return {"action_id": action_id, "state": "NOT_RUN"}
    return json.loads(path.read_text(encoding="utf-8"))


class ActionContext:
    def __init__(self, action_id, action, button=None, output=None, status=None):
        self.action_id = action_id
        self.action = action
        self.button = button
        self.output_widget = output
        self.status_widget = status
        self.output_channel = action.get("output_channel")
        self.log_file_path = str(action_log_path(action_id))
        self.status_file_path = str(action_status_path(action_id))

    def run(self):
        return run_action(self.action_id)

    def append_stdout(self, text):
        emit(self.action_id, str(text))

    def state(self):
        return read_status(self.action_id).get("state", "NOT_RUN")

    def as_bridge_record(self):
        return {
            "action_id": self.action_id,
            "description": self.action.get("label", self.action_id),
            "kind": self.action.get("kind"),
            "handler": self.action.get("handler"),
            "output_channel": self.output_channel,
            "button_variable": f"button_{self.action_id}",
            "state_label_variable": f"status_{self.action_id}",
            "output_variable": f"output_{self.action_id}",
            "job_context_variable": f"context_{self.action_id}",
            "button_registry_expression": f"button_widgets[{self.action_id!r}]",
            "output_registry_expression": f"output_widgets[{self.action_id!r}]",
            "status_registry_expression": f"status_widgets[{self.action_id!r}]",
            "context_registry_expression": f"job_contexts[{self.action_id!r}]",
            "run_expression": f"run_action({self.action_id!r})",
            "log_file_path": self.log_file_path,
            "status_file_path": self.status_file_path,
        }


def render_tail(action_id):
    widget = output_widgets.get(action_id)
    if widget is None:
        return
    with widget:
        widget.clear_output(wait=True)
        print("".join(tail_buffers[action_id]), end="")


def emit(action_id, text):
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if not text.endswith("\n"):
        text += "\n"
    with action_log_path(action_id).open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(text)
    tail_buffers[action_id].extend(text.splitlines(keepends=True))
    render_tail(action_id)


def set_buttons_disabled(disabled):
    for button in button_widgets.values():
        button.disabled = disabled


def ensure_dependencies(action_id, action):
    blocked = []
    for required in action.get("requires_pass", []):
        state = read_status(str(required)).get("state", "NOT_RUN")
        if state not in pass_states:
            blocked.append(f"{required}={state}")
    if blocked:
        raise RuntimeError("Blocked by unmet dependencies: " + ", ".join(blocked))


def run_capture(cmd):
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        return {"error": repr(exc)}


def parse_gpu_resource_query(text):
    rows = []
    for line in str(text or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 7:
            try:
                rows.append(
                    {
                        "name": parts[0],
                        "memory_total_mb": float(parts[1]),
                        "memory_used_mb": float(parts[2]),
                        "memory_free_mb": float(parts[3]),
                        "utilization_gpu_percent": float(parts[4]),
                        "utilization_memory_percent": float(parts[5]),
                        "temperature_gpu_c": float(parts[6]),
                    }
                )
            except ValueError:
                rows.append({"raw": line})
    return rows


def system_resource_snapshot():
    snapshot = {"disk_content": {}}
    try:
        total, used, free = shutil.disk_usage("/content")
        snapshot["disk_content"] = {"total_bytes": total, "used_bytes": used, "free_bytes": free}
    except Exception as exc:
        snapshot["disk_error"] = repr(exc)
    try:
        meminfo = {}
        for raw in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = raw.split(":", 1)
            meminfo[key] = int(value.strip().split()[0]) * 1024
        snapshot["system_ram"] = {
            "total_bytes": meminfo.get("MemTotal"),
            "available_bytes": meminfo.get("MemAvailable"),
            "used_estimate_bytes": (
                meminfo.get("MemTotal") - meminfo.get("MemAvailable")
                if meminfo.get("MemTotal") and meminfo.get("MemAvailable")
                else None
            ),
        }
    except Exception as exc:
        snapshot["system_ram_error"] = repr(exc)
    return snapshot


def safe_job_dest(rel_path):
    dest = (root / rel_path).resolve()
    try:
        dest.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Refusing path outside job root: {rel_path}") from exc
    return dest


def safe_child(base, rel_path):
    base_resolved = Path(base).resolve()
    dest = (base_resolved / rel_path).resolve()
    try:
        dest.relative_to(base_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Refusing path outside staged input root: {rel_path}") from exc
    return dest


def trusted_runtime_preflight(action_id):
    emit(action_id, "Starting runtime preflight.")
    precision = MANIFEST.get("precision", {})
    preflight = {
        "timestamp_utc": now_iso(),
        "cwd": os.getcwd(),
        "python": sys.version,
        "nvidia_smi": run_capture(["nvidia-smi"]),
        "nvidia_smi_query": run_capture([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]),
        "resource_profile": RESOURCE_PROFILE,
        "host_resources": system_resource_snapshot(),
    }
    preflight["gpu_resources"] = parse_gpu_resource_query(preflight["nvidia_smi_query"].get("stdout", ""))
    import numpy as np
    import jax
    if precision.get("jax_enable_x64", True):
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jaxlib
    try:
        import scipy
        preflight["scipy_version"] = scipy.__version__
    except Exception as exc:
        preflight["scipy_import_error"] = repr(exc)
    devices = jax.devices()
    x = jnp.array([1.0, 2.0], dtype=jnp.float64)
    y = (x * jnp.float64(2.0)).block_until_ready()
    matrix = jnp.eye(8, dtype=jnp.float64)
    smoke = (matrix @ matrix + jnp.fft.fft(jnp.ones(8, dtype=jnp.float64))).block_until_ready()
    preflight.update(
        {
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "numpy_version": np.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in devices],
            "jax_x64_enabled": bool(jax.config.jax_enable_x64),
            "float64_kernel_dtype": str(y.dtype),
            "numerical_gpu_smoke_dtype": str(smoke.dtype),
            "numerical_gpu_smoke_finite": bool(jnp.all(jnp.isfinite(smoke))),
        }
    )
    (root / "gpu_preflight_control_T1.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    emit(action_id, json.dumps(preflight, indent=2, sort_keys=True)[:4000])
    policy = MANIFEST.get("environment_policy", {})
    if policy.get("require_gpu", True) and preflight.get("jax_backend") != "gpu":
        raise RuntimeError("JAX GPU backend is required but was not detected.")
    accepted = policy.get("accepted_gpu_names", ["A100"])
    if accepted:
        haystack = "\n".join(preflight.get("jax_devices", [])) + "\n" + preflight.get("nvidia_smi", {}).get("stdout", "")
        if not any(str(name).lower() in haystack.lower() for name in accepted):
            raise RuntimeError(f"GPU is not in accepted list {accepted}.")
    if precision.get("jax_enable_x64", True):
        if not preflight.get("jax_x64_enabled") or preflight.get("float64_kernel_dtype") != "float64":
            raise RuntimeError("JAX x64 preflight failed.")
    if not preflight.get("numerical_gpu_smoke_finite"):
        raise RuntimeError("Numerical GPU smoke test failed.")
    return {"artifact": str(root / "gpu_preflight_control_T1.json")}


def generic_jax_resource_probe():
    profile = RESOURCE_PROFILE if isinstance(RESOURCE_PROFILE, dict) else {}
    if not profile.get("autotune_before_j1", True):
        return {"enabled": False, "reason": "resource_profile.autotune_before_j1 is false"}
    import math
    import jax
    if MANIFEST.get("precision", {}).get("jax_enable_x64", True):
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    target_mb = max(64, int(profile.get("autotune_target_mb", 1024)))
    max_seconds = max(1, int(profile.get("autotune_max_seconds", 20)))
    dtype_bytes = 8
    # Matmul uses a, b and output arrays. Keep the probe bounded and side-effect free.
    n_target = int(math.sqrt((target_mb * 1024 * 1024) / (dtype_bytes * 3)))
    candidates = sorted({256, 512, 1024, max(256, min(n_target, 4096))})
    started = time.time()
    results = []
    before = parse_gpu_resource_query(run_capture([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]).get("stdout", ""))
    for n in candidates:
        if time.time() - started > max_seconds:
            break
        try:
            a = jnp.ones((n, n), dtype=jnp.float64)
            b = jnp.eye(n, dtype=jnp.float64)
            t0 = time.time()
            c = (a @ b).block_until_ready()
            elapsed = time.time() - t0
            checksum = float(jnp.sum(c[: min(8, n), : min(8, n)]))
            results.append(
                {
                    "n": n,
                    "approx_working_set_mb": round((n * n * dtype_bytes * 3) / (1024 * 1024), 2),
                    "elapsed_seconds": elapsed,
                    "checksum": checksum,
                }
            )
            del a, b, c
        except Exception as exc:
            results.append({"n": n, "error": repr(exc)})
            break
    after = parse_gpu_resource_query(run_capture([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]).get("stdout", ""))
    warning = None
    if after:
        total = after[0].get("memory_total_mb") or 0
        used = after[0].get("memory_used_mb") or 0
        used_fraction = used / total if total else None
        threshold = float(profile.get("warn_if_gpu_memory_used_fraction_below", 0.15))
        if used_fraction is not None and used_fraction < threshold:
            warning = (
                "GPU memory use is low after the generic probe. This can be healthy for compute-bound kernels, "
                "but if utilization is also low, inspect host transfers, Python loops, or scientific scale settings."
            )
    return {
        "enabled": True,
        "profile": profile,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_seconds": time.time() - started,
        "gpu_before": before,
        "gpu_after": after,
        "results": results,
        "warning": warning,
    }


def trusted_numeric_smoke_test(action_id):
    emit(action_id, "Starting capsule smoke test.")
    for row in MANIFEST.get("rows", []) or [{"row_id": "main"}]:
        compile_row_entrypoints(row)
    import jax
    if MANIFEST.get("precision", {}).get("jax_enable_x64", True):
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sample = jnp.arange(16, dtype=jnp.float64).reshape(4, 4)
    result = (sample @ sample.T).block_until_ready()
    smoke_payload = {
        "entrypoints_compiled": True,
        "jax_backend": jax.default_backend(),
        "result_dtype": str(result.dtype),
        "result_finite": bool(jnp.all(jnp.isfinite(result))),
        "sum": float(jnp.sum(result)),
        "resource_probe": generic_jax_resource_probe(),
        "timestamp_utc": now_iso(),
    }
    out = root / "control_panel_smoke_T2.json"
    out.write_text(json.dumps(smoke_payload, indent=2, sort_keys=True), encoding="utf-8")
    emit(action_id, json.dumps(smoke_payload, indent=2, sort_keys=True))
    if not smoke_payload["result_finite"]:
        raise RuntimeError("Smoke calculation produced non-finite output.")
    return {"artifact": str(out)}


def trusted_stage_external_inputs(action_id):
    staged = []
    missing = []
    for item in MANIFEST.get("external_inputs", []):
        path = Path(item["path"])
        required = item.get("required", True)
        if required and not path.exists():
            missing.append(str(path))
            continue
        if not path.exists():
            continue
        stage_to = item.get("stage_to")
        staged_root = path
        if stage_to:
            dest = safe_job_dest(stage_to)
            if path.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(path, dest)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
            staged_root = dest
        verified_files = []
        if path.is_file() and item.get("sha256"):
            expected = item["sha256"]
            if str(expected).startswith("REPLACE_"):
                raise RuntimeError(f"Replace placeholder sha256 for external input {path}.")
            actual = file_sha256(staged_root)
            if actual != expected:
                raise RuntimeError(f"External input hash mismatch for {path}: {actual} != {expected}")
        for required_file in item.get("required_files", []):
            rel_path = required_file["path"]
            target = safe_child(staged_root, rel_path)
            if not target.exists():
                raise FileNotFoundError(f"Missing required staged external file: {target}")
            expected = required_file.get("sha256")
            if not expected or str(expected).startswith("REPLACE_"):
                raise RuntimeError(f"Replace placeholder sha256 for external required file {rel_path}.")
            actual = file_sha256(target)
            if actual != expected:
                raise RuntimeError(f"External required file hash mismatch for {rel_path}: {actual} != {expected}")
            verified_files.append({"path": rel_path, "sha256": actual})
        staged.append({"source_path": str(path), "staged_path": str(staged_root), "verified_files": verified_files})
    if missing:
        raise FileNotFoundError("Missing required external inputs: " + ", ".join(missing))
    out = root / "external_inputs_staged_control.json"
    out.write_text(json.dumps({"external_inputs": staged}, indent=2, sort_keys=True), encoding="utf-8")
    emit(action_id, json.dumps({"external_inputs": staged}, indent=2, sort_keys=True))
    return staged


def trusted_persistence_test(action_id):
    emit(action_id, "Starting persistence/recovery test.")
    drive_dir = MANIFEST.get("drive_results_dir")
    local_test_dir = root / "control_panel_persistence"
    local_test_dir.mkdir(parents=True, exist_ok=True)
    payload = f"control-panel-persistence-test {now_iso()}\n".encode("utf-8")
    local_file = local_test_dir / "persistence_test.txt"
    local_file.write_bytes(payload)
    local_hash = file_sha256(local_file)
    result = {"local_file": str(local_file), "local_sha256": local_hash, "local_committed": True}
    if drive_dir:
        dest_dir = Path(drive_dir) / "_control_panel_tests"
        if str(dest_dir).startswith("/content/drive"):
            try:
                from google.colab import drive
                drive.mount("/content/drive", force_remount=False)
            except ModuleNotFoundError as exc:
                raise RuntimeError("Google Drive persistence test requires Colab.") from exc
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "persistence_test.txt"
        tmp = Path(str(dest) + ".tmp")
        shutil.copy2(local_file, tmp)
        copied_hash = file_sha256(tmp)
        if copied_hash != local_hash:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Persistence copy hash mismatch: {copied_hash} != {local_hash}")
        tmp.replace(dest)
        result.update({"drive_file": str(dest), "drive_sha256": copied_hash, "drive_committed": True})
    out = root / "control_panel_persistence_T3.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    emit(action_id, json.dumps(result, indent=2, sort_keys=True))
    return {"artifact": str(out)}


def row_by_id(row_id):
    rows = MANIFEST.get("rows") or [{"row_id": "main", "arguments": MANIFEST.get("arguments", []), "expected_outputs": MANIFEST.get("expected_outputs", [])}]
    for row in rows:
        if str(row.get("row_id", "main")) == str(row_id):
            return row
    raise KeyError(f"Unknown row_id: {row_id}")


def render_manifest_value(value, row_id, stage_id=None):
    mapping = {
        "job_root": str(root),
        "job_id": str(MANIFEST.get("job_id", "colab_capsule_job")),
        "row_id": str(row_id),
        "stage_id": "" if stage_id is None else str(stage_id),
    }
    if isinstance(value, str):
        rendered = value
        for key, replacement in mapping.items():
            rendered = rendered.replace("{" + key + "}", replacement)
        return rendered
    if isinstance(value, list):
        return [render_manifest_value(item, row_id, stage_id) for item in value]
    if isinstance(value, dict):
        return {key: render_manifest_value(item, row_id, stage_id) for key, item in value.items()}
    return value


def row_stages(row):
    stages = row.get("stages")
    if stages is None:
        stages = MANIFEST.get("stages")
    if stages:
        return stages
    return [
        {
            "stage_id": "main",
            "entrypoint": MANIFEST["entrypoint"],
            "arguments": row.get("arguments", MANIFEST.get("arguments", [])),
            "expected_outputs": row.get("expected_outputs", []),
        }
    ]


def stage_command(stage, row_id, stage_id):
    rendered = render_manifest_value(stage, row_id, stage_id)
    command = rendered.get("command")
    if command:
        if not isinstance(command, list):
            raise RuntimeError(f"Stage {stage_id} command must be a list.")
        command = [str(part) for part in command]
        if command and command[0] in {"python", "python3", "{python}"}:
            command[0] = sys.executable
        return command
    entry = rendered.get("entrypoint") if isinstance(rendered.get("entrypoint"), dict) else rendered
    path = entry.get("path") if isinstance(entry, dict) else None
    if not path:
        raise RuntimeError(f"Stage {stage_id} must define entrypoint.path or command.")
    arguments = rendered.get("arguments")
    if arguments is None and isinstance(entry, dict):
        arguments = entry.get("arguments", [])
    return [sys.executable, "-u", str(path)] + [str(arg) for arg in (arguments or [])]


def stage_entrypoint_path(stage):
    entry = stage.get("entrypoint") if isinstance(stage.get("entrypoint"), dict) else stage
    path = entry.get("path") if isinstance(entry, dict) else None
    return path


def compile_row_entrypoints(row):
    for stage in row_stages(row):
        path = stage_entrypoint_path(stage)
        if path:
            subprocess.check_call([sys.executable, "-m", "py_compile", str(root / str(path))], cwd=root)


def load_ledger():
    ledger_path = root / "completion_ledger.json"
    if ledger_path.exists():
        return json.loads(ledger_path.read_text(encoding="utf-8"))
    return {"completed_rows": [], "failed_rows": [], "rows": []}


def write_ledger(ledger):
    (root / "completion_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")


def checkpoint_row(row_id):
    checkpoint_dir = MANIFEST.get("drive_checkpoint_dir")
    if not checkpoint_dir:
        return None
    dest_dir = Path(checkpoint_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = dest_dir / f"{MANIFEST.get('job_id', 'colab_capsule_job')}_{row_id}_checkpoint.json"
    payload = {
        "row_id": row_id,
        "ledger": load_ledger(),
        "timestamp_utc": now_iso(),
    }
    tmp = Path(str(checkpoint) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(checkpoint)
    return str(checkpoint)


def verify_expected_outputs(expected_outputs):
    missing = []
    for rel_path in expected_outputs:
        if not (root / str(rel_path)).exists():
            missing.append(str(rel_path))
    if missing:
        (root / "MISSING_EXPECTED_OUTPUTS.json").write_text(
            json.dumps({"missing_expected_outputs": missing}, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError("Missing expected outputs: " + ", ".join(missing))


def copy_control_panel_runtime_into_root():
    panel_dir = root / "control_panel_runtime"
    logs_dest = panel_dir / "logs"
    status_dest = panel_dir / "status"
    logs_dest.mkdir(parents=True, exist_ok=True)
    status_dest.mkdir(parents=True, exist_ok=True)
    for path in log_root.glob("*.log"):
        shutil.copy2(path, logs_dest / path.name)
    for path in status_root.glob("*.json"):
        shutil.copy2(path, status_dest / path.name)


def trusted_archive_results(action_id, expected_outputs):
    verify_expected_outputs(expected_outputs)
    copy_control_panel_runtime_into_root()
    job_id = MANIFEST.get("job_id", "colab_capsule_job")
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    archive_path = Path("/content") / f"{job_id}_{stamp}.tar.gz"
    provenance = {
        "job_manifest": MANIFEST,
        "workflow": WORKFLOW,
        "resource_profile": RESOURCE_PROFILE,
        "control_panel": CONTROL_PANEL,
        "source_manifest": SOURCE_MANIFEST,
        "completion_ledger": load_ledger(),
        "generated_utc": now_iso(),
    }
    (root / "capsule_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    integrity_files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "result_integrity_manifest.json":
            integrity_files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    (root / "result_integrity_manifest.json").write_text(
        json.dumps({"files": integrity_files}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(root, arcname=job_id)
    archive_size = archive_path.stat().st_size
    policy = WORKFLOW.get("result_policy", {})
    ideal_bytes = int(policy.get("ideal_archive_bytes", 1000000000))
    hard_bytes = int(policy.get("hard_archive_bytes", 5000000000))
    if archive_size > hard_bytes:
        raise RuntimeError(f"Result archive exceeds hard cap: {archive_size} bytes > {hard_bytes} bytes.")
    report = {
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_size,
        "archive_sha256": file_sha256(archive_path),
        "ideal_archive_bytes": ideal_bytes,
        "hard_archive_bytes": hard_bytes,
        "over_ideal": archive_size > ideal_bytes,
    }
    (root / "archive_size_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    emit(action_id, json.dumps(report, indent=2, sort_keys=True))
    drive_dir = MANIFEST.get("drive_results_dir")
    if drive_dir:
        dest_dir = Path(drive_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / archive_path.name
        tmp = Path(str(dest) + ".tmp")
        shutil.copy2(archive_path, tmp)
        copied_hash = file_sha256(tmp)
        if copied_hash != report["archive_sha256"]:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Drive archive hash mismatch: {copied_hash} != {report['archive_sha256']}")
        tmp.replace(dest)
        sidecar = dest_dir / f"{archive_path.name}.sha256.json"
        sidecar_tmp = Path(str(sidecar) + ".tmp")
        sidecar_tmp.write_text(json.dumps({**report, "drive_path": str(dest), "copied_utc": now_iso()}, indent=2, sort_keys=True), encoding="utf-8")
        sidecar_tmp.replace(sidecar)
        report.update({"drive_path": str(dest), "drive_sidecar": str(sidecar)})
    return report


def trusted_run_registered_job(action_id, action):
    row_id = str(action.get("row_id", "main"))
    row = row_by_id(row_id)
    emit(action_id, f"Starting registered job row {row_id}.")
    trusted_stage_external_inputs(action_id)
    compile_row_entrypoints(row)
    ledger = load_ledger()
    if row_id in ledger.get("completed_rows", []):
        emit(action_id, f"Row {row_id} is already marked complete; skipping execution.")
    else:
        row_marker_dir = root / "row_markers"
        row_marker_dir.mkdir(exist_ok=True)
        stages = row_stages(row)
        row_start = time.time()
        (row_marker_dir / f"{row_id}_STARTED.json").write_text(
            json.dumps(
                {
                    "row_id": row_id,
                    "stages": [
                        str(stage.get("stage_id", f"stage_{idx}"))
                        for idx, stage in enumerate(stages, start=1)
                    ],
                    "started_utc": now_iso(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        stage_records = []
        return_code = 0
        failed_stage = None
        for idx, stage in enumerate(stages, start=1):
            stage_id = str(stage.get("stage_id", f"stage_{idx}"))
            cmd = stage_command(stage, row_id, stage_id)
            emit(action_id, f"Running stage {stage_id}: " + " ".join(cmd))
            stage_start = time.time()
            proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert proc.stdout is not None
            for chunk in iter(proc.stdout.readline, b""):
                emit(action_id, chunk)
            return_code = proc.wait()
            stage_record = {
                "stage_id": stage_id,
                "command": cmd,
                "return_code": return_code,
                "duration_seconds": time.time() - stage_start,
                "finished_utc": now_iso(),
            }
            stage_records.append(stage_record)
            if return_code != 0:
                failed_stage = stage_id
                break
            verify_expected_outputs(stage.get("expected_outputs", []))
        finished = {
            "row_id": row_id,
            "return_code": return_code,
            "duration_seconds": time.time() - row_start,
            "finished_utc": now_iso(),
            "log_path": str(action_log_path(action_id)),
            "log_sha256": file_sha256(action_log_path(action_id)),
            "stages": stage_records,
        }
        ledger.setdefault("rows", []).append(finished)
        if return_code == 0:
            ledger.setdefault("completed_rows", []).append(row_id)
            (row_marker_dir / f"{row_id}_COMPLETE.json").write_text(json.dumps(finished, indent=2), encoding="utf-8")
            write_ledger(ledger)
            checkpoint = checkpoint_row(row_id)
            if checkpoint:
                emit(action_id, f"Wrote row checkpoint: {checkpoint}")
        else:
            ledger.setdefault("failed_rows", []).append(row_id)
            (row_marker_dir / f"{row_id}_FAILED.json").write_text(json.dumps(finished, indent=2), encoding="utf-8")
            write_ledger(ledger)
            raise RuntimeError(f"Row {row_id} failed in stage {failed_stage} with return code {return_code}.")
    expected_outputs = row.get("expected_outputs", MANIFEST.get("expected_outputs", []))
    archive_report = trusted_archive_results(action_id, expected_outputs)
    return {"row_id": row_id, **archive_report}


def run_action(action_id):
    action = actions[action_id]
    set_buttons_disabled(True)
    started = time.time()
    action_log_path(action_id).write_text("", encoding="utf-8")
    tail_buffers[action_id].clear()
    write_status(action_id, "RUNNING", started_utc=now_iso())
    emit(action_id, f"Action {action_id} started with handler {action['handler']}.")
    try:
        ensure_dependencies(action_id, action)
        handler = action["handler"]
        if handler == "runtime_preflight":
            result = trusted_runtime_preflight(action_id)
            final_state = "PASS"
        elif handler == "numeric_smoke_test":
            result = trusted_numeric_smoke_test(action_id)
            final_state = "PASS"
        elif handler == "persistence_test":
            result = trusted_persistence_test(action_id)
            final_state = "PASS"
        elif handler == "run_registered_job":
            result = trusted_run_registered_job(action_id, action)
            final_state = "DRIVE_COMMITTED" if result.get("drive_path") else "LOCAL_COMMITTED"
        else:
            raise RuntimeError(f"Unsupported handler: {handler}")
        emit(action_id, f"Action {action_id} finished: {final_state}")
        write_status(
            action_id,
            final_state,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            finished_utc=now_iso(),
            duration_seconds=time.time() - started,
            result=result,
            log_path=str(action_log_path(action_id)),
            log_sha256=file_sha256(action_log_path(action_id)),
        )
    except Exception as exc:
        tb = traceback.format_exc()
        emit(action_id, "ERROR: " + repr(exc))
        emit(action_id, tb)
        write_status(
            action_id,
            "FAILED",
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            finished_utc=now_iso(),
            duration_seconds=time.time() - started,
            error=repr(exc),
            traceback=tb,
            log_path=str(action_log_path(action_id)),
            log_sha256=file_sha256(action_log_path(action_id)) if action_log_path(action_id).exists() else None,
        )
    finally:
        set_buttons_disabled(False)


def make_button_callback(action_id):
    def _callback(_button):
        run_action(action_id)
    return _callback


def action_button_style(action_id):
    if action_id == "T1":
        return "primary"
    if action_id == "T2":
        return "warning"
    if action_id == "T3":
        return "info"
    return "success"


def build_bridge_manifest():
    records = []
    for action in CONTROL_PANEL["actions"]:
        action_id = action["action_id"]
        context = job_contexts.get(action_id) or ActionContext(action_id, action)
        records.append(context.as_bridge_record())
    return {
        "schema_version": "qm-colab-ui-bridge/1.0",
        "job_id": MANIFEST.get("job_id", "colab_capsule_job"),
        "control_panel_schema": CONTROL_PANEL.get("schema_version"),
        "log_base_directory": str(log_root),
        "status_base_directory": str(status_root),
        "button_registry": "button_widgets",
        "output_registry": "output_widgets",
        "status_registry": "status_widgets",
        "context_registry": "job_contexts",
        "runner_function": "run_action",
        "safety_boundary": "Dashboard states report infrastructure status only, not scientific validation.",
        "actions": records,
    }


if widgets is None:
    for action in CONTROL_PANEL["actions"]:
        action_id = action["action_id"]
        context = ActionContext(action_id, action)
        job_contexts[action_id] = context
        globals()[f"context_{action_id}"] = context
    UI_BRIDGE_MANIFEST = build_bridge_manifest()
    (root / "control_panel_ui_bridge.json").write_text(
        json.dumps(UI_BRIDGE_MANIFEST, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(UI_BRIDGE_MANIFEST, indent=2, sort_keys=True))
else:
    css = widgets.HTML(
        """
        <style>
        .qm-panel-title { font-size: 20px; font-weight: 700; margin: 4px 0 8px 0; }
        .qm-panel-note { color: #333; margin-bottom: 10px; }
        .qm-section-label { font-weight: 700; margin: 12px 0 6px 0; }
        </style>
        """
    )
    title = widgets.HTML("<div class='qm-panel-title'>Quantule Mapper Colab Control Panel</div>")
    note = widgets.HTML(
        "<div class='qm-panel-note'>Buttons run registered trusted handlers only. "
        "Green status means infrastructure success, not scientific validation. "
        "Full logs are persisted under <code>{}</code>.</div>".format(log_root)
    )

    button_bar_items = []
    output_rows = []
    for section_name, predicate in (
        ("SYSTEM TESTS", lambda item: item["action_id"].startswith("T")),
        ("PACKAGED JOBS", lambda item: item["action_id"].startswith("J")),
    ):
        section_outputs = [widgets.HTML(f"<div class='qm-section-label'>{section_name}</div>")]
        for action in CONTROL_PANEL["actions"]:
            if not predicate(action):
                continue
            action_id = action["action_id"]
            initial_state = read_status(action_id).get("state", "READY" if action_id.startswith("J") else "NOT_RUN")
            color = status_color(initial_state)
            button = widgets.Button(
                description=f"{action_id}: {action.get('label', action_id)}",
                button_style=action_button_style(action_id),
                tooltip=f"Handler: {action.get('handler')} | Output: {action.get('output_channel')}",
                layout=widgets.Layout(width="220px", margin="0 8px 8px 0"),
            )
            status_label = widgets.HTML(
                value=status_html(initial_state),
                layout=widgets.Layout(
                    width="180px",
                    min_width="180px",
                    border=f"2px solid {color}",
                    padding="6px",
                    margin="0 8px 0 0",
                ),
            )
            output = widgets.Output(
                layout=widgets.Layout(
                    border=f"2px solid {color}",
                    height="170px",
                    max_height="170px",
                    overflow_y="auto",
                    width="100%",
                    padding="4px",
                )
            )
            button.on_click(make_button_callback(action_id))
            button_widgets[action_id] = button
            status_widgets[action_id] = status_label
            output_widgets[action_id] = output
            context = ActionContext(action_id, action, button=button, output=output, status=status_label)
            job_contexts[action_id] = context
            globals()[f"button_{action_id}"] = button
            globals()[f"output_{action_id}"] = output
            globals()[f"status_{action_id}"] = status_label
            globals()[f"context_{action_id}"] = context

            button_bar_items.append(button)
            section_outputs.append(
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.HTML(f"<b>{action_id}</b><br>{action.get('label', action_id)}"),
                                status_label,
                                widgets.HTML(f"<small>{action.get('output_channel')}</small>"),
                            ],
                            layout=widgets.Layout(width="220px", min_width="220px"),
                        ),
                        output,
                    ],
                    layout=widgets.Layout(
                        border="1px solid #d0d0d0",
                        padding="6px",
                        margin="6px 0",
                        align_items="stretch",
                        width="100%",
                    ),
                )
            )
        output_rows.extend(section_outputs)

    UI_BRIDGE_MANIFEST = build_bridge_manifest()
    (root / "control_panel_ui_bridge.json").write_text(
        json.dumps(UI_BRIDGE_MANIFEST, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    bridge_summary = widgets.HTML(
        "<details><summary>Agent bridge manifest</summary><pre style='max-height:220px; overflow:auto'>{}</pre></details>".format(
            json.dumps(UI_BRIDGE_MANIFEST, indent=2, sort_keys=True)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
    )
    display(
        widgets.VBox(
            [
                css,
                title,
                note,
                widgets.HBox(
                    button_bar_items,
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="row wrap",
                        align_items="center",
                        border="1px solid #d0d0d0",
                        padding="8px",
                        margin="0 0 10px 0",
                    ),
                ),
                *output_rows,
                bridge_summary,
            ],
            layout=widgets.Layout(width="100%"),
        )
    )
'''

    @staticmethod
    def _preflight_cell() -> str:
        return r'''
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

root = Path(JOB_ROOT)
preflight = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "cwd": os.getcwd(),
    "python": sys.version,
    "platform": platform.platform(),
}

def run_capture(cmd):
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        return {"error": repr(exc)}

def parse_gpu_resource_query(text):
    rows = []
    for line in str(text or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 7:
            try:
                rows.append(
                    {
                        "name": parts[0],
                        "memory_total_mb": float(parts[1]),
                        "memory_used_mb": float(parts[2]),
                        "memory_free_mb": float(parts[3]),
                        "utilization_gpu_percent": float(parts[4]),
                        "utilization_memory_percent": float(parts[5]),
                        "temperature_gpu_c": float(parts[6]),
                    }
                )
            except ValueError:
                rows.append({"raw": line})
    return rows

def system_resource_snapshot():
    snapshot = {"disk_content": {}}
    try:
        total, used, free = shutil.disk_usage("/content")
        snapshot["disk_content"] = {"total_bytes": total, "used_bytes": used, "free_bytes": free}
    except Exception as exc:
        snapshot["disk_error"] = repr(exc)
    try:
        meminfo = {}
        for raw in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = raw.split(":", 1)
            meminfo[key] = int(value.strip().split()[0]) * 1024
        snapshot["system_ram"] = {
            "total_bytes": meminfo.get("MemTotal"),
            "available_bytes": meminfo.get("MemAvailable"),
            "used_estimate_bytes": (
                meminfo.get("MemTotal") - meminfo.get("MemAvailable")
                if meminfo.get("MemTotal") and meminfo.get("MemAvailable")
                else None
            ),
        }
    except Exception as exc:
        snapshot["system_ram_error"] = repr(exc)
    return snapshot

preflight["nvidia_smi"] = run_capture(["nvidia-smi"])
preflight["nvidia_smi_query"] = run_capture([
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu",
    "--format=csv,noheader,nounits",
])
preflight["resource_profile"] = RESOURCE_PROFILE
preflight["host_resources"] = system_resource_snapshot()
preflight["gpu_resources"] = parse_gpu_resource_query(preflight["nvidia_smi_query"].get("stdout", ""))
preflight["pip_freeze"] = run_capture([sys.executable, "-m", "pip", "freeze"])
(root / "environment_pip_freeze.txt").write_text(
    preflight["pip_freeze"].get("stdout", ""),
    encoding="utf-8",
)

precision = MANIFEST.get("precision", {})
try:
    import numpy as np
    try:
        import scipy
        preflight["scipy_version"] = scipy.__version__
    except Exception as exc:
        preflight["scipy_import_error"] = repr(exc)
    try:
        import pywt
        preflight["pywavelets_version"] = pywt.__version__
    except Exception as exc:
        preflight["pywavelets_import_error"] = repr(exc)
    import jax
    if precision.get("jax_enable_x64", True):
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jaxlib

    devices = jax.devices()
    x = jnp.array([1.0, 2.0], dtype=jnp.float64)
    y = (x * jnp.float64(2.0)).block_until_ready()
    matrix = jnp.eye(8, dtype=jnp.float64)
    smoke = (matrix @ matrix + jnp.fft.fft(jnp.ones(8, dtype=jnp.float64))).block_until_ready()
    preflight.update(
        {
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "numpy_version": np.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in devices],
            "jax_process_index": jax.process_index(),
            "jax_x64_enabled": bool(jax.config.jax_enable_x64),
            "float64_kernel_dtype": str(y.dtype),
            "numerical_gpu_smoke_dtype": str(smoke.dtype),
            "numerical_gpu_smoke_finite": bool(jnp.all(jnp.isfinite(smoke))),
        }
    )
except Exception as exc:
    preflight["jax_preflight_error"] = repr(exc)

(root / "gpu_preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(preflight, indent=2, sort_keys=True)[:4000])

policy = MANIFEST.get("environment_policy", {})
if policy.get("require_gpu", True):
    if preflight.get("jax_backend") != "gpu":
        raise RuntimeError("JAX GPU backend is required but was not detected.")

accepted = policy.get("accepted_gpu_names", ["A100"])
if accepted:
    haystack = "\n".join(preflight.get("jax_devices", [])) + "\n" + preflight.get("nvidia_smi", {}).get("stdout", "")
    if not any(name.lower() in haystack.lower() for name in accepted):
        raise RuntimeError(f"GPU is not in accepted list {accepted}.")

if precision.get("jax_enable_x64", True):
    if not preflight.get("jax_x64_enabled") or preflight.get("float64_kernel_dtype") != "float64":
        raise RuntimeError("JAX x64 preflight failed.")
if not preflight.get("numerical_gpu_smoke_finite"):
    raise RuntimeError("Numerical GPU smoke test failed.")
'''

    @staticmethod
    def _dependencies_cell() -> str:
        return r'''
from pathlib import Path
import json
import subprocess
import sys

root = Path(JOB_ROOT)
policy = MANIFEST.get("install_policy", {})
requirements = MANIFEST.get("requirements_colab")

if not requirements:
    print("No requirements_colab declared.")
elif not policy.get("allow_pip_install", False):
    print(f"requirements_colab declared ({requirements}) but allow_pip_install is false; preserving runtime.")
else:
    req_path = root / requirements
    if not req_path.exists():
        raise FileNotFoundError(req_path)
    filtered = []
    skipped = []
    for raw_line in req_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        package_name = line.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].split("[", 1)[0].lower()
        if package_name in {"jax", "jaxlib"}:
            skipped.append(line)
            continue
        filtered.append(line)
    if skipped:
        print("Skipped JAX packages to preserve Colab GPU runtime:", skipped)
    if filtered:
        temp_req = root / "requirements_colab.filtered.txt"
        temp_req.write_text("\n".join(filtered) + "\n", encoding="utf-8")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(temp_req)])
    else:
        print("No non-JAX packages to install.")

post = {"stage": "after_dependencies"}
try:
    import jax
    import jax.numpy as jnp
    if MANIFEST.get("precision", {}).get("jax_enable_x64", True):
        jax.config.update("jax_enable_x64", True)
    x = jnp.array([1.0, 2.0], dtype=jnp.float64)
    y = (x + jnp.float64(1.0)).block_until_ready()
    post.update(
        {
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_x64_enabled": bool(jax.config.jax_enable_x64),
            "float64_kernel_dtype": str(y.dtype),
            "numerical_gpu_smoke_finite": bool(jnp.all(jnp.isfinite(y))),
        }
    )
except Exception as exc:
    post["jax_post_dependency_error"] = repr(exc)

(root / "gpu_preflight_after_dependencies.json").write_text(
    json.dumps(post, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(post, indent=2, sort_keys=True))
if MANIFEST.get("environment_policy", {}).get("require_gpu", True) and post.get("jax_backend") != "gpu":
    raise RuntimeError("JAX GPU backend failed after dependency handling.")
if MANIFEST.get("precision", {}).get("jax_enable_x64", True):
    if not post.get("jax_x64_enabled") or post.get("float64_kernel_dtype") != "float64":
        raise RuntimeError("JAX x64 post-dependency preflight failed.")
'''

    @staticmethod
    def _external_and_smoke_cell() -> str:
        return r'''
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

root = Path(JOB_ROOT)
root_resolved = root.resolve()

def safe_dest(rel_path):
    dest = (root / rel_path).resolve()
    try:
        dest.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Refusing external stage path outside job root: {rel_path}") from exc
    return dest

def safe_child(base, rel_path):
    base_resolved = Path(base).resolve()
    dest = (base_resolved / rel_path).resolve()
    try:
        dest.relative_to(base_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Refusing external required file outside staged root: {rel_path}") from exc
    return dest

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

missing = []
staged = []
for item in MANIFEST.get("external_inputs", []):
    path = Path(item["path"])
    required = item.get("required", True)
    if required and not path.exists():
        missing.append(str(path))
        continue
    if not path.exists():
        continue

    stage_to = item.get("stage_to")
    staged_root = path
    if stage_to:
        dest = safe_dest(stage_to)
        if path.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(path, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        staged_root = dest

    if path.is_file() and item.get("sha256"):
        actual = file_sha256(staged_root)
        expected = item["sha256"]
        if str(expected).startswith("REPLACE_"):
            raise RuntimeError(f"Replace placeholder sha256 for external input {path}.")
        if actual != expected:
            raise RuntimeError(f"External input hash mismatch for {path}: {actual} != {expected}")

    verified_files = []
    for required_file in item.get("required_files", []):
        rel_path = required_file["path"]
        target = safe_child(staged_root, rel_path)
        if not target.exists():
            raise FileNotFoundError(f"Missing required staged external file: {target}")
        expected = required_file.get("sha256")
        if not expected:
            raise RuntimeError(f"Missing sha256 for external required file {rel_path}")
        if str(expected).startswith("REPLACE_"):
            raise RuntimeError(f"Replace placeholder sha256 for external required file {rel_path}.")
        actual = file_sha256(target)
        if actual != expected:
            raise RuntimeError(f"External required file hash mismatch for {rel_path}: {actual} != {expected}")
        verified_files.append({"path": rel_path, "sha256": actual})

    staged.append(
        {
            "source_path": str(path),
            "staged_path": str(staged_root),
            "verified_files": verified_files,
        }
    )
if missing:
    raise FileNotFoundError("Missing required external inputs: " + ", ".join(missing))

(root / "external_inputs_staged.json").write_text(
    json.dumps({"external_inputs": staged}, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps({"external_inputs": staged}, indent=2, sort_keys=True))

entrypoint_paths = set()

def add_entrypoint_path(value):
    if isinstance(value, dict) and value.get("path"):
        entrypoint_paths.add(str(value["path"]))

def add_stage_paths(stages):
    if not stages:
        return
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        add_entrypoint_path(stage.get("entrypoint"))
        add_entrypoint_path(stage)

add_entrypoint_path(MANIFEST.get("entrypoint"))
add_stage_paths(MANIFEST.get("stages"))
for row in MANIFEST.get("rows", []) or []:
    add_stage_paths(row.get("stages"))

if not entrypoint_paths:
    raise RuntimeError("No manifest entrypoints were declared for py_compile smoke test.")
for rel_path in sorted(entrypoint_paths):
    subprocess.check_call([sys.executable, "-m", "py_compile", str(root / rel_path)], cwd=root)
print("Entrypoint py_compile smoke test passed:", sorted(entrypoint_paths))

smoke_command = MANIFEST.get("smoke_command", [])
if smoke_command:
    print("Running manifest smoke command:", smoke_command)
    subprocess.check_call([str(part) for part in smoke_command], cwd=root)
'''

    @staticmethod
    def _run_cell() -> str:
        return r'''
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(JOB_ROOT)
ledger_path = root / "completion_ledger.json"
row_marker_dir = root / "row_markers"
row_marker_dir.mkdir(exist_ok=True)

def load_ledger():
    if ledger_path.exists():
        return json.loads(ledger_path.read_text(encoding="utf-8"))
    return {"completed_rows": [], "failed_rows": [], "rows": []}

def write_ledger(ledger):
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def render_manifest_value(value, row_id, stage_id=None):
    mapping = {
        "job_root": str(root),
        "job_id": str(MANIFEST.get("job_id", "colab_capsule_job")),
        "row_id": str(row_id),
        "stage_id": "" if stage_id is None else str(stage_id),
    }
    if isinstance(value, str):
        rendered = value
        for key, replacement in mapping.items():
            rendered = rendered.replace("{" + key + "}", replacement)
        return rendered
    if isinstance(value, list):
        return [render_manifest_value(item, row_id, stage_id) for item in value]
    if isinstance(value, dict):
        return {key: render_manifest_value(item, row_id, stage_id) for key, item in value.items()}
    return value

def row_stages(row):
    stages = row.get("stages")
    if stages is None:
        stages = MANIFEST.get("stages")
    if stages:
        return stages
    return [
        {
            "stage_id": "main",
            "entrypoint": MANIFEST["entrypoint"],
            "arguments": row.get("arguments", MANIFEST.get("arguments", [])),
            "expected_outputs": row.get("expected_outputs", []),
        }
    ]

def stage_command(stage, row_id, stage_id):
    rendered = render_manifest_value(stage, row_id, stage_id)
    command = rendered.get("command")
    if command:
        if not isinstance(command, list):
            raise RuntimeError(f"Stage {stage_id} command must be a list.")
        command = [str(part) for part in command]
        if command and command[0] in {"python", "python3", "{python}"}:
            command[0] = sys.executable
        return command
    entry = rendered.get("entrypoint") if isinstance(rendered.get("entrypoint"), dict) else rendered
    path = entry.get("path") if isinstance(entry, dict) else None
    if not path:
        raise RuntimeError(f"Stage {stage_id} must define entrypoint.path or command.")
    arguments = rendered.get("arguments")
    if arguments is None and isinstance(entry, dict):
        arguments = entry.get("arguments", [])
    return [sys.executable, "-u", str(path)] + [str(arg) for arg in (arguments or [])]

def verify_expected_outputs(expected_outputs):
    missing = []
    for rel_path in expected_outputs:
        if not (root / str(rel_path)).exists():
            missing.append(str(rel_path))
    if missing:
        raise RuntimeError("Missing expected outputs: " + ", ".join(missing))

def checkpoint_row(row_id):
    checkpoint_dir = MANIFEST.get("drive_checkpoint_dir")
    if not checkpoint_dir:
        return None
    dest_dir = Path(checkpoint_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = dest_dir / f"{MANIFEST.get('job_id', 'colab_capsule_job')}_{row_id}_checkpoint.json"
    payload = {
        "row_id": row_id,
        "ledger": json.loads(ledger_path.read_text(encoding="utf-8")),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = Path(str(checkpoint) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(checkpoint)
    return str(checkpoint)

rows = MANIFEST.get("rows")
if rows:
    run_rows = rows
else:
    run_rows = [
        {
            "row_id": "main",
            "arguments": MANIFEST.get("arguments", []),
            "expected_outputs": MANIFEST.get("expected_outputs", []),
        }
    ]

ledger = load_ledger()
for row in run_rows:
    row_id = str(row.get("row_id", f"row_{len(ledger['rows']) + 1}"))
    if row_id in ledger.get("completed_rows", []):
        print(f"Skipping completed row {row_id}")
        continue

    log_path = root / f"run_{row_id}.log"
    start = time.time()
    stages = row_stages(row)
    started = {
        "row_id": row_id,
        "stages": [
            str(stage.get("stage_id", f"stage_{idx}"))
            for idx, stage in enumerate(stages, start=1)
        ],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (row_marker_dir / f"{row_id}_STARTED.json").write_text(json.dumps(started, indent=2), encoding="utf-8")

    stage_records = []
    return_code = 0
    failed_stage = None
    with log_path.open("wb") as log:
        for idx, stage in enumerate(stages, start=1):
            stage_id = str(stage.get("stage_id", f"stage_{idx}"))
            cmd = stage_command(stage, row_id, stage_id)
            print(f"Running stage {stage_id}:", " ".join(cmd))
            log.write((f"Running stage {stage_id}: " + " ".join(cmd) + "\n").encode("utf-8"))
            log.flush()
            stage_start = time.time()
            proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert proc.stdout is not None
            for chunk in iter(proc.stdout.readline, b""):
                text = chunk.decode("utf-8", errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                log.write(chunk)
                log.flush()
            return_code = proc.wait()
            stage_records.append(
                {
                    "stage_id": stage_id,
                    "command": cmd,
                    "return_code": return_code,
                    "duration_seconds": time.time() - stage_start,
                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            if return_code != 0:
                failed_stage = stage_id
                break
            verify_expected_outputs(stage.get("expected_outputs", []))

    finished = {
        "row_id": row_id,
        "return_code": return_code,
        "duration_seconds": time.time() - start,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "log_path": str(log_path),
        "log_sha256": file_sha256(log_path),
        "stages": stage_records,
    }
    ledger.setdefault("rows", []).append(finished)

    if return_code == 0:
        ledger.setdefault("completed_rows", []).append(row_id)
        (row_marker_dir / f"{row_id}_COMPLETE.json").write_text(json.dumps(finished, indent=2), encoding="utf-8")
        write_ledger(ledger)
        checkpoint = checkpoint_row(row_id)
        if checkpoint:
            print(f"Wrote row checkpoint: {checkpoint}")
    else:
        ledger.setdefault("failed_rows", []).append(row_id)
        (row_marker_dir / f"{row_id}_FAILED.json").write_text(json.dumps(finished, indent=2), encoding="utf-8")
        write_ledger(ledger)
        raise RuntimeError(f"Row {row_id} failed in stage {failed_stage} with return code {return_code}; see {log_path}")

marker = root / "RUN_COMPLETE_CAPSULE.json"
marker.write_text(
    json.dumps(
        {
            "status": "RUN_COMPLETE_CAPSULE",
            "completed_rows": ledger.get("completed_rows", []),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(json.dumps(json.loads(marker.read_text(encoding="utf-8")), indent=2))
'''

    @staticmethod
    def _archive_cell() -> str:
        return r'''
import json
import hashlib
from pathlib import Path
import shutil
import tarfile
import time

root = Path(JOB_ROOT)
job_id = MANIFEST.get("job_id", "colab_capsule_job")
stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
archive_path = Path("/content") / f"{job_id}_{stamp}.tar.gz"

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

expected_outputs = MANIFEST.get("expected_outputs", [])
missing_expected = []
for rel_path in expected_outputs:
    candidate = root / str(rel_path)
    if not candidate.exists():
        missing_expected.append(str(rel_path))
if missing_expected:
    (root / "MISSING_EXPECTED_OUTPUTS.json").write_text(
        json.dumps({"missing_expected_outputs": missing_expected}, indent=2),
        encoding="utf-8",
    )
    raise RuntimeError("Missing expected outputs: " + ", ".join(missing_expected))

provenance = {
    "job_manifest": MANIFEST,
    "workflow": WORKFLOW,
    "resource_profile": RESOURCE_PROFILE,
    "source_manifest": SOURCE_MANIFEST,
    "completion_ledger": json.loads((root / "completion_ledger.json").read_text(encoding="utf-8")) if (root / "completion_ledger.json").exists() else {},
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(root / "capsule_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")

integrity_files = []
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name != "result_integrity_manifest.json":
        integrity_files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
(root / "result_integrity_manifest.json").write_text(
    json.dumps({"files": integrity_files}, indent=2, sort_keys=True),
    encoding="utf-8",
)

with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(root, arcname=job_id)

archive_size = archive_path.stat().st_size
policy = WORKFLOW.get("result_policy", {})
ideal_bytes = int(policy.get("ideal_archive_bytes", 1000000000))
hard_bytes = int(policy.get("hard_archive_bytes", 5000000000))
size_report = {
    "archive_path": str(archive_path),
    "archive_size_bytes": archive_size,
    "ideal_archive_bytes": ideal_bytes,
    "hard_archive_bytes": hard_bytes,
    "over_ideal": archive_size > ideal_bytes,
    "over_hard": archive_size > hard_bytes,
}
(root / "archive_size_report.json").write_text(json.dumps(size_report, indent=2), encoding="utf-8")
print(json.dumps(size_report, indent=2))

if archive_size > hard_bytes:
    raise RuntimeError(
        f"Result archive exceeds hard cap: {archive_size} bytes > {hard_bytes} bytes. "
        "Reduce expected outputs or archive policy before copying to Drive."
    )
if archive_size > ideal_bytes:
    print(f"Warning: archive exceeds ideal size target ({archive_size} bytes > {ideal_bytes} bytes).")

print(f"Archived capsule results to {archive_path}")

drive_dir = MANIFEST.get("drive_results_dir")
if drive_dir:
    dest_dir = Path(drive_dir)
    if str(dest_dir).startswith("/content/drive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception as exc:
            print(f"Drive mount attempt skipped/failed: {exc}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / archive_path.name
    tmp = Path(str(dest) + ".tmp")
    shutil.copy2(archive_path, tmp)
    source_hash = file_sha256(archive_path)
    copied_hash = file_sha256(tmp)
    if source_hash != copied_hash:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Drive copy hash mismatch: {source_hash} != {copied_hash}")
    tmp.replace(dest)
    sidecar = dest_dir / f"{archive_path.name}.sha256.json"
    sidecar_tmp = Path(str(sidecar) + ".tmp")
    sidecar_tmp.write_text(
        json.dumps(
            {
                "archive": archive_path.name,
                "sha256": source_hash,
                "size_bytes": archive_size,
                "copied_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    sidecar_tmp.replace(sidecar)
    print(f"Copied archive to {dest}")
    print(f"Copied archive integrity sidecar to {sidecar}")
    print(f"Download the archive from Drive and place it in {WORKFLOW.get('local_results_dir')}")
else:
    print("No drive_results_dir declared; archive remains in /content.")
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic Colab capsule notebook.")
    parser.add_argument("project_root", help="Project root containing manifest-relative files.")
    parser.add_argument("--manifest", required=True, help="JSON/YAML job manifest.")
    parser.add_argument("-o", "--output", help="Output .ipynb path.")
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument("--max-notebook-bytes", type=int, default=DEFAULT_MAX_NOTEBOOK_BYTES)
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and report capsule contents.")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    if not project_root.is_dir():
        raise SystemExit(f"Project root not found: {project_root}")

    packager = CapsulePackager(
        project_root,
        Path(args.manifest),
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
        max_notebook_bytes=args.max_notebook_bytes,
    )

    if args.dry_run:
        files, findings = packager.collect_files()
        if findings:
            for finding in findings:
                print(f"{finding['path']}:{finding['line']} ({finding['kind']})")
            raise SystemExit(2)
        manifest = packager.source_manifest(files)
        manifest["workflow"] = packager.workflow
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    notebook = packager.generate_notebook()
    if args.output:
        output = Path(args.output)
        if output.suffix.lower() != ".ipynb":
            output = output.with_suffix(".ipynb")
    else:
        job_id = str(packager.manifest.get("job_id", "colab_capsule_job"))
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = project_root / f"{job_id}_{stamp}.ipynb"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(notebook, encoding="utf-8")
    print(f"[Success] Wrote Colab capsule notebook: {output}")


if __name__ == "__main__":
    main()
