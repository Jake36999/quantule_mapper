"""
mcp_server.write_tools — staging + guarded GPU/CPU launchers.

Design (MCP_TOOLS_SPEC.md §3.7-3.10, §4):
  * stage_simulation_manifest  — write, NO GPU.  Validates + stages a manifest and
    returns a review summary.  Fully testable here.
  * run_simulation_manifest    — write, GPU.  Executes a *staged* manifest only,
    with age/confirm/path-redirect/audit gates.  Gate logic testable; the actual
    launch is delegated to an injectable `launcher` (default = subprocess to
    worker_cupy.py) that runs only on a GPU box.
  * run_smoke_simulation       — write, GPU.  Hard-capped smoke run under runs/_smoke/.
  * validate_artifact          — write, CPU.  Guarded wrapper over the validation
    pipeline (injectable `runner`).

All GPU/CPU execution is injected so every guard here is unit-tested without a GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from mcp_server import guards
from mcp_server.config import McpConfig

STALE_MANIFEST_SECONDS = 1800  # staged manifest must be < 30 min old (spec §3.9)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_audit(cfg: McpConfig, event: Dict[str, Any]) -> None:
    """Append one JSONL audit event to cfg.audit_log (best-effort, never raises)."""
    try:
        os.makedirs(os.path.dirname(cfg.audit_log), exist_ok=True)
        event = {"timestamp": _utc_now_iso(), **event}
        with open(cfg.audit_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
    except Exception:
        pass


def _hunt_existing_keys(cfg: McpConfig, hunt_name: str) -> List[tuple]:
    """Distinct (solver_contract_version, variant_label, n_grid) already present for a hunt."""
    keys: List[tuple] = []
    try:
        import sqlite3
        if not os.path.exists(cfg.db_path):
            return keys
        conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True, timeout=10.0)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
            if not {"solver_contract_version", "variant_label", "n_grid", "hunt_name"}.issubset(cols):
                return keys
            rows = conn.execute(
                "SELECT DISTINCT solver_contract_version, variant_label, n_grid "
                "FROM runs WHERE hunt_name = ? AND solver_contract_version IS NOT NULL",
                (hunt_name,),
            ).fetchall()
            keys = [tuple(r) for r in rows]
        finally:
            conn.close()
    except Exception:
        pass
    return keys


# ---------------------------------------------------------------------------
# stage_simulation_manifest  (write, no GPU)
# ---------------------------------------------------------------------------

def stage_simulation_manifest(
    cfg: McpConfig,
    params: Dict[str, Any],
    hunt_name: str,
    generation: int,
    seed: int = 0,
    N_grid: Optional[int] = None,
    T_steps: Optional[int] = None,
    dt: Optional[float] = None,
    L_domain: float = 10.0,
    variant_label: Optional[str] = None,
    overwrite: bool = False,
    utc_date: Optional[str] = None,
) -> Dict[str, Any]:
    from orchestrator import run_identity as ri
    from orchestrator import path_utils as pu

    out: Dict[str, Any] = {
        "staged": False, "staged_manifest_path": None, "config_hash": None,
        "variant_label": None, "affect_topology": None, "affect_strength": None,
        "solver_contract_version": None, "expected_output_path": None,
        "compatibility_warnings": [], "validation_errors": [],
        "review_required": True, "message": "",
    }
    params = dict(params or {})

    # Defaults pulled from a `simulation` sub-dict if present, else explicit args.
    sim_in = params.get("simulation", {}) if isinstance(params.get("simulation"), dict) else {}
    N_grid = int(N_grid if N_grid is not None else sim_in.get("N_grid", 64))
    T_steps = int(T_steps if T_steps is not None else sim_in.get("T_steps", 250))
    dt = float(dt if dt is not None else sim_in.get("dt", 0.001))
    L_domain = float(L_domain if L_domain is not None else sim_in.get("L_domain", 10.0))

    errors, warnings = guards.validate_manifest(params, N_grid, T_steps, dt, L_domain)

    # Derive variant / contract / topology (params-driven; explicit override sanity-checked).
    derived_variant = ri.variant_label_for_params(params)
    if variant_label and variant_label != derived_variant:
        warnings.append(
            f"requested variant_label={variant_label!r} but params imply {derived_variant!r}; using {derived_variant!r}"
        )
    variant = derived_variant
    contract = ri.solver_contract_version_for_params(params)
    topology = ri.affect_topology_for_params(params)
    strength = ri.affect_strength_for_params(params)
    out.update({
        "variant_label": variant, "solver_contract_version": contract,
        "affect_topology": topology, "affect_strength": strength,
        "validation_errors": errors, "compatibility_warnings": warnings,
    })

    if errors:
        out["message"] = "Staging rejected: " + "; ".join(errors)
        return out

    # config_hash over physics+simulation (NOT seed) so multi-seed runs share it.
    from orchestrator.contracts import JobManifest
    hash_params = dict(params)
    hash_params["simulation"] = {
        "N_grid": N_grid, "L_domain": L_domain, "T_steps": T_steps, "dt": dt,
        "collapse_threshold": params.get("collapse_threshold"),
    }
    manifest = JobManifest.from_params(hash_params, generation=generation, seed=seed, hunt_name=hunt_name)
    config_hash = manifest.config_hash
    run_id = manifest.job_id
    out["config_hash"] = config_hash

    utc_date = utc_date or _utc_now_iso()[:10]
    runs_root = os.path.join(cfg.root, "runs")
    expected_output = pu.build_artifact_path(
        runs_root, hunt_name, utc_date, contract, variant, generation, config_hash, seed,
    )
    out["expected_output_path"] = expected_output

    # No silent overwrite of a completed artifact.
    ow_errors = guards.check_no_overwrite(expected_output, overwrite)
    if ow_errors:
        out["validation_errors"] = errors + ow_errors
        out["message"] = "Staging rejected: " + "; ".join(ow_errors)
        return out

    # Compatibility advisory: warn if the hunt already holds a different variant/grid.
    incoming_key = (contract, variant, N_grid)
    existing = _hunt_existing_keys(cfg, hunt_name)
    foreign = [k for k in existing if k != incoming_key]
    if foreign:
        warnings.append(
            f"hunt {hunt_name!r} already contains other variant/grid keys {foreign}; "
            f"this run ({incoming_key}) will be isolated and not ranked against them"
        )

    # Build the worker-facing manifest (adds global_seed for the RNG) + staging block.
    worker_params = dict(params)
    worker_params["simulation"] = hash_params["simulation"]
    worker_params["global_seed"] = int(seed)
    staged_manifest = {
        "config_hash": config_hash,
        "job_id": run_id,
        "generation": int(generation),
        "seed": int(seed),
        "hunt_name": hunt_name,
        "params": worker_params,
        "_staging": {
            "expected_output_path": expected_output,
            "solver_contract_version": contract,
            "variant_label": variant,
            "affect_topology": topology,
            "affect_strength": strength,
            "N_grid": N_grid, "T_steps": T_steps, "dt": dt, "L_domain": L_domain,
            "staged_at_utc": _utc_now_iso(),
            "overwrite": bool(overwrite),
            "review_required": True,
        },
    }

    staged_path = pu.staged_manifest_path(cfg.root, config_hash, seed, run_id)
    os.makedirs(os.path.dirname(staged_path), exist_ok=True)
    with open(staged_path, "w", encoding="utf-8") as fh:
        json.dump(staged_manifest, fh, indent=2)

    out["staged"] = True
    out["staged_manifest_path"] = staged_path
    out["compatibility_warnings"] = warnings
    degen = any("DEGENERATE_GEOMETRY" in w for w in warnings)
    out["message"] = (
        f"Staged {variant} N={N_grid} T={T_steps} dt={dt} seed={seed}. "
        f"{'DEGENERATE_GEOMETRY — excluded from main leaderboard. ' if degen else ''}"
        f"{len(warnings)} warning(s). Review, then run_simulation_manifest(confirm=true)."
    )
    return out


# ---------------------------------------------------------------------------
# default GPU launcher (runs only on a GPU box)
# ---------------------------------------------------------------------------

def _default_launcher(manifest_path: str, output_path: str, root: str) -> Dict[str, Any]:
    """Launch worker_cupy.py on the staged manifest. Parses its printed result payload."""
    cmd = ["python", os.path.join(root, "worker_cupy.py"),
           "--manifest", manifest_path, "--output", output_path]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=86400)
    payload: Dict[str, Any] = {"status": "FAIL", "artifact_url": output_path, "reason": "no_payload"}
    for line in (proc.stdout or "").splitlines():
        marker = "Worker result payload: "
        if marker in line:
            try:
                payload = json.loads(line.split(marker, 1)[1])
            except Exception:
                pass
    payload.setdefault("returncode", proc.returncode)
    if proc.returncode != 0 and payload.get("status") not in ("SUCCESS", "PENDING_VALIDATION"):
        payload["status"] = "FAIL"
        payload.setdefault("reason", f"worker_exit_{proc.returncode}")
    return payload


# ---------------------------------------------------------------------------
# run_simulation_manifest  (write, GPU)
# ---------------------------------------------------------------------------

def run_simulation_manifest(
    cfg: McpConfig,
    staged_manifest_path: str,
    confirm: bool = False,
    launcher: Optional[Callable[..., Dict[str, Any]]] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    out = {"status": "REJECTED", "artifact_path": None, "config_hash": None,
           "sentinel_code": None, "wall_time_seconds": 0.0, "audit_event_id": None,
           "errors": []}

    if confirm is not True:
        out["errors"].append("confirm must be true — review the staged manifest before launching")
        return out
    if not staged_manifest_path or not cfg.is_path_allowed(staged_manifest_path) or not os.path.exists(staged_manifest_path):
        out["errors"].append("staged_manifest_path missing or outside project root")
        return out

    age = (now if now is not None else time.time()) - os.path.getmtime(staged_manifest_path)
    if age > STALE_MANIFEST_SECONDS:
        out["errors"].append(f"staged manifest is stale ({int(age)}s > {STALE_MANIFEST_SECONDS}s); re-stage before running")
        return out

    with open(staged_manifest_path, "r", encoding="utf-8") as fh:
        staged = json.load(fh)
    staging = staged.get("_staging", {})
    output_path = staging.get("expected_output_path")
    config_hash = staged.get("config_hash")
    out["config_hash"] = config_hash
    if not output_path:
        out["errors"].append("staged manifest missing expected_output_path")
        return out

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    event_id = f"dispatch_{config_hash[:12]}_{int(time.time())}"
    out["audit_event_id"] = event_id
    _append_audit(cfg, {"stage": "dispatch", "config_hash": config_hash,
                        "generation": staged.get("generation"), "job_id": staged.get("job_id"),
                        "details": {"output_path": output_path, "event_id": event_id}})

    launcher = launcher or (lambda m, o: _default_launcher(m, o, cfg.root))
    t0 = time.time()
    try:
        result = launcher(staged_manifest_path, output_path)
    except Exception as exc:
        _append_audit(cfg, {"stage": "dispatch_failure", "config_hash": config_hash,
                            "details": {"error": f"{type(exc).__name__}: {exc}", "event_id": event_id}})
        out["status"] = "FAIL"
        out["errors"].append(f"launch failed: {type(exc).__name__}: {exc}")
        out["wall_time_seconds"] = round(time.time() - t0, 3)
        return out

    out["wall_time_seconds"] = round(time.time() - t0, 3)
    out["status"] = result.get("status", "FAIL")
    out["artifact_path"] = result.get("artifact_url", output_path)
    out["sentinel_code"] = result.get("sentinel")
    stage = "h5_write" if out["status"] in ("SUCCESS", "PENDING_VALIDATION") else "dispatch_failure"
    _append_audit(cfg, {"stage": stage, "config_hash": config_hash,
                        "details": {"status": out["status"], "artifact_path": out["artifact_path"], "event_id": event_id}})
    return out


# ---------------------------------------------------------------------------
# run_smoke_simulation  (write, GPU, hard-capped)
# ---------------------------------------------------------------------------

def run_smoke_simulation(
    cfg: McpConfig,
    params: Dict[str, Any],
    seed: int = 0,
    N_grid: int = 16,
    T_steps: int = 50,
    L_domain: float = 10.0,
    launcher: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    from orchestrator import run_identity as ri
    from orchestrator import path_utils as pu
    from orchestrator.contracts import JobManifest

    out = {"status": "REJECTED", "artifact_path": None, "sentinel_code": None,
           "sentinel_reason": None, "wall_time_seconds": 0.0, "final_energy": None,
           "solver_contract_version": ri.solver_contract_version_for_params(params or {}),
           "errors": []}

    cap_errors = guards.validate_smoke_caps(N_grid, T_steps)
    if cap_errors:
        out["errors"] = cap_errors
        return out

    params = dict(params or {})
    sim = {"N_grid": int(N_grid), "L_domain": float(L_domain), "T_steps": int(T_steps),
           "dt": float(params.get("dt", 0.001)), "collapse_threshold": params.get("collapse_threshold")}
    hash_params = dict(params); hash_params["simulation"] = sim
    manifest = JobManifest.from_params(hash_params, generation=0, seed=seed, hunt_name="_SMOKE")
    config_hash = manifest.config_hash

    worker_params = dict(params); worker_params["simulation"] = sim; worker_params["global_seed"] = int(seed)
    staged = {"config_hash": config_hash, "job_id": manifest.job_id, "generation": 0,
              "seed": int(seed), "hunt_name": "_SMOKE", "params": worker_params}
    smoke_manifest_path = os.path.join(cfg.root, "runs", "_smoke", f"manifest_{config_hash[:12]}_{seed}.json")
    os.makedirs(os.path.dirname(smoke_manifest_path), exist_ok=True)
    with open(smoke_manifest_path, "w", encoding="utf-8") as fh:
        json.dump(staged, fh, indent=2)

    output_path = pu.smoke_output_path(cfg.root, config_hash, seed)
    _append_audit(cfg, {"stage": "smoke_run", "config_hash": config_hash,
                        "details": {"N_grid": N_grid, "T_steps": T_steps, "output_path": output_path}})

    launcher = launcher or (lambda m, o: _default_launcher(m, o, cfg.root))
    t0 = time.time()
    try:
        result = launcher(smoke_manifest_path, output_path)
    except Exception as exc:
        out["status"] = "FAIL"
        out["errors"].append(f"launch failed: {type(exc).__name__}: {exc}")
        out["wall_time_seconds"] = round(time.time() - t0, 3)
        return out

    out["wall_time_seconds"] = round(time.time() - t0, 3)
    out["status"] = result.get("status", "FAIL")
    out["artifact_path"] = result.get("artifact_url", output_path)
    out["sentinel_code"] = result.get("sentinel")
    out["sentinel_reason"] = result.get("reason")
    out["final_energy"] = result.get("final_energy")
    # Smoke runs are explicitly NOT written to the ledger.
    return out


# ---------------------------------------------------------------------------
# validate_artifact  (write, CPU)
# ---------------------------------------------------------------------------

def validate_artifact(
    cfg: McpConfig,
    artifact_path: str,
    params_path: str,
    output_dir: Optional[str] = None,
    force: bool = False,
    runner: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    from orchestrator import run_identity as ri

    out = {"status": "SKIPPED", "provenance_path": None, "log_prime_sse": None,
           "solver_contract_version": None, "compatibility_ok": False,
           "validation_schema_version": None, "early_rejected": False, "errors": []}

    if not cfg.is_path_allowed(artifact_path) or not os.path.exists(artifact_path):
        out["errors"].append("artifact_path missing or outside project root")
        return out
    if not cfg.is_path_allowed(params_path) or not os.path.exists(params_path):
        out["errors"].append("params_path missing or outside project root")
        return out

    out_dir = output_dir or os.path.dirname(artifact_path)

    # Read contract from /identity for a compatibility note (does not block).
    try:
        import h5py
        with h5py.File(artifact_path, "r") as f:
            ident = ri.read_identity_group(f)
            out["solver_contract_version"] = ident.get("solver_contract_version")
            out["compatibility_ok"] = bool(ident.get("solver_contract_version"))
    except Exception:
        out["compatibility_ok"] = False

    # Provenance overwrite guard.
    existing_prov = ri.provenance_path_for_artifact(out_dir, artifact_path, _config_hash_from(artifact_path))
    if existing_prov and os.path.exists(existing_prov) and not force:
        out["status"] = "SKIPPED"
        out["provenance_path"] = existing_prov
        out["errors"].append("provenance already exists; pass force=true to re-validate")
        return out

    runner = runner or _default_validation_runner
    try:
        result = runner(artifact_path, params_path, out_dir)
    except Exception as exc:
        out["status"] = "FAILED"
        out["errors"].append(f"validation runner error: {type(exc).__name__}: {exc}")
        return out

    out["status"] = "PASSED" if result.get("ok") else "FAILED"
    out["provenance_path"] = result.get("provenance_path")
    out["log_prime_sse"] = result.get("log_prime_sse")
    out["validation_schema_version"] = result.get("schema_version")
    out["early_rejected"] = bool(result.get("early_rejected", False))
    _append_audit(cfg, {"stage": "validation_write", "details": {
        "artifact_path": artifact_path, "status": out["status"], "log_prime_sse": out["log_prime_sse"]}})
    return out


def _config_hash_from(artifact_path: str) -> str:
    base = os.path.basename(artifact_path)
    if base.startswith("rho_history_") and base.endswith(".h5"):
        return base[len("rho_history_"):-3]
    return base.rsplit(".", 1)[0]


def _default_validation_runner(artifact_path: str, params_path: str, output_dir: str) -> Dict[str, Any]:
    """Run the real validation pipeline (CPU).  Imports lazily — heavy deps."""
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from validation_pipeline import ValidationPipeline, SCHEMA_VERSION
    pipeline = ValidationPipeline(input_path=artifact_path, params_path=params_path, output_dir=output_dir)
    ok = pipeline.run()
    prov_path = getattr(pipeline, "provenance_path", None)
    sse = None
    if prov_path and os.path.exists(prov_path):
        try:
            with open(prov_path, "r", encoding="utf-8") as fh:
                sse = json.load(fh).get("spectral_fidelity", {}).get("log_prime_sse")
        except Exception:
            sse = None
    return {"ok": bool(ok), "provenance_path": prov_path, "log_prime_sse": sse,
            "schema_version": SCHEMA_VERSION, "early_rejected": False}
