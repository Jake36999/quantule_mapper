"""
mcp_server.server — FastMCP entrypoint exposing the read-only IRER tools.

Run with:  python -m mcp_server.server   (stdio transport)

Every tool here is READ-ONLY and safe to call without confirmation.  Each is a
thin wrapper over mcp_server.data_access (which holds the testable logic).  Tools
that take an explicit filesystem path enforce the project-root read whitelist.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.config import default_config
from mcp_server import data_access as da
from mcp_server import write_tools as wt

CFG = default_config()
mcp = FastMCP("quantule-mapper")


@mcp.tool()
def get_run_status(config_hash: str, seed: Optional[int] = None, hunt_name: Optional[str] = None) -> Dict[str, Any]:
    """Lifecycle status of a run by config_hash (full or 12-char prefix), optionally a seed/hunt.
    Reads the ledger only; returns status, fitness, contract/variant, refinement status, and
    located artifact/provenance paths."""
    return da.get_run_status(CFG.db_path, config_hash, seed, hunt_name, CFG.provenance_dir, CFG.artifact_roots)


@mcp.tool()
def query_ledger(
    hunt_name: Optional[str] = None,
    generation_min: Optional[int] = None,
    generation_max: Optional[int] = None,
    solver_contract_version: Optional[str] = None,
    variant_label: Optional[str] = None,
    status: Optional[str] = None,
    sse_max: Optional[float] = None,
    limit: int = 50,
    order_by: str = "log_prime_sse",
) -> Dict[str, Any]:
    """Filtered query over the simulation ledger. Sets compatibility_warning (and does NOT
    silently mix) when results span multiple solver contracts/variants."""
    return da.query_ledger(
        CFG.db_path, hunt_name, generation_min, generation_max, solver_contract_version,
        variant_label, status, sse_max, limit, order_by,
    )


@mcp.tool()
def read_audit_log(
    stage: Optional[str] = None,
    config_hash: Optional[str] = None,
    hunt_name: Optional[str] = None,
    since_utc: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Recent run-lifecycle audit events (JSONL), optionally filtered by stage/config_hash/hunt/time."""
    return da.read_audit_log(CFG.audit_log, stage, config_hash, hunt_name, since_utc, limit)


@mcp.tool()
def read_provenance(
    config_hash: Optional[str] = None,
    seed: Optional[int] = None,
    provenance_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Full structured provenance report for a run (spectral fidelity, contract, falsifiability)."""
    if provenance_path and not CFG.is_path_allowed(provenance_path):
        return {"found": False, "error": "path outside project root"}
    return da.read_provenance(CFG.provenance_dir, config_hash, seed, provenance_path)


@mcp.tool()
def list_artifacts(
    hunt_name: Optional[str] = None,
    generation: Optional[int] = None,
    solver_contract_version: Optional[str] = None,
    variant_label: Optional[str] = None,
    include_legacy: bool = False,
    limit: int = 100,
) -> Dict[str, Any]:
    """List HDF5 artifacts under the output hierarchy with /identity metadata. Reads only the
    identity group; does not load field data. Legacy (no-identity) artifacts excluded by default."""
    return da.list_artifacts(
        CFG.artifact_roots, hunt_name, generation, solver_contract_version,
        variant_label, include_legacy, limit,
    )


@mcp.tool()
def inspect_hdf5_schema(artifact_path: str) -> Dict[str, Any]:
    """Structure (dataset shapes/dtypes), /identity, solver_contract, and sentinel info of an
    HDF5 artifact, without loading field data."""
    if not CFG.is_path_allowed(artifact_path):
        return {"path": artifact_path, "error": "path outside project root"}
    return da.inspect_hdf5_schema(artifact_path)


@mcp.tool()
def summarise_generation(
    hunt_name: str,
    generation: int,
    solver_contract_version: str,
    variant_label: str,
) -> Dict[str, Any]:
    """Structured summary of one generation for a specific contract+variant: run counts,
    SSE distribution, sentinel breakdown, champion, degenerate-geometry count."""
    return da.summarise_generation(CFG.db_path, hunt_name, generation, solver_contract_version, variant_label)


@mcp.tool()
def audit_data_contract(
    config_hash: Optional[str] = None,
    hunt_name: Optional[str] = None,
    sample_size: int = 20,
) -> Dict[str, Any]:
    """Cross-check runs for DC-v1.0 compliance: /identity present, contract consistent between
    HDF5 and ledger, provenance present, discriminator columns populated."""
    return da.audit_data_contract(
        CFG.db_path, config_hash, hunt_name, sample_size, CFG.provenance_dir, CFG.artifact_roots,
    )


# ---------------------------------------------------------------------------
# Write / GPU tools — staging is safe (no GPU); run tools require explicit
# confirmation and a freshly staged manifest (see MCP_TOOLS_SPEC.md §4).
# ---------------------------------------------------------------------------

@mcp.tool()
def stage_simulation_manifest(
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
) -> Dict[str, Any]:
    """WRITE (no GPU). Validate and stage a run manifest for review. Performs power-of-2,
    degenerate-geometry, CFL, and no-overwrite checks; derives contract/variant; builds the
    hierarchical output path. MUST be reviewed before run_simulation_manifest. review_required
    is always true."""
    return wt.stage_simulation_manifest(
        CFG, params, hunt_name, generation, seed, N_grid, T_steps, dt, L_domain, variant_label, overwrite,
    )


@mcp.tool()
def run_simulation_manifest(staged_manifest_path: str, confirm: bool = False) -> Dict[str, Any]:
    """WRITE (GPU). Execute a previously staged manifest. Requires confirm=true and a staged
    manifest < 30 min old; the output path is taken from the staged manifest and cannot be
    redirected. Launches worker_cupy.py and writes dispatch/h5_write audit events."""
    return wt.run_simulation_manifest(CFG, staged_manifest_path, confirm)


@mcp.tool()
def run_smoke_simulation(
    params: Dict[str, Any],
    seed: int = 0,
    N_grid: int = 16,
    T_steps: int = 50,
    L_domain: float = 10.0,
) -> Dict[str, Any]:
    """WRITE (GPU). Low-cost smoke test (hard caps N_grid<=32, T_steps<=100). Output under
    runs/_smoke/; never written to the ledger. Appends a smoke_run audit event."""
    return wt.run_smoke_simulation(CFG, params, seed, N_grid, T_steps, L_domain)


@mcp.tool()
def validate_artifact(
    artifact_path: str,
    params_path: str,
    output_dir: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """WRITE (CPU). Run the validation pipeline on an artifact and write provenance. Will not
    overwrite existing provenance unless force=true. Appends a validation_write audit event."""
    if not CFG.is_path_allowed(artifact_path) or not CFG.is_path_allowed(params_path):
        return {"status": "SKIPPED", "errors": ["path outside project root"]}
    return wt.validate_artifact(CFG, artifact_path, params_path, output_dir, force)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
