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
from mcp_server import research_tools as rt

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


# ----------------------------------------------------------------------------
# RETIRED 2026-10-05: run_simulation_manifest, run_smoke_simulation, validate_artifact.
#
# These three were MCP tools that let a connected agent START work: two launched worker_cupy.py on the
# GPU, and one ran the CPU validation pipeline and wrote provenance. That contradicts the rule adopted for
# agent oversight in docs/research_infrastructure/PROCESS_PLAN_2026-10.md (P6) and
# EXPERIMENT_SPECS_MCP_UI.md: an agent may READ everything and PROPOSE work (research_propose_spec ->
# specs/proposed/), but never LAUNCH it. A human approves a spec and runs tools/run_spec.py. It is the same
# no-control-path rule as tools/hud_monitor.py and tools/serve_spec_ui.py.
#
# They also targeted the legacy orchestrator pipeline (simulation_ledger.db, config_hash runs), which is
# no longer how experiments are run.
#
# The implementations are NOT deleted: mcp_server/write_tools.py still provides run_simulation_manifest,
# run_smoke_simulation and validate_artifact for a person to call directly. stage_simulation_manifest
# (above) stays as a tool, because it only validates and stages a file for human review and runs nothing.
# ----------------------------------------------------------------------------


# ============================================================================
# Research-platform tools (2026-10-04, IMPLEMENTATION_PLAN_2026-10 Phase E2).
# Read-only, except propose_spec / draft_reading, which write only into fenced folders.
# There is deliberately no tool that launches a run.
# ============================================================================

@mcp.tool()
def research_list_runs(substrate: Optional[str] = None, branch: Optional[str] = None,
                       stepper: Optional[str] = None, harness: Optional[str] = None,
                       stale_only: bool = False, limit: int = 50) -> Dict[str, Any]:
    """READ. Runs from the results index (docs/runs/_index.sqlite), newest first. Filters: substrate
    (e.g. 'dissipative-S-NCGL', 'TG-dual-substrate'), branch, stepper ('ETDRK4', 'KG-strang', 'TG-RK4'),
    harness id, stale_only (runs that used a since-fixed component)."""
    return rt.list_runs(CFG.root, substrate, branch, stepper, harness, stale_only, limit)


@mcp.tool()
def research_get_run(run_id: str) -> Dict[str, Any]:
    """READ. One run: index row, params, metrics, staleness against component fixes, edges, summary.json."""
    return rt.get_run(CFG.root, run_id)


@mcp.tool()
def research_tail_telemetry(run_id: str, n: int = 50) -> Dict[str, Any]:
    """READ. Last n live-telemetry samples per arm, the declared invariants, and every breach so far.
    Use this to watch a running simulation."""
    return rt.tail_telemetry(CFG.root, run_id, n)


@mcp.tool()
def research_list_stale_runs(fix_id: Optional[str] = None) -> Dict[str, Any]:
    """READ. Runs flagged STALE_PENDING_REVALIDATION by docs/registry/COMPONENT_FIXES.json."""
    return rt.list_stale_runs(CFG.root, fix_id)


@mcp.tool()
def research_list_components() -> Dict[str, Any]:
    """READ. Substrates, initial conditions and observers an experiment spec can name."""
    return rt.list_components(CFG.root)


@mcp.tool()
def research_get_schema() -> Dict[str, Any]:
    """READ. The experiment-spec JSON Schema (irer_specs.SCHEMA)."""
    return rt.get_schema(CFG.root)


@mcp.tool()
def research_list_specs(status: Optional[str] = None) -> Dict[str, Any]:
    """READ. Experiment specs by folder: drafts, proposed (agent suggestions awaiting review), approved."""
    return rt.list_specs(CFG.root, status)


@mcp.tool()
def research_get_spec(spec_id: str) -> Dict[str, Any]:
    """READ. One spec by id."""
    return rt.get_spec(CFG.root, spec_id)


@mcp.tool()
def research_harness_registry(status: Optional[str] = None) -> Dict[str, Any]:
    """READ. Harness lifecycle registry (ACTIVE / SUPERSEDED / RETIRED / PROPOSED / UNREVIEWED)."""
    return rt.harness_registry(CFG.root, status)


@mcp.tool()
def research_propose_spec(spec: Dict[str, Any], author: str = "agent") -> Dict[str, Any]:
    """WRITE (fenced). Validate an experiment spec and file it in specs/proposed/. Never overwrites and
    NEVER launches: a human reviews, moves it to specs/approved/, and runs tools/run_spec.py. The spec
    must include a `prediction` written before any run (what the equations say should happen)."""
    return rt.propose_spec(CFG.root, spec, author)


@mcp.tool()
def research_draft_reading(run_id: str, text: str, author: str = "agent") -> Dict[str, Any]:
    """WRITE (fenced). Append a MACHINE-DRAFTED reading of a run to docs/runs/_machine_drafts/<run_id>.md.
    Never edits the human paired-reading blocks."""
    return rt.draft_reading(CFG.root, run_id, text, author)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
