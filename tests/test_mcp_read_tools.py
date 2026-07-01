"""
tests/test_mcp_read_tools.py

Read-only MCP tools (mcp_server.data_access) against a synthetic project tree:
ledger + provenance + audit JSONL + HDF5 artifact with /identity.  No GPU.
"""
import json
import os
import sqlite3
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from orchestrator.schema_utils import initialize_ledger_schema  # noqa: E402
from orchestrator import run_identity as ri  # noqa: E402
from mcp_server import data_access as da  # noqa: E402
from mcp_server.config import McpConfig  # noqa: E402

HASH = "a3f8c2e91b4d0000"
CONTRACT = "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1"


@pytest.fixture
def env():
    h5py = pytest.importorskip("h5py")
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "simulation_data"))
        os.makedirs(os.path.join(d, "provenance_reports"))
        os.makedirs(os.path.join(d, "runtime_logs"))
        cfg = McpConfig(root=d)

        # --- ledger ---
        initialize_ledger_schema(cfg.db_path)
        conn = sqlite3.connect(cfg.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO runs (config_hash, seed, generation, status, fitness, "
            "solver_contract_version, variant_label, affect_topology, n_grid, hunt_name, utc_start, champion_eligible) "
            "VALUES (?, 7, 3, 'SUCCESS', 0.73, ?, 'LOCAL-RHO', 'none', 64, 'HUNT_X', '2026-06-18T12:00:00Z', 1)",
            (HASH, CONTRACT),
        )
        conn.execute(
            "INSERT OR REPLACE INTO metrics (config_hash, log_prime_sse, bragg_peaks_detected, refinement_status, solver_contract_version) "
            "VALUES (?, 0.73, 5, 'VALIDATED_PROVISIONAL', ?)",
            (HASH, CONTRACT),
        )
        conn.execute(
            "INSERT OR REPLACE INTO parameters (config_hash, param_rho_vac, param_omega0) VALUES (?, 0.8, 0.8)",
            (HASH,),
        )
        # a foreign-variant run that must not be mixed in ranking
        conn.execute(
            "INSERT OR REPLACE INTO runs (config_hash, seed, generation, status, fitness, "
            "solver_contract_version, variant_label, affect_topology, n_grid, hunt_name, champion_eligible) "
            "VALUES ('foreignhash01', 0, 3, 'SUCCESS', 0.01, 'IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1', 'CAUSAL-AFFECT', 'vacuum_ref', 64, 'HUNT_X', 0)",
        )
        conn.execute(
            "INSERT OR REPLACE INTO metrics (config_hash, log_prime_sse, solver_contract_version) "
            "VALUES ('foreignhash01', 0.01, 'IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1')",
        )
        conn.commit()
        conn.close()

        # --- artifact with /identity ---
        ident = ri.build_identity(
            config_hash=HASH, seed=7, generation=3, N_grid=64, dt=0.005, T_steps=1200,
            params={"param_rho_vac": 0.8}, run_id="feedface0000", hunt_name="HUNT_X",
            utc_start="2026-06-18T12:00:00Z", gpu_backend="numpy",
        )
        art = os.path.join(cfg.artifact_roots[0], f"rho_history_{HASH}.h5")
        with h5py.File(art, "w") as f:
            f.create_dataset("psi_final", data=np.zeros((4, 4, 4), dtype=np.complex128), compression="gzip")
            f.create_dataset("A_dot_k_final", data=np.zeros((4, 4, 4), dtype=np.complex128))
            f.create_dataset("solver_contract", data=np.array([json.dumps({"solver_contract_version": CONTRACT})], dtype="S512"))
            ri.write_identity_group(f, ident)

        # --- provenance ---
        prov = os.path.join(cfg.provenance_dir, f"provenance_{HASH}_seed7_20260618_feedface.json")
        with open(prov, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {"schema_version": "SFP-v3.2-ARCS", "run_metadata": {"variant_label": "LOCAL-RHO"}},
                "solver_contract": {"solver_contract_version": CONTRACT},
                "spectral_fidelity": {"log_prime_sse": 0.73, "bragg_peaks_detected": 5},
                "falsifiability": {"phase_scramble_sse": 4.2},
            }, f)

        # --- audit log ---
        with open(cfg.audit_log, "w", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": "2026-06-18T11:59:00Z", "stage": "h5_write", "config_hash": HASH, "generation": 3, "details": {}}) + "\n")
            f.write(json.dumps({"timestamp": "2026-06-18T12:01:00Z", "stage": "result_ingest", "config_hash": HASH, "generation": 3, "details": {"log_prime_sse": 0.73}}) + "\n")

        yield cfg


def test_get_run_status(env):
    out = da.get_run_status(env.db_path, "a3f8c2e91b4d", provenance_dir=env.provenance_dir, artifact_roots=env.artifact_roots)
    assert out["status"] == "SUCCESS"
    assert out["seed"] == 7
    assert out["variant_label"] == "LOCAL-RHO"
    assert out["refinement_status"] == "VALIDATED_PROVISIONAL"
    assert out["artifact_path"] and out["artifact_path"].endswith(".h5")
    assert out["provenance_path"] and "seed7" in out["provenance_path"]


def test_get_run_status_not_found(env):
    out = da.get_run_status(env.db_path, "deadbeef")
    assert out["status"] == "NOT_FOUND"


def test_query_ledger_compatibility_warning(env):
    out = da.query_ledger(env.db_path, hunt_name="HUNT_X")
    assert out["total_matched"] == 2
    # spans LOCAL-RHO + CAUSAL-AFFECT -> must warn, not silently mix
    assert out["compatibility_warning"] is not None
    assert "not comparable" in out["compatibility_warning"].lower()


def test_query_ledger_filtered_single_variant(env):
    out = da.query_ledger(env.db_path, hunt_name="HUNT_X", solver_contract_version=CONTRACT, variant_label="LOCAL-RHO")
    assert out["compatibility_warning"] is None
    assert all(r["solver_contract_version"] == CONTRACT for r in out["rows"])


def test_read_audit_log(env):
    out = da.read_audit_log(env.audit_log, config_hash=HASH)
    assert out["total_lines_scanned"] == 2
    assert len(out["events"]) == 2
    stages = {e["stage"] for e in out["events"]}
    assert stages == {"h5_write", "result_ingest"}


def test_read_audit_log_stage_filter(env):
    out = da.read_audit_log(env.audit_log, stage="h5_write")
    assert len(out["events"]) == 1
    assert out["events"][0]["stage"] == "h5_write"


def test_read_provenance(env):
    out = da.read_provenance(env.provenance_dir, config_hash=HASH, seed=7)
    assert out["found"] is True
    assert out["schema_version"] == "SFP-v3.2-ARCS"
    assert out["solver_contract_version"] == CONTRACT
    assert out["spectral_fidelity"]["log_prime_sse"] == 0.73
    assert out["full_payload"]["falsifiability"]["phase_scramble_sse"] == 4.2


def test_list_artifacts(env):
    out = da.list_artifacts(env.artifact_roots)
    assert out["total_found"] == 1
    a = out["artifacts"][0]
    assert a["has_identity_group"] is True
    assert a["config_hash"] == HASH
    assert a["variant_label"] == "LOCAL-RHO"
    assert a["seed"] == 7


def test_list_artifacts_variant_filter(env):
    assert da.list_artifacts(env.artifact_roots, variant_label="CAUSAL-AFFECT")["total_found"] == 0
    assert da.list_artifacts(env.artifact_roots, variant_label="LOCAL-RHO")["total_found"] == 1


def test_inspect_hdf5_schema(env):
    art = os.path.join(env.artifact_roots[0], f"rho_history_{HASH}.h5")
    out = da.inspect_hdf5_schema(art)
    names = {d["name"] for d in out["datasets"]}
    assert "/psi_final" in names
    assert "/A_dot_k_final" in names
    assert out["identity"]["config_hash"] == HASH
    assert out["solver_contract_json"]["solver_contract_version"] == CONTRACT
    assert out["is_fail_artifact"] is False


def test_summarise_generation(env):
    out = da.summarise_generation(env.db_path, "HUNT_X", 3, CONTRACT, "LOCAL-RHO")
    assert out["total_runs"] == 1  # only the LOCAL-RHO run matches contract+variant
    assert out["succeeded"] == 1
    assert out["sse_stats"]["golden_count"] == 1  # 0.73 < 1.0
    assert out["new_champion"]["config_hash"] == HASH


def test_audit_data_contract_compliant(env):
    out = da.audit_data_contract(env.db_path, config_hash=HASH, provenance_dir=env.provenance_dir, artifact_roots=env.artifact_roots)
    assert out["total_checked"] == 1
    assert out["compliant"] == 1, out["issues"]


def test_audit_data_contract_flags_missing_artifact(env):
    out = da.audit_data_contract(env.db_path, config_hash="foreignhash01", provenance_dir=env.provenance_dir, artifact_roots=env.artifact_roots)
    assert out["non_compliant"] == 1
    assert any("artifact not found" in i for issue in out["issues"] for i in issue["issues"])


def test_server_wires_eight_readonly_tools():
    import mcp_server.server as srv
    for name in [
        "get_run_status", "query_ledger", "read_audit_log", "read_provenance",
        "list_artifacts", "inspect_hdf5_schema", "summarise_generation", "audit_data_contract",
    ]:
        assert callable(getattr(srv, name)), f"server missing tool {name}"
    assert srv.mcp is not None


def test_server_path_whitelist_blocks_outside_root(env):
    import mcp_server.server as srv
    srv.CFG = env  # point the server at the sandbox
    blocked = srv.inspect_hdf5_schema(os.path.join(os.path.dirname(env.root), "outside.h5"))
    assert blocked.get("error") == "path outside project root"
