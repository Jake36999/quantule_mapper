"""
tests/test_mcp_write_tools.py

Write/GPU MCP tools: stage_simulation_manifest (no GPU), and the guarded
launchers run_simulation_manifest / run_smoke_simulation / validate_artifact.
GPU/CPU execution is injected via fake launchers/runners, so every safety gate
is exercised here without a GPU.
"""
import json
import os
import sqlite3
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp_server import write_tools as wt  # noqa: E402
from mcp_server import guards  # noqa: E402
from mcp_server.config import McpConfig  # noqa: E402

BASE_PARAMS = {"param_D": 1.5, "param_eta": 0.3, "param_rho_vac": 0.8, "param_a_coupling": 2.0}


@pytest.fixture
def cfg(tmp_path):
    return McpConfig(root=str(tmp_path))


# ---------------------------------------------------------------------------
# guards (pure)
# ---------------------------------------------------------------------------

class TestGuards:
    def test_power_of_two(self):
        assert guards.is_power_of_two(64)
        assert not guards.is_power_of_two(100)

    def test_cfl_number_scales(self):
        small = guards.cfl_number(1.0, 0.001, 64, 10.0)
        big = guards.cfl_number(1.0, 0.5, 64, 10.0)
        assert big > guards.CFL_LIMIT > small

    def test_validate_manifest_clean(self):
        errors, warnings = guards.validate_manifest(BASE_PARAMS, 64, 250, 0.001, 10.0)
        assert errors == []

    def test_validate_manifest_bad_grid(self):
        errors, _ = guards.validate_manifest(BASE_PARAMS, 100, 250, 0.001, 10.0)
        assert any("power of 2" in e for e in errors)

    def test_validate_manifest_degenerate_warning(self):
        errors, warnings = guards.validate_manifest({"param_rho_vac": 0.0}, 64, 250, 0.001, 10.0)
        assert errors == []  # warning, not error
        assert any("DEGENERATE_GEOMETRY" in w for w in warnings)

    def test_validate_manifest_cfl_error(self):
        errors, _ = guards.validate_manifest({"param_rho_vac": 0.8, "param_c_affect": 1.0}, 64, 250, 0.5, 10.0)
        assert any("CFL" in e for e in errors)


# ---------------------------------------------------------------------------
# stage_simulation_manifest
# ---------------------------------------------------------------------------

class TestStaging:
    def test_happy_path(self, cfg):
        out = wt.stage_simulation_manifest(cfg, BASE_PARAMS, "HUNT_A", generation=3, seed=7,
                                           N_grid=64, T_steps=250, dt=0.001, utc_date="2026-06-18")
        assert out["staged"] is True
        assert out["validation_errors"] == []
        assert out["variant_label"] == "LOCAL-RHO"
        assert out["solver_contract_version"].endswith("LOCAL-RHO-ETDRK4-v1")
        ep = out["expected_output_path"]
        for part in ("HUNT_A", "2026-06-18", "LOCAL-RHO", "gen_0003", "_7", "artifact.h5"):
            assert part in ep
        assert os.path.exists(out["staged_manifest_path"])
        assert out["review_required"] is True

    def test_staged_manifest_has_worker_fields(self, cfg):
        out = wt.stage_simulation_manifest(cfg, BASE_PARAMS, "HUNT_A", 3, seed=7, N_grid=64, T_steps=250, dt=0.001)
        with open(out["staged_manifest_path"], encoding="utf-8") as f:
            m = json.load(f)
        assert m["config_hash"] == out["config_hash"]
        assert m["params"]["global_seed"] == 7
        assert m["params"]["simulation"]["N_grid"] == 64
        assert m["_staging"]["expected_output_path"] == out["expected_output_path"]

    def test_bad_grid_rejected(self, cfg):
        out = wt.stage_simulation_manifest(cfg, BASE_PARAMS, "HUNT_A", 3, N_grid=100, T_steps=250, dt=0.001)
        assert out["staged"] is False
        assert any("power of 2" in e for e in out["validation_errors"])

    def test_cfl_rejected(self, cfg):
        out = wt.stage_simulation_manifest(cfg, BASE_PARAMS, "HUNT_A", 3, N_grid=64, T_steps=250, dt=0.5)
        assert out["staged"] is False
        assert any("CFL" in e for e in out["validation_errors"])

    def test_degenerate_warns_but_stages(self, cfg):
        params = dict(BASE_PARAMS, param_rho_vac=0.0)
        out = wt.stage_simulation_manifest(cfg, params, "HUNT_A", 3, N_grid=64, T_steps=250, dt=0.001)
        assert out["staged"] is True
        assert any("DEGENERATE_GEOMETRY" in w for w in out["compatibility_warnings"])

    def test_multi_seed_share_config_hash(self, cfg):
        a = wt.stage_simulation_manifest(cfg, BASE_PARAMS, "HUNT_A", 3, seed=1, N_grid=64, T_steps=250, dt=0.001)
        b = wt.stage_simulation_manifest(cfg, BASE_PARAMS, "HUNT_A", 3, seed=2, N_grid=64, T_steps=250, dt=0.001)
        assert a["config_hash"] == b["config_hash"]  # seed excluded from hash
        assert a["staged_manifest_path"] != b["staged_manifest_path"]

    def test_overwrite_guard(self, cfg):
        out1 = wt.stage_simulation_manifest(cfg, BASE_PARAMS, "HUNT_A", 3, seed=7, N_grid=64, T_steps=250, dt=0.001)
        os.makedirs(os.path.dirname(out1["expected_output_path"]), exist_ok=True)
        open(out1["expected_output_path"], "w").close()  # simulate a completed artifact
        out2 = wt.stage_simulation_manifest(cfg, BASE_PARAMS, "HUNT_A", 3, seed=7, N_grid=64, T_steps=250, dt=0.001)
        assert out2["staged"] is False
        assert any("already exists" in e for e in out2["validation_errors"])
        out3 = wt.stage_simulation_manifest(cfg, BASE_PARAMS, "HUNT_A", 3, seed=7, N_grid=64, T_steps=250, dt=0.001, overwrite=True)
        assert out3["staged"] is True

    def test_compatibility_warning_for_foreign_variant(self, cfg):
        from orchestrator.schema_utils import initialize_ledger_schema
        initialize_ledger_schema(cfg.db_path)
        conn = sqlite3.connect(cfg.db_path)
        conn.execute(
            "INSERT INTO runs (config_hash, seed, status, solver_contract_version, variant_label, n_grid, hunt_name) "
            "VALUES ('other', 0, 'SUCCESS', 'IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1', 'CAUSAL-AFFECT', 64, 'HUNT_A')"
        )
        conn.commit(); conn.close()
        out = wt.stage_simulation_manifest(cfg, BASE_PARAMS, "HUNT_A", 3, seed=7, N_grid=64, T_steps=250, dt=0.001)
        assert out["staged"] is True
        assert any("isolated" in w for w in out["compatibility_warnings"])


# ---------------------------------------------------------------------------
# run_simulation_manifest (gates with injected launcher)
# ---------------------------------------------------------------------------

class TestRunManifest:
    def _stage(self, cfg):
        return wt.stage_simulation_manifest(cfg, BASE_PARAMS, "HUNT_A", 3, seed=7, N_grid=64, T_steps=250, dt=0.001)

    def test_requires_confirm(self, cfg):
        staged = self._stage(cfg)
        out = wt.run_simulation_manifest(cfg, staged["staged_manifest_path"], confirm=False)
        assert out["status"] == "REJECTED"
        assert any("confirm" in e for e in out["errors"])

    def test_missing_manifest_rejected(self, cfg):
        out = wt.run_simulation_manifest(cfg, os.path.join(cfg.root, "runs", "_staged", "nope.json"), confirm=True)
        assert out["status"] == "REJECTED"

    def test_stale_manifest_rejected(self, cfg):
        staged = self._stage(cfg)
        future = time.time() + wt.STALE_MANIFEST_SECONDS + 100
        out = wt.run_simulation_manifest(cfg, staged["staged_manifest_path"], confirm=True, now=future)
        assert out["status"] == "REJECTED"
        assert any("stale" in e for e in out["errors"])

    def test_happy_path_uses_staged_output_path(self, cfg):
        staged = self._stage(cfg)
        seen = {}
        def fake_launcher(manifest_path, output_path):
            seen["manifest"] = manifest_path
            seen["output"] = output_path
            return {"status": "PENDING_VALIDATION", "artifact_url": output_path}
        out = wt.run_simulation_manifest(cfg, staged["staged_manifest_path"], confirm=True, launcher=fake_launcher)
        assert out["status"] == "PENDING_VALIDATION"
        # path cannot be redirected: launcher received the staged expected_output_path
        assert seen["output"] == staged["expected_output_path"]
        assert out["artifact_path"] == staged["expected_output_path"]
        # audit events written
        with open(cfg.audit_log, encoding="utf-8") as f:
            stages = {json.loads(l)["stage"] for l in f if l.strip()}
        assert "dispatch" in stages and "h5_write" in stages

    def test_launcher_exception_writes_failure_audit(self, cfg):
        staged = self._stage(cfg)
        def boom(m, o):
            raise RuntimeError("gpu exploded")
        out = wt.run_simulation_manifest(cfg, staged["staged_manifest_path"], confirm=True, launcher=boom)
        assert out["status"] == "FAIL"
        with open(cfg.audit_log, encoding="utf-8") as f:
            stages = {json.loads(l)["stage"] for l in f if l.strip()}
        assert "dispatch_failure" in stages


# ---------------------------------------------------------------------------
# run_smoke_simulation
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_caps_enforced(self, cfg):
        out = wt.run_smoke_simulation(cfg, BASE_PARAMS, N_grid=64)  # > 32
        assert out["status"] == "REJECTED"
        assert any("cap" in e for e in out["errors"])

    def test_happy_path_no_ledger(self, cfg):
        from orchestrator.schema_utils import initialize_ledger_schema
        initialize_ledger_schema(cfg.db_path)
        def fake(m, o):
            return {"status": "SUCCESS", "artifact_url": o}
        out = wt.run_smoke_simulation(cfg, BASE_PARAMS, seed=0, N_grid=16, T_steps=50, launcher=fake)
        assert out["status"] == "SUCCESS"
        assert os.sep + "_smoke" + os.sep in out["artifact_path"]
        # smoke runs must NOT write to the ledger
        conn = sqlite3.connect(cfg.db_path)
        n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert n == 0


# ---------------------------------------------------------------------------
# validate_artifact
# ---------------------------------------------------------------------------

class TestValidate:
    def test_missing_artifact(self, cfg):
        out = wt.validate_artifact(cfg, os.path.join(cfg.root, "nope.h5"), os.path.join(cfg.root, "p.json"))
        assert out["status"] == "SKIPPED"
        assert out["errors"]

    def test_happy_path_with_runner(self, cfg):
        art = os.path.join(cfg.root, "rho_history_hash123.h5")
        params = os.path.join(cfg.root, "params.json")
        open(art, "w").close()
        with open(params, "w") as f:
            json.dump({"config_hash": "hash123"}, f)
        prov = os.path.join(cfg.root, "provenance_hash123.json")
        def fake_runner(a, p, o):
            with open(prov, "w") as f:
                json.dump({"spectral_fidelity": {"log_prime_sse": 0.42}}, f)
            return {"ok": True, "provenance_path": prov, "log_prime_sse": 0.42, "schema_version": "SFP-v3.2-ARCS"}
        out = wt.validate_artifact(cfg, art, params, runner=fake_runner)
        assert out["status"] == "PASSED"
        assert out["log_prime_sse"] == 0.42
        assert out["validation_schema_version"] == "SFP-v3.2-ARCS"

    def test_no_overwrite_without_force(self, cfg):
        art = os.path.join(cfg.root, "rho_history_hash123.h5")
        params = os.path.join(cfg.root, "params.json")
        open(art, "w").close()
        with open(params, "w") as f:
            json.dump({}, f)
        # pre-existing provenance (config-hash fallback name)
        open(os.path.join(cfg.root, "provenance_hash123.json"), "w").close()
        called = {"n": 0}
        def runner(a, p, o):
            called["n"] += 1
            return {"ok": True}
        out = wt.validate_artifact(cfg, art, params, output_dir=cfg.root, runner=runner)
        assert out["status"] == "SKIPPED"
        assert called["n"] == 0  # runner not invoked


# ---------------------------------------------------------------------------
# server wiring
# ---------------------------------------------------------------------------

def test_server_registers_twelve_tools():
    import asyncio
    import mcp_server.server as srv
    tools = asyncio.run(srv.mcp.list_tools())
    names = {t.name for t in tools}
    for w in ("stage_simulation_manifest", "run_simulation_manifest", "run_smoke_simulation", "validate_artifact"):
        assert w in names
    assert len(names) == 12  # 8 read + 4 write
