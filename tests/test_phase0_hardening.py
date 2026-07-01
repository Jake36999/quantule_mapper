"""
test_phase0_hardening.py
Phase 0 hardening gate tests:
  - gen_n folder creation via ensure_generation_dir()
  - Smart Sim Manifest init and per-generation updates
  - Manifest atomic write (partial writes leave no corrupt state)
  - Mode/backlog_source fields round-trip through JobManifest
  - Backlog mode validation in app.py stage endpoint
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from config_utils import (
    create_session_workspace,
    ensure_generation_dir,
    init_run_manifest,
    run_manifest_path,
    update_run_manifest_generation,
    RUN_MANIFEST_SUFFIX,
)
from orchestrator.job_manifest import JobManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(tmp_path: Path) -> dict:
    """Create a real session workspace rooted at tmp_path."""
    import uuid
    session = create_session_workspace("TEST_HUNT")
    return session


# ---------------------------------------------------------------------------
# Tests: ensure_generation_dir
# ---------------------------------------------------------------------------

class TestEnsureGenerationDir:
    def test_creates_gen_dir(self, sandbox):
        """ensure_generation_dir must proactively create gen_n/ under session_dir."""
        ptr_path = Path(".active_run_pointer.json")
        with open(ptr_path, "r", encoding="utf-8") as fh:
            ptr = json.load(fh)
        session_dir = ptr["session_dir"]

        gen_dir = ensure_generation_dir(session_dir, 1)
        assert gen_dir.is_dir(), "gen_1/ must be created"
        assert gen_dir.name == "gen_1"

    def test_idempotent(self, sandbox):
        """Calling ensure_generation_dir twice must not raise."""
        ptr_path = Path(".active_run_pointer.json")
        with open(ptr_path, "r", encoding="utf-8") as fh:
            ptr = json.load(fh)
        session_dir = ptr["session_dir"]

        gen_dir_1 = ensure_generation_dir(session_dir, 5)
        gen_dir_2 = ensure_generation_dir(session_dir, 5)
        assert gen_dir_1 == gen_dir_2

    def test_sequential_gens(self, sandbox):
        """Multiple generation dirs can coexist."""
        ptr_path = Path(".active_run_pointer.json")
        with open(ptr_path, "r", encoding="utf-8") as fh:
            ptr = json.load(fh)
        session_dir = ptr["session_dir"]

        for n in range(3):
            g = ensure_generation_dir(session_dir, n)
            assert g.is_dir(), f"gen_{n}/ must exist"


# ---------------------------------------------------------------------------
# Tests: init_run_manifest
# ---------------------------------------------------------------------------

class TestInitRunManifest:
    def test_creates_manifest_file(self, sandbox):
        ptr_path = Path(".active_run_pointer.json")
        with open(ptr_path, "r", encoding="utf-8") as fh:
            ptr = json.load(fh)

        session_info = {
            "session_name": f"{ptr['hunt_name']}_{ptr['run_id']}",
            "hunt_name": ptr["hunt_name"],
            "run_id": ptr["run_id"],
            "session_dir": ptr["session_dir"],
            "db_file": ptr["db_file"],
            "created_at": ptr.get("created_at", ""),
        }
        manifest_p = init_run_manifest(session_info)
        assert manifest_p.exists(), "Manifest file must be created"
        assert manifest_p.suffix == ".json"
        assert RUN_MANIFEST_SUFFIX.rstrip(".json") in manifest_p.name or "____manifest" in manifest_p.name

    def test_manifest_valid_json(self, sandbox):
        ptr_path = Path(".active_run_pointer.json")
        with open(ptr_path, "r", encoding="utf-8") as fh:
            ptr = json.load(fh)

        session_info = {
            "session_name": f"{ptr['hunt_name']}_{ptr['run_id']}",
            "hunt_name": ptr["hunt_name"],
            "run_id": ptr["run_id"],
            "session_dir": ptr["session_dir"],
            "db_file": ptr["db_file"],
            "created_at": ptr.get("created_at", ""),
        }
        manifest_p = init_run_manifest(session_info)
        with open(manifest_p, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert data["schema_version"] == "1.0"
        assert data["hunt_name"] == ptr["hunt_name"]
        assert isinstance(data["generations"], list)
        assert len(data["generations"]) == 0

    def test_manifest_idempotent(self, sandbox):
        """Calling init_run_manifest twice must not raise; second call overwrites cleanly."""
        ptr_path = Path(".active_run_pointer.json")
        with open(ptr_path, "r", encoding="utf-8") as fh:
            ptr = json.load(fh)

        session_info = {
            "session_name": f"{ptr['hunt_name']}_{ptr['run_id']}",
            "hunt_name": ptr["hunt_name"],
            "run_id": ptr["run_id"],
            "session_dir": ptr["session_dir"],
            "db_file": ptr["db_file"],
            "created_at": ptr.get("created_at", ""),
        }
        p1 = init_run_manifest(session_info)
        p2 = init_run_manifest(session_info)
        assert p1 == p2


# ---------------------------------------------------------------------------
# Tests: update_run_manifest_generation
# ---------------------------------------------------------------------------

class TestUpdateRunManifestGeneration:
    def _session_info(self) -> dict:
        ptr_path = Path(".active_run_pointer.json")
        with open(ptr_path, "r", encoding="utf-8") as fh:
            ptr = json.load(fh)
        return {
            "session_name": f"{ptr['hunt_name']}_{ptr['run_id']}",
            "hunt_name": ptr["hunt_name"],
            "run_id": ptr["run_id"],
            "session_dir": ptr["session_dir"],
            "db_file": ptr["db_file"],
            "created_at": ptr.get("created_at", ""),
        }

    def test_appends_generation_entry(self, sandbox):
        si = self._session_info()
        init_run_manifest(si)

        update_run_manifest_generation(
            session_dir=si["session_dir"],
            session_name=si["session_name"],
            generation=1,
            artifacts={"champion_config_hash": "abc123", "champion_fitness": 0.9},
        )

        mp = run_manifest_path(si["session_dir"], si["session_name"])
        with open(mp, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        gens = data["generations"]
        assert len(gens) == 1
        assert gens[0]["generation"] == 1
        assert gens[0]["champion_fitness"] == 0.9

    def test_multiple_generations_ordered(self, sandbox):
        si = self._session_info()
        init_run_manifest(si)

        for n in [3, 1, 2]:
            update_run_manifest_generation(
                session_dir=si["session_dir"],
                session_name=si["session_name"],
                generation=n,
                artifacts={"g": n},
            )

        mp = run_manifest_path(si["session_dir"], si["session_name"])
        with open(mp, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        generation_indices = [g["generation"] for g in data["generations"]]
        assert generation_indices == sorted(generation_indices)

    def test_idempotent_update(self, sandbox):
        """Updating the same generation twice replaces the entry, not appends."""
        si = self._session_info()
        init_run_manifest(si)

        update_run_manifest_generation(
            session_dir=si["session_dir"],
            session_name=si["session_name"],
            generation=2,
            artifacts={"fitness": 0.5},
        )
        update_run_manifest_generation(
            session_dir=si["session_dir"],
            session_name=si["session_name"],
            generation=2,
            artifacts={"fitness": 0.8},
        )

        mp = run_manifest_path(si["session_dir"], si["session_name"])
        with open(mp, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        gens = [g for g in data["generations"] if g["generation"] == 2]
        assert len(gens) == 1, "Idempotent update must not duplicate entries"
        assert gens[0]["fitness"] == 0.8

    def test_no_tmp_file_left_after_write(self, sandbox):
        """Atomic write must clean up .tmp file on success."""
        si = self._session_info()
        init_run_manifest(si)

        update_run_manifest_generation(
            session_dir=si["session_dir"],
            session_name=si["session_name"],
            generation=0,
            artifacts={"test": True},
        )

        mp = run_manifest_path(si["session_dir"], si["session_name"])
        tmp_p = mp.with_suffix(".tmp")
        assert not tmp_p.exists(), ".tmp file must not remain after atomic write"


# ---------------------------------------------------------------------------
# Tests: JobManifest mode/backlog_source round-trip
# ---------------------------------------------------------------------------

class TestJobManifestModeFields:
    def test_default_mode_is_evolution(self):
        m = JobManifest.from_params({"x": 1}, generation=0, seed=0)
        assert m.mode == "evolution"
        assert m.backlog_source is None

    def test_backlog_mode_round_trip(self):
        m = JobManifest.from_params(
            {"x": 1},
            generation=0,
            seed=0,
            mode="backlog",
            backlog_source="backlog_queue.json",
        )
        assert m.mode == "backlog"
        assert m.backlog_source == "backlog_queue.json"

        serialized = m.to_json()
        m2 = JobManifest.from_json(serialized)
        assert m2.mode == "backlog"
        assert m2.backlog_source == "backlog_queue.json"

    def test_evolution_mode_round_trip(self):
        m = JobManifest.from_params({"x": 1}, mode="evolution")
        s = m.to_json()
        m2 = JobManifest.from_json(s)
        assert m2.mode == "evolution"

    def test_mode_preserved_in_json_fields(self):
        m = JobManifest.from_params({"y": 2}, mode="backlog", backlog_source="my_queue.json")
        d = json.loads(m.to_json())
        assert d["mode"] == "backlog"
        assert d["backlog_source"] == "my_queue.json"


# ---------------------------------------------------------------------------
# Tests: app.py stage endpoint mode validation
# ---------------------------------------------------------------------------

class TestAppStageModeValidation:
    """Smoke-tests for mode validation wired into ControlStageRequest."""

    def _build_stage_payload(self, mode=None):
        base = {
            "hunt_name": "TEST_HUNT",
            "generations": 5,
            "batch_size": 2,
            "population_size": 4,
            "seeds_per_candidate": 1,
            "n_grid": 32,
            "t_steps": 100,
            "dt": 0.01,
        }
        if mode is not None:
            base["mode"] = mode
        return base

    def test_evolution_mode_accepted(self):
        from app import ControlStageRequest
        req = ControlStageRequest(**self._build_stage_payload(mode="evolution"))
        assert req.mode == "evolution"

    def test_backlog_mode_accepted(self):
        from app import ControlStageRequest
        req = ControlStageRequest(**self._build_stage_payload(mode="backlog"))
        assert req.mode == "backlog"

    def test_default_mode_is_none_in_model(self):
        """Pydantic model default is None; server normalises to 'evolution' in handler."""
        from app import ControlStageRequest
        req = ControlStageRequest(**self._build_stage_payload())
        assert req.mode is None  # Default at model level is None; server normalises

    def test_backlog_source_accepted(self):
        from app import ControlStageRequest
        payload = self._build_stage_payload(mode="backlog")
        payload["backlog_source"] = "my_custom_queue.json"
        req = ControlStageRequest(**payload)
        assert req.backlog_source == "my_custom_queue.json"
