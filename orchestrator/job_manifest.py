"""Job manifest serialization and management."""

from __future__ import annotations

from typing import Any, Dict, Optional

from orchestrator.contracts import JobManifest as _JobManifest


class JobManifest(_JobManifest):
    """Compatibility wrapper exposing legacy helper methods."""

    @staticmethod
    def from_params(
        params: Dict[str, Any],
        generation: int = 0,
        seed: int = 0,
        origin: str = "NATURAL",
        staged_path: Optional[str] = None,
        staged_at: Optional[str] = None,
        staged_config_hash: Optional[str] = None,
        hunt_name: Optional[str] = None,
        session_dir: Optional[str] = None,
        session_db_path: Optional[str] = None,
        generation_dir: Optional[str] = None,
        pointer_snapshot_version: Optional[str] = None,
        mode: Optional[str] = None,
        backlog_source: Optional[str] = None,
    ) -> "JobManifest":
        model = _JobManifest.from_params(
            params=params,
            generation=generation,
            seed=seed,
            origin=origin,
            staged_path=staged_path,
            staged_at=staged_at,
            staged_config_hash=staged_config_hash,
            hunt_name=hunt_name,
            session_dir=session_dir,
            session_db_path=session_db_path,
            generation_dir=generation_dir,
            pointer_snapshot_version=pointer_snapshot_version,
            mode=mode,
            backlog_source=backlog_source,
        )
        return JobManifest.model_validate(model.model_dump())

    def to_json(self) -> str:
        """Serialize manifest to JSON string."""
        return self.model_dump_json()

    @staticmethod
    def from_json(json_str: str) -> "JobManifest":
        """Deserialize manifest from JSON string."""
        return JobManifest.model_validate_json(json_str)
