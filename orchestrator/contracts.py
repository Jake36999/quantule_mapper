from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


SOLVER_CONTRACT_VERSION = "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1"

# Contract version for A-coupled runs (vacuum-reference modulation, γ_A ≠ 0).
# Not yet active — reserved so result_processor can detect and segregate these
# runs before the coupling is implemented.  See docs/DATA_CONTRACT.md §3.1.
CAUSAL_AFFECT_CONTRACT_VERSION = "IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1"
ADDITIVE_POT_CONTRACT_VERSION = "IRER-SNCGL-ADDITIVE-POT-ETDRK4-v1"

PROVISIONAL_SSE_THRESHOLD = 1.0

# Canonical physics-parameter defaults — single source of truth, imported by the
# solver so the geometry reference and the oscillator frequency defaults cannot
# silently diverge (resolves the param_rho_vac dual-role / default conflict; see
# docs/IRER_MATH_SANITY_CHECK.md §7.2).
#   param_rho_vac -> conformal reference density in  Omega^2 = (rho_vac / rho)^a
#   param_omega0  -> vacuum oscillator frequency in  L_k = ... + i*omega0
DEFAULT_PARAM_RHO_VAC = 1.0
DEFAULT_PARAM_OMEGA0 = 1.0

RESULT_STATES = {
    "VALIDATED_PROVISIONAL": "Passed SSE threshold; refinement not yet run.",
    "REFINEMENT_PENDING": "Refinement variants dispatched; awaiting results.",
    "REFINEMENT_STABLE": "RefinementAudit passed; champion-eligible.",
    "REFINEMENT_REJECTED": "RefinementAudit failed; not champion-eligible.",
    "PHYSICS_UNCERTIFIED": "Solver contract missing or wrong version.",
}


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    N_grid: int = 64
    L_domain: float = 10.0
    T_steps: int = 250
    dt: float = 0.001
    collapse_threshold: float | None = None

    @field_validator("N_grid")
    @classmethod
    def _validate_n_grid(cls, value: int) -> int:
        if not _is_power_of_two(value):
            raise ValueError("N_grid must be a positive power of 2")
        return value

    @field_validator("T_steps")
    @classmethod
    def _validate_t_steps(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("T_steps must be positive")
        return value

    @field_validator("L_domain", "dt")
    @classmethod
    def _validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return value


class JobManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_hash: str
    job_id: str
    generation: int
    seed: int = 0
    params: Dict[str, Any] = Field(default_factory=dict)
    origin: str = "NATURAL"
    staged_path: str | None = None
    staged_at: str | None = None
    staged_config_hash: str | None = None
    hunt_name: str | None = None
    session_dir: str | None = None
    session_db_path: str | None = None
    generation_dir: str | None = None
    pointer_snapshot_version: str | None = None
    mode: Literal["evolution", "backlog"] = "evolution"
    backlog_source: str | None = None

    @model_validator(mode="after")
    def _align_params(self) -> "JobManifest":
        params = dict(self.params)
        params.setdefault("config_hash", self.config_hash)
        params.setdefault("generation", self.generation)
        params.setdefault("origin", self.origin)
        if self.staged_path:
            params.setdefault("staged_path", self.staged_path)
        if self.staged_at:
            params.setdefault("staged_at", self.staged_at)
        if self.staged_config_hash:
            params.setdefault("staged_config_hash", self.staged_config_hash)

        if "simulation" in params and isinstance(params["simulation"], dict):
            sim_cfg = SimulationConfig.model_validate(params["simulation"])
            params["simulation"] = sim_cfg.model_dump()

        self.params = params
        return self

    @classmethod
    def from_params(
        cls,
        params: Dict[str, Any],
        generation: int = 0,
        seed: int = 0,
        origin: str = "NATURAL",
        staged_path: str | None = None,
        staged_at: str | None = None,
        staged_config_hash: str | None = None,
        hunt_name: str | None = None,
        session_dir: str | None = None,
        session_db_path: str | None = None,
        generation_dir: str | None = None,
        pointer_snapshot_version: str | None = None,
        mode: str | None = None,
        backlog_source: str | None = None,
    ) -> "JobManifest":
        canonical = json.dumps(params, sort_keys=True, default=str)
        config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        job_id = str(uuid.uuid4())[:16]
        _raw_mode = "evolution" if mode is None else str(mode)
        if _raw_mode not in ("evolution", "backlog"):
            raise ValueError(f"mode must be 'evolution' or 'backlog', got {_raw_mode!r}")
        mode_value: Literal["evolution", "backlog"] = _raw_mode  # type: ignore[assignment]

        return cls(
            config_hash=config_hash,
            job_id=job_id,
            generation=int(generation),
            seed=int(seed),
            params=dict(params),
            origin=str(origin),
            staged_path=staged_path,
            staged_at=staged_at,
            staged_config_hash=staged_config_hash,
            hunt_name=hunt_name,
            session_dir=session_dir,
            session_db_path=session_db_path,
            generation_dir=generation_dir,
            pointer_snapshot_version=pointer_snapshot_version,
            mode=mode_value,
            backlog_source=backlog_source,
        )


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    generation: int
    config_hash: str
    artifact_url: str
    status: Literal["SUCCESS", "FAIL", "PENDING_VALIDATION"]
    config: Dict[str, Any] = Field(default_factory=dict)
    provenance_path: str | None = None
    origin: str | None = None
    staged_path: str | None = None
    staged_at: str | None = None
    staged_config_hash: str | None = None
    reason: str | None = None
    error: str | None = None
    params_path: str | None = None
    energy_sparkline: Optional[List[float]] = None
    persist_hunter: bool = False

    @model_validator(mode="after")
    def _validate_result_shape(self) -> "JobResult":
        if self.status == "SUCCESS" and not self.provenance_path and not self.params_path:
            raise ValueError("SUCCESS result requires provenance_path or params_path")
        if self.status == "FAIL" and not (self.reason or self.error):
            # Keep permissive fallback for legacy callers, but prefer explicit reason.
            self.reason = "worker_reported_fail"
        return self


class WorkerTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str
    heartbeat_epoch: float
    in_flight_claims: int = 0
    state: Literal["active", "stale"] = "active"


def parse_job_manifest_payload(payload: str | Dict[str, Any] | JobManifest) -> JobManifest:
    if isinstance(payload, JobManifest):
        return payload
    if isinstance(payload, str):
        return JobManifest.model_validate_json(payload)
    return JobManifest.model_validate(payload)


def parse_job_result_payload(payload: str | Dict[str, Any] | JobResult) -> JobResult:
    if isinstance(payload, JobResult):
        return payload
    if isinstance(payload, str):
        return JobResult.model_validate_json(payload)
    return JobResult.model_validate(payload)


def is_valid_manifest_payload(payload: str | Dict[str, Any]) -> bool:
    try:
        parse_job_manifest_payload(payload)
        return True
    except ValidationError:
        return False


def is_valid_result_payload(payload: str | Dict[str, Any]) -> bool:
    try:
        parse_job_result_payload(payload)
        return True
    except ValidationError:
        return False
