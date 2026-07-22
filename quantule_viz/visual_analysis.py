"""Completed-run visual analysis registry, adapters, and FastAPI router."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import h5py
import numpy as np
from fastapi import APIRouter, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from skimage import measure

try:
    from orchestrator.run_identity import read_identity_group
except Exception:  # pragma: no cover - import guard for standalone tooling
    read_identity_group = None


ADAPTER_VERSION = "visual-analysis-v1"
MAX_RENDER_RESOLUTION = 96
CONFIG_HASH_RE = re.compile(r"^[0-9a-fA-F]{8,128}$")


class RenderRequest(BaseModel):
    tile: str = Field(default="density_volume")
    field: str = Field(default="rho")
    frame: int = Field(default=0, ge=0)
    resolution: int = Field(default=96, ge=8, le=128)
    threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    normalization: str = Field(default="linear")


class RerunRequest(BaseModel):
    config_hash: str
    wait_for_gpu_idle: bool = Field(default=True)
    gpu_idle_percent: int = Field(default=20, ge=0, le=100)
    gpu_idle_seconds: int = Field(default=60, ge=1, le=3600)
    gpu_wait_timeout_seconds: int = Field(default=900, ge=1, le=86400)


@dataclass(frozen=True)
class ArtifactRecord:
    run_id: str
    config_hash: str | None
    seed: int | None
    run_type: str
    status: str
    artifact_path: Path
    artifact_kind: str
    git_commit: str | None
    start_time: str | None
    end_time: str | None
    available_fields: list[str]
    validation_status: str
    frame_count: int
    grid_shape: tuple[int, int, int] | None
    source_datasets: dict[str, str]
    source_mtime: float


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _decode_h5_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_h5_string(handle: h5py.File, name: str) -> str | None:
    if name not in handle:
        return None
    try:
        raw = handle[name][0]
        return str(_decode_h5_scalar(raw))
    except Exception:
        return None


def _hash_path(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _config_hash_from_name(path: Path) -> str | None:
    match = re.search(r"(?:rho_history_|provenance_)?([0-9a-fA-F]{16,128})", path.name)
    return match.group(1) if match else None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _dataset_shape3(shape: tuple[int, ...]) -> tuple[int, int, int] | None:
    if len(shape) == 3:
        return tuple(int(v) for v in shape)
    if len(shape) == 4:
        return tuple(int(v) for v in shape[-3:])
    return None


class CompletedRunAdapter:
    """Logical adapter over current HDF5 and NPZ run-artifact layouts."""

    def __init__(self, artifact_path: Path):
        self.path = artifact_path.resolve()
        self.kind = self.path.suffix.lower().lstrip(".")
        if self.kind == "hdf5":
            self.kind = "h5"

    def inspect(self) -> ArtifactRecord | None:
        if self.kind in {"h5", "hdf5"}:
            return self._inspect_hdf5()
        if self.kind == "npz":
            return self._inspect_npz()
        return None

    def _inspect_hdf5(self) -> ArtifactRecord | None:
        try:
            with h5py.File(self.path, "r") as handle:
                identity = read_identity_group(handle) if read_identity_group else {}
                fields: list[str] = []
                sources: dict[str, str] = {}
                frame_count = 1
                grid_shape: tuple[int, int, int] | None = None

                if "rho_history" in handle:
                    fields.append("rho")
                    sources["rho"] = "/rho_history"
                    shape = tuple(int(v) for v in handle["rho_history"].shape)
                    frame_count = int(shape[0]) if len(shape) == 4 else 1
                    grid_shape = _dataset_shape3(shape)
                if "rho_final" in handle:
                    fields.append("rho")
                    sources["rho"] = "/rho_final"
                    grid_shape = grid_shape or _dataset_shape3(tuple(handle["rho_final"].shape))
                if "psi_final" in handle:
                    if "rho" not in fields:
                        fields.append("rho")
                    fields.append("phase")
                    sources.setdefault("rho", "/psi_final")
                    sources["phase"] = "/psi_final"
                    grid_shape = grid_shape or _dataset_shape3(tuple(handle["psi_final"].shape))
                if "omega_sq_final" in handle:
                    fields.append("omega2")
                    sources["omega2"] = "/omega_sq_final"
                    grid_shape = grid_shape or _dataset_shape3(tuple(handle["omega_sq_final"].shape))

                fields = sorted(set(fields))
                if not fields:
                    return None

                run_id = str(identity.get("run_id") or _hash_path(self.path))
                config_hash = str(identity.get("config_hash") or _config_hash_from_name(self.path) or "")
                seed = _safe_int(identity.get("seed"))
                status = "FAIL" if "sentinel_code" in handle else "COMPLETE"
                validation_status = "FAIL_ARTIFACT" if "sentinel_code" in handle else "UNKNOWN"
                git_commit = identity.get("git_commit")
                start_time = identity.get("utc_start")

            return ArtifactRecord(
                run_id=run_id,
                config_hash=config_hash or None,
                seed=seed,
                run_type="hdf5_completed_run",
                status=status,
                artifact_path=self.path,
                artifact_kind="h5",
                git_commit=str(git_commit) if git_commit else None,
                start_time=str(start_time) if start_time else None,
                end_time=None,
                available_fields=fields,
                validation_status=validation_status,
                frame_count=frame_count,
                grid_shape=grid_shape,
                source_datasets=sources,
                source_mtime=self.path.stat().st_mtime,
            )
        except OSError:
            return None

    def _inspect_npz(self) -> ArtifactRecord | None:
        try:
            with np.load(self.path, allow_pickle=True) as bundle:
                key = self._npz_field_key(bundle)
                if key is None:
                    return None
                arr = np.asarray(bundle[key])
                frame_count = int(arr.shape[0]) if arr.ndim == 4 else 1
                grid_shape = _dataset_shape3(tuple(arr.shape))
                fields = ["rho"]
                sources = {"rho": key}
                if np.iscomplexobj(arr):
                    fields.append("phase")
                    sources["phase"] = key
        except Exception:
            return None

        return ArtifactRecord(
            run_id=_hash_path(self.path),
            config_hash=_config_hash_from_name(self.path),
            seed=None,
            run_type="npz_frame_bundle",
            status="COMPLETE",
            artifact_path=self.path,
            artifact_kind="npz",
            git_commit=None,
            start_time=None,
            end_time=None,
            available_fields=fields,
            validation_status="UNKNOWN",
            frame_count=frame_count,
            grid_shape=grid_shape,
            source_datasets=sources,
            source_mtime=self.path.stat().st_mtime,
        )

    @staticmethod
    def _npz_field_key(bundle: np.lib.npyio.NpzFile) -> str | None:
        for key in ("frames", "psi", "rho", "rho_history"):
            if key in bundle.files:
                return key
        return str(bundle.files[0]) if bundle.files else None

    def read_field(self, field: str, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        if self.kind in {"h5", "hdf5"}:
            return self._read_hdf5_field(field, frame)
        if self.kind == "npz":
            return self._read_npz_field(field, frame)
        raise ValueError(f"Unsupported artifact kind: {self.kind}")

    def _read_hdf5_field(self, field: str, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        with h5py.File(self.path, "r") as handle:
            if field == "rho":
                if "rho_history" in handle:
                    ds = handle["rho_history"]
                    if ds.ndim == 4:
                        safe_frame = min(max(int(frame), 0), int(ds.shape[0]) - 1)
                        data = np.asarray(ds[safe_frame])
                    else:
                        safe_frame = 0
                        data = np.asarray(ds[()])
                    return self._as_rho(data), {"source_dataset": "/rho_history", "frame": safe_frame}
                if "rho_final" in handle:
                    return self._as_rho(np.asarray(handle["rho_final"][()])), {"source_dataset": "/rho_final", "frame": 0}
                if "psi_final" in handle:
                    return self._as_rho(np.asarray(handle["psi_final"][()])), {"source_dataset": "/psi_final", "frame": 0}
            if field == "phase" and "psi_final" in handle:
                return np.angle(np.asarray(handle["psi_final"][()])).astype(np.float32), {"source_dataset": "/psi_final", "frame": 0}
            if field == "omega2" and "omega_sq_final" in handle:
                return np.asarray(handle["omega_sq_final"][()], dtype=np.float32), {"source_dataset": "/omega_sq_final", "frame": 0}
        raise ValueError(f"Logical field {field!r} is not available in {self.path.name}")

    def _read_npz_field(self, field: str, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        with np.load(self.path, allow_pickle=True) as bundle:
            key = self._npz_field_key(bundle)
            if key is None:
                raise ValueError("NPZ bundle contains no arrays")
            arr = np.asarray(bundle[key])
            safe_frame = min(max(int(frame), 0), int(arr.shape[0]) - 1) if arr.ndim == 4 else 0
            data = arr[safe_frame] if arr.ndim == 4 else arr
            if field == "rho":
                return self._as_rho(data), {"source_dataset": key, "frame": safe_frame}
            if field == "phase" and np.iscomplexobj(data):
                return np.angle(data).astype(np.float32), {"source_dataset": key, "frame": safe_frame}
        raise ValueError(f"Logical field {field!r} is not available in {self.path.name}")

    @staticmethod
    def _as_rho(data: np.ndarray) -> np.ndarray:
        return (np.abs(data) ** 2 if np.iscomplexobj(data) else np.asarray(data)).astype(np.float32)

    def read_metrics(self) -> dict[str, Any]:
        if self.kind not in {"h5", "hdf5"}:
            return {"series": [], "stability_metrics": None}
        with h5py.File(self.path, "r") as handle:
            series = []
            if "telemetry" in handle:
                telemetry = handle["telemetry"]
                keys = [key for key in ("step", "energy", "C_invariant", "max_amplitude") if key in telemetry]
                if keys:
                    arrays = {key: np.asarray(telemetry[key][()]).reshape(-1) for key in keys}
                    count = min(len(arr) for arr in arrays.values())
                    for idx in range(count):
                        row = {key: float(arrays[key][idx]) for key in keys}
                        if "step" in row:
                            row["step"] = int(row["step"])
                        series.append(row)
            stability_metrics = None
            raw_stability = _first_h5_string(handle, "stability_metrics")
            if raw_stability:
                try:
                    stability_metrics = json.loads(raw_stability)
                except json.JSONDecodeError:
                    stability_metrics = raw_stability
        return {"series": series, "stability_metrics": stability_metrics}


class VisualRunRegistry:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS visual_runs (
                    run_id TEXT PRIMARY KEY,
                    config_hash TEXT,
                    git_commit TEXT,
                    seed INTEGER,
                    run_type TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    status TEXT,
                    artifact_path TEXT NOT NULL,
                    artifact_kind TEXT NOT NULL,
                    rolling_buffer_expiry TEXT,
                    available_fields TEXT NOT NULL,
                    validation_status TEXT,
                    frame_count INTEGER,
                    grid_shape TEXT,
                    source_datasets TEXT,
                    source_mtime REAL,
                    discovered_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_visual_runs_config_hash ON visual_runs(config_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_visual_runs_discovered ON visual_runs(discovered_at DESC)")

    def upsert(self, record: ArtifactRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO visual_runs (
                    run_id, config_hash, git_commit, seed, run_type, start_time, end_time,
                    status, artifact_path, artifact_kind, rolling_buffer_expiry,
                    available_fields, validation_status, frame_count, grid_shape,
                    source_datasets, source_mtime, discovered_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    config_hash=excluded.config_hash,
                    git_commit=excluded.git_commit,
                    seed=excluded.seed,
                    run_type=excluded.run_type,
                    start_time=excluded.start_time,
                    end_time=excluded.end_time,
                    status=excluded.status,
                    artifact_path=excluded.artifact_path,
                    artifact_kind=excluded.artifact_kind,
                    available_fields=excluded.available_fields,
                    validation_status=excluded.validation_status,
                    frame_count=excluded.frame_count,
                    grid_shape=excluded.grid_shape,
                    source_datasets=excluded.source_datasets,
                    source_mtime=excluded.source_mtime,
                    discovered_at=excluded.discovered_at
                """,
                (
                    record.run_id,
                    record.config_hash,
                    record.git_commit,
                    record.seed,
                    record.run_type,
                    record.start_time,
                    record.end_time,
                    record.status,
                    str(record.artifact_path),
                    record.artifact_kind,
                    None,
                    _json_dumps(record.available_fields),
                    record.validation_status,
                    record.frame_count,
                    _json_dumps(record.grid_shape) if record.grid_shape else None,
                    _json_dumps(record.source_datasets),
                    record.source_mtime,
                    time.time(),
                ),
            )

    def list_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM visual_runs
                ORDER BY COALESCE(end_time, start_time) DESC, source_mtime DESC, discovered_at DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM visual_runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def find_by_config_hash(self, config_hash: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM visual_runs WHERE config_hash = ? ORDER BY discovered_at DESC",
                (config_hash,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key in ("available_fields", "grid_shape", "source_datasets"):
            if data.get(key):
                data[key] = json.loads(data[key])
        data["hdf5_path"] = data["artifact_path"] if data["artifact_kind"] in {"h5", "hdf5"} else None
        data["completed_at"] = data.get("end_time") or data.get("start_time")
        if not data["completed_at"] and data.get("source_mtime") is not None:
            try:
                data["completed_at"] = datetime.fromtimestamp(float(data["source_mtime"]), timezone.utc).isoformat()
            except (TypeError, ValueError, OSError):
                data["completed_at"] = None
        return data


def _unique_existing_roots(roots: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if not resolved.exists() or not resolved.is_dir():
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def discover_runs(registry: VisualRunRegistry, roots: Iterable[Path]) -> list[dict[str, Any]]:
    for root in _unique_existing_roots(roots):
        for suffix in ("*.h5", "*.hdf5", "*.npz"):
            for path in root.rglob(suffix):
                if not path.is_file():
                    continue
                record = CompletedRunAdapter(path).inspect()
                if record is not None:
                    registry.upsert(record)
    return registry.list_runs()


def _downsample(field: np.ndarray, resolution: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(field)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D field, got shape {arr.shape}")
    max_dim = max(int(v) for v in arr.shape)
    stride = max(1, int(math.ceil(max_dim / float(min(resolution, MAX_RENDER_RESOLUTION)))))
    return arr[::stride, ::stride, ::stride].astype(np.float32, copy=False), stride


def _normalize(field: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, float]]:
    finite = np.asarray(field, dtype=np.float32)
    finite = np.nan_to_num(finite, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(finite)) if finite.size else 0.0
    hi = float(np.max(finite)) if finite.size else 0.0
    if mode == "none" or hi <= lo:
        return finite.astype(np.float32, copy=False), {"min": lo, "max": hi}
    scaled = (finite - lo) / (hi - lo)
    return scaled.astype(np.float32, copy=False), {"min": lo, "max": hi}


def _cache_key(run: dict[str, Any], req: RenderRequest) -> str:
    payload = {
        "adapter_version": ADAPTER_VERSION,
        "run_id": run["run_id"],
        "artifact_path": run["artifact_path"],
        "source_mtime": run.get("source_mtime"),
        "tile": req.tile,
        "field": req.field,
        "frame": req.frame,
        "resolution": req.resolution,
        "threshold": req.threshold,
        "normalization": req.normalization,
    }
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()[:24]


def _asset_url(cache_key: str, asset_name: str) -> str:
    return f"/api/visual/cache/{cache_key}/{asset_name}"


def render_manifest(run: dict[str, Any], req: RenderRequest, cache_root: Path) -> dict[str, Any]:
    if req.field not in run.get("available_fields", []):
        raise ValueError(f"Field {req.field!r} is not available for run {run['run_id']}")

    cache_key = _cache_key(run, req)
    out_dir = cache_root / cache_key
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    adapter = CompletedRunAdapter(Path(run["artifact_path"]))
    field, field_meta = adapter.read_field(req.field, req.frame)
    reduced, stride = _downsample(field, req.resolution)
    normalized, scalar_range = _normalize(reduced, req.normalization)

    out_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, str] = {}
    derived = req.field != "rho" or req.tile in {"density_isosurface", "orthogonal_slices"}

    if req.tile == "density_volume":
        (out_dir / "volume.f32").write_bytes(np.ascontiguousarray(normalized, dtype=np.float32).ravel().tobytes())
        assets["volume"] = _asset_url(cache_key, "volume.f32")
    elif req.tile == "density_isosurface":
        verts, faces, normals, _values = measure.marching_cubes(normalized, level=float(req.threshold))
        polys = np.column_stack(
            (np.full((faces.shape[0], 1), 3, dtype=np.uint32), faces.astype(np.uint32))
        )
        (out_dir / "points.f32").write_bytes(np.ascontiguousarray(verts.astype(np.float32)).ravel().tobytes())
        (out_dir / "normals.f32").write_bytes(np.ascontiguousarray(normals.astype(np.float32)).ravel().tobytes())
        (out_dir / "polys.u32").write_bytes(np.ascontiguousarray(polys.astype(np.uint32)).ravel().tobytes())
        assets.update(
            {
                "points": _asset_url(cache_key, "points.f32"),
                "normals": _asset_url(cache_key, "normals.f32"),
                "polys": _asset_url(cache_key, "polys.u32"),
            }
        )
    elif req.tile == "orthogonal_slices":
        z, y, x = normalized.shape
        slices = {
            "xy": normalized[z // 2, :, :].tolist(),
            "xz": normalized[:, y // 2, :].tolist(),
            "yz": normalized[:, :, x // 2].tolist(),
        }
        (out_dir / "slices.json").write_text(_json_dumps(slices), encoding="utf-8")
        assets["slices"] = _asset_url(cache_key, "slices.json")
    elif req.tile == "validation_dashboard":
        pass
    else:
        raise ValueError(f"Unsupported visual-analysis tile: {req.tile}")

    manifest = {
        "cache_key": cache_key,
        "adapter_version": ADAPTER_VERSION,
        "run_id": run["run_id"],
        "config_hash": run.get("config_hash"),
        "tile": req.tile,
        "field": req.field,
        "frame": int(field_meta.get("frame", req.frame)),
        "frame_count": run.get("frame_count", 1),
        "shape": list(normalized.shape),
        "dtype": "float32",
        "normalization": req.normalization,
        "scalar_range": scalar_range,
        "downsampling": {"stride": stride, "max_resolution": req.resolution},
        "threshold": req.threshold if req.tile == "density_isosurface" else None,
        "source_dataset": field_meta.get("source_dataset"),
        "source_artifact": run["artifact_path"],
        "assets": assets,
        "labels": {
            "raw_or_derived": "DERIVED VISUALISATION - NOT RAW FIELD" if derived else "RAW FIELD DOWNSAMPLED FOR DISPLAY",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def _query_nvidia_gpu_utilization() -> int | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    values: list[int] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(int(float(stripped)))
        except ValueError:
            continue
    return max(values) if values else None


def wait_for_gpu_idle(
    *,
    max_percent: int = 20,
    stable_seconds: int = 60,
    timeout_seconds: int = 900,
    sample_interval_seconds: float = 5.0,
) -> dict[str, Any]:
    started = time.time()
    idle_since: float | None = None
    last_utilization: int | None = None

    while True:
        now = time.time()
        utilization = _query_nvidia_gpu_utilization()
        last_utilization = utilization
        if utilization is None:
            return {
                "status": "unavailable",
                "message": "GPU utilization monitoring unavailable; nvidia-smi was not readable.",
                "last_utilization_percent": None,
                "waited_seconds": round(now - started, 1),
            }

        if utilization < max_percent:
            idle_since = idle_since or now
            idle_for = now - idle_since
            if idle_for >= stable_seconds:
                return {
                    "status": "idle",
                    "last_utilization_percent": utilization,
                    "idle_for_seconds": round(idle_for, 1),
                    "waited_seconds": round(now - started, 1),
                    "threshold_percent": max_percent,
                    "required_idle_seconds": stable_seconds,
                }
        else:
            idle_since = None

        if now - started >= timeout_seconds:
            return {
                "status": "timeout",
                "last_utilization_percent": last_utilization,
                "waited_seconds": round(now - started, 1),
                "threshold_percent": max_percent,
                "required_idle_seconds": stable_seconds,
                "message": "Timed out waiting for GPU utilization to remain below threshold.",
            }

        time.sleep(sample_interval_seconds)


def create_visual_router(
    *,
    project_root: Path,
    data_dir_getter: Callable[[], Path],
    active_session_dir_getter: Callable[[], Path | None],
    runs_root_name: str,
) -> APIRouter:
    router = APIRouter(prefix="/api/visual", tags=["visual-analysis"])
    cache_root = project_root / ".cache" / "render_service"
    registry = VisualRunRegistry(cache_root / "visual_registry.sqlite")

    def roots() -> list[Path]:
        active = active_session_dir_getter()
        candidates = [
            data_dir_getter(),
            project_root / runs_root_name,
            project_root / "archive_runs",
            project_root / "quantule_viz" / "outputs",
        ]
        if active is not None:
            candidates.insert(0, active)
        return candidates

    @router.get("/runs")
    async def list_visual_runs(limit: int = 50, include_artifacts: bool = False):
        runs = await run_in_threadpool(discover_runs, registry, roots())
        if not include_artifacts:
            runs = [run for run in runs if run.get("artifact_kind") in {"h5", "hdf5"}]
        return {"status": "success", "runs": runs[: max(1, min(int(limit), 500))]}

    @router.get("/runs/{run_id}")
    async def get_visual_run(run_id: str):
        await run_in_threadpool(discover_runs, registry, roots())
        run = registry.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="visual run not found")
        return {"status": "success", "run": run}

    @router.get("/runs/{run_id}/metrics")
    async def get_visual_metrics(run_id: str):
        await run_in_threadpool(discover_runs, registry, roots())
        run = registry.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="visual run not found")
        metrics = await run_in_threadpool(CompletedRunAdapter(Path(run["artifact_path"])).read_metrics)
        return {"status": "success", "run_id": run_id, **metrics}

    @router.post("/runs/{run_id}/render")
    async def render_visual_run(run_id: str, req: RenderRequest):
        await run_in_threadpool(discover_runs, registry, roots())
        run = registry.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="visual run not found")
        try:
            manifest = await run_in_threadpool(render_manifest, run, req, cache_root)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "success", "manifest": manifest}

    @router.get("/cache/{cache_key}/{asset_name}")
    async def get_cache_asset(cache_key: str, asset_name: str):
        if not re.fullmatch(r"[0-9a-f]{24}", cache_key):
            raise HTTPException(status_code=400, detail="invalid cache key")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", asset_name):
            raise HTTPException(status_code=400, detail="invalid asset name")
        asset_path = (cache_root / cache_key / asset_name).resolve()
        try:
            asset_path.relative_to(cache_root.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid cache path") from exc
        if not asset_path.exists() or not asset_path.is_file():
            raise HTTPException(status_code=404, detail="cache asset not found")
        media = "application/json" if asset_path.suffix == ".json" else "application/octet-stream"
        return Response(content=asset_path.read_bytes(), media_type=media)

    @router.post("/rerun")
    async def request_visual_rerun(req: RerunRequest):
        config_hash = req.config_hash.strip()
        if not CONFIG_HASH_RE.fullmatch(config_hash):
            raise HTTPException(status_code=400, detail="invalid config_hash format")
        await run_in_threadpool(discover_runs, registry, roots())
        matches = registry.find_by_config_hash(config_hash)
        if not matches:
            raise HTTPException(status_code=404, detail="config_hash not found in visual registry")
        gpu_gate = None
        if req.wait_for_gpu_idle:
            gpu_gate = await run_in_threadpool(
                wait_for_gpu_idle,
                max_percent=req.gpu_idle_percent,
                stable_seconds=req.gpu_idle_seconds,
                timeout_seconds=req.gpu_wait_timeout_seconds,
            )
            if gpu_gate.get("status") == "timeout":
                return {
                    "status": "waiting_for_gpu_idle",
                    "config_hash": config_hash,
                    "message": gpu_gate.get("message") or "GPU is still busy; rerun was not queued.",
                    "gpu_gate": gpu_gate,
                    "matched_run_ids": [row["run_id"] for row in matches],
                }
        return {
            "status": "requires_reproduction_config",
            "config_hash": config_hash,
            "message": (
                "The original config hash is known, but V1 will only queue exact reruns "
                "when an immutable saved configuration payload is available."
            ),
            "gpu_gate": gpu_gate,
            "matched_run_ids": [row["run_id"] for row in matches],
        }

    return router
