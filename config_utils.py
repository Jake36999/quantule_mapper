import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


RUNS_ROOT_DIR = "Run_combined_data"
ACTIVE_RUN_POINTER_FILE = ".active_run_pointer.json"
SESSION_DB_BASENAME = "simulation_ledger.db"
PROVENANCE_DIRNAME = "provenance_reports"
RUN_MANIFEST_SUFFIX = "____manifest.json"  # four underscores — naming contract


class ActiveRunPointerError(RuntimeError):
    """Raised when the active run pointer is missing or invalid."""


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _pointer_path() -> Path:
    return _repo_root() / ACTIVE_RUN_POINTER_FILE


def sanitize_hunt_name(hunt_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(hunt_name or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("hunt_name must contain at least one alphanumeric character")
    return cleaned


def generation_dir_name(generation: int) -> str:
    return f"gen_{int(generation)}"


def resolve_generation_dir(session_dir: str | Path, generation: int) -> Path:
    return Path(session_dir).resolve() / generation_dir_name(generation)


def create_session_workspace(hunt_name: str) -> Dict[str, str]:
    safe_hunt_name = sanitize_hunt_name(hunt_name)
    run_id = uuid.uuid4().hex[:8]
    session_name = f"{safe_hunt_name}_{run_id}"
    runs_root = (_repo_root() / RUNS_ROOT_DIR).resolve()
    session_dir = runs_root / session_name
    db_file = session_dir / SESSION_DB_BASENAME
    provenance_dir = session_dir / PROVENANCE_DIRNAME
    gen0_dir = resolve_generation_dir(session_dir, 0)

    runs_root.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=False)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    gen0_dir.mkdir(parents=True, exist_ok=True)

    return {
        "hunt_name": safe_hunt_name,
        "run_id": run_id,
        "session_name": session_name,
        "session_dir": str(session_dir),
        "db_file": str(db_file),
        "provenance_dir": str(provenance_dir),
        "generation_root": str(session_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_active_run_pointer(payload: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(payload, dict):
        raise ActiveRunPointerError("active run pointer payload must be an object")

    required_keys = ("hunt_name", "run_id", "session_dir", "db_file", "provenance_dir", "created_at")
    missing = [key for key in required_keys if not payload.get(key)]
    if missing:
        raise ActiveRunPointerError(f"active run pointer missing required fields: {', '.join(missing)}")

    session_dir = Path(str(payload["session_dir"])).resolve()
    db_file = Path(str(payload["db_file"])).resolve()
    provenance_dir = Path(str(payload["provenance_dir"])).resolve()

    if not session_dir.exists() or not session_dir.is_dir():
        raise ActiveRunPointerError(f"active session directory not found: {session_dir}")
    if db_file.parent != session_dir:
        raise ActiveRunPointerError("db_file must be located directly inside session_dir")
    if not str(provenance_dir).startswith(str(session_dir)):
        raise ActiveRunPointerError("provenance_dir must be inside session_dir")

    return {
        "hunt_name": str(payload["hunt_name"]),
        "run_id": str(payload["run_id"]),
        "session_dir": str(session_dir),
        "db_file": str(db_file),
        "provenance_dir": str(provenance_dir),
        "created_at": str(payload["created_at"]),
    }


def read_active_run_pointer() -> Dict[str, str]:
    pointer_path = _pointer_path()
    if not pointer_path.exists():
        raise ActiveRunPointerError(f"active run pointer missing: {pointer_path}")
    try:
        with open(pointer_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ActiveRunPointerError(f"active run pointer is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ActiveRunPointerError(f"failed to read active run pointer: {exc}") from exc

    return validate_active_run_pointer(payload)


def write_active_run_pointer_atomic(payload: Dict[str, Any]) -> Dict[str, str]:
    normalized = validate_active_run_pointer(payload)
    pointer_path = _pointer_path()
    tmp_path = pointer_path.with_suffix(pointer_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)
    os.replace(tmp_path, pointer_path)
    return normalized


def clear_active_run_pointer() -> None:
    pointer_path = _pointer_path()
    if pointer_path.exists():
        pointer_path.unlink()


def resolve_active_session_paths(require_exists: bool = True) -> Dict[str, str]:
    pointer = read_active_run_pointer()
    session_dir = Path(pointer["session_dir"]).resolve()
    db_file = Path(pointer["db_file"]).resolve()
    provenance_dir = Path(pointer["provenance_dir"]).resolve()
    if require_exists:
        if not session_dir.exists():
            raise ActiveRunPointerError(f"session_dir does not exist: {session_dir}")
        if not provenance_dir.exists():
            raise ActiveRunPointerError(f"provenance_dir does not exist: {provenance_dir}")

    return {
        "hunt_name": pointer["hunt_name"],
        "run_id": pointer["run_id"],
        "session_dir": str(session_dir),
        "db_file": str(db_file),
        "provenance_dir": str(provenance_dir),
        "generation_root": str(session_dir),
        "created_at": pointer["created_at"],
    }

def generate_canonical_hash(config_dict: Dict[str, Any]) -> str:
    """
    Generate a deterministic SHA-256 hash for a configuration dictionary.
    The config is serialized with sorted keys to ensure determinism.
    """
    config_str = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()


def normalize_run_mode(mode: Any) -> str:
    """Normalize run mode values and enforce known modes."""
    normalized = str(mode or "evolution").strip().lower()
    if normalized not in {"evolution", "backlog"}:
        raise ValueError("mode must be one of: evolution, backlog")
    return normalized


def normalize_backlog_source(backlog_source: Any, default_value: str = "backlog_queue.json") -> str:
    """Normalize backlog source path string with a stable default."""
    normalized = str(backlog_source or default_value).strip()
    if not normalized:
        raise ValueError("backlog_source must be a non-empty path")
    return normalized


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        unique.append(path)
        seen.add(key)
    return unique


def _path_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_backlog_source_path(
    backlog_source: Any,
    default_value: str = "backlog_queue.json",
    require_exists: bool = False,
) -> Path:
    """
    Resolve backlog queue path with tolerant, repo-scoped fallback behavior.

    Accepted forms:
    - backlog_queue.json
    - input_configs/backlog_queue.json
    - absolute path under repo root
    """
    normalized = normalize_backlog_source(backlog_source, default_value)
    repo_root = _repo_root()
    raw = Path(normalized).expanduser()

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw.resolve())
    else:
        candidates.append((repo_root / raw).resolve())
        if raw.parent == Path("."):
            candidates.append((repo_root / "input_configs" / raw.name).resolve())
        else:
            candidates.append((repo_root / raw.name).resolve())
            candidates.append((repo_root / "input_configs" / raw.name).resolve())

    for candidate in _dedupe_paths(candidates):
        if not _path_within_root(candidate, repo_root):
            continue
        if require_exists and not candidate.exists():
            continue
        return candidate

    if require_exists:
        attempted = ", ".join(str(path) for path in _dedupe_paths(candidates))
        raise ValueError(f"backlog_source not found under repo roots. attempted: {attempted}")
    raise ValueError("backlog_source must resolve inside repository root")


# ---------------------------------------------------------------------------
# Phase 0: Run-directory and Smart Sim Manifest hardening helpers
# ---------------------------------------------------------------------------

def run_manifest_path(session_dir: "str | Path", session_name: str) -> Path:
    """Return the canonical path for the Smart Sim Manifest inside the session dir."""
    return Path(session_dir).resolve() / f"{session_name}{RUN_MANIFEST_SUFFIX}"


def ensure_generation_dir(session_dir: "str | Path", generation: int) -> Path:
    """
    Proactively create gen_<n>/ inside session_dir and return its Path.

    Raises RuntimeError if the directory cannot be created so the caller can
    surface a fatal error rather than silently spilling artifacts to the root.
    """
    gen_dir = resolve_generation_dir(session_dir, generation)
    try:
        gen_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"[Hardening] Failed to create generation directory {gen_dir}: {exc}"
        ) from exc
    return gen_dir


def init_run_manifest(session_info: Dict[str, Any]) -> Path:
    """
    Create the Smart Sim Manifest at run start inside session_dir.

    The manifest carries run-level metadata and an empty 'generations' list
    that `update_run_manifest_generation()` populates as the run progresses.
    Returns the Path to the written manifest file.

    Raises RuntimeError on write failure so callers can stop the run safely.
    """
    session_dir = Path(str(session_info["session_dir"])).resolve()
    session_name = str(
        session_info.get("session_name")
        or f"{session_info['hunt_name']}_{session_info['run_id']}"
    )
    manifest_path = run_manifest_path(session_dir, session_name)
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "session_name": session_name,
        "hunt_name": str(session_info.get("hunt_name", "")),
        "run_id": str(session_info.get("run_id", "")),
        "session_dir": str(session_dir),
        "db_file": str(session_info.get("db_file", "")),
        "created_at": str(
            session_info.get("created_at") or datetime.now(timezone.utc).isoformat()
        ),
        "generations": [],
    }
    tmp_path = manifest_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, manifest_path)
    except OSError as exc:
        raise RuntimeError(
            f"[Hardening] Failed to write run manifest {manifest_path}: {exc}"
        ) from exc
    return manifest_path


def update_run_manifest_generation(
    session_dir: "str | Path",
    session_name: str,
    generation: int,
    artifacts: Dict[str, Any],
) -> None:
    """
    Append or update a generation entry in the Smart Sim Manifest atomically.

    `artifacts` is a free-form dict of artifact paths/metadata for the
    generation (champion_h5, champion_input_configs, champion_provenance,
    merged_csv, fitness, etc.).  Existing entries for the same generation
    index are replaced (idempotent).

    Raises RuntimeError on write failure.
    """
    manifest_path = run_manifest_path(session_dir, session_name)

    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            payload = {"schema_version": "1.0", "generations": []}
    else:
        payload = {"schema_version": "1.0", "generations": []}

    # Remove existing entry for this generation so the update is idempotent
    existing_gens: list = [
        g for g in payload.get("generations", []) if g.get("generation") != generation
    ]
    entry: Dict[str, Any] = {
        "generation": generation,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **artifacts,
    }
    existing_gens.append(entry)
    existing_gens.sort(key=lambda g: g.get("generation", 0))
    payload["generations"] = existing_gens

    tmp_path = manifest_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, manifest_path)
    except OSError as exc:
        raise RuntimeError(
            f"[Hardening] Failed to update run manifest {manifest_path} for gen {generation}: {exc}"
        ) from exc
