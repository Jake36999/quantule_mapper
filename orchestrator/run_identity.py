"""
orchestrator/run_identity.py

Canonical, dependency-light run-identity contract for the IRER simulation stack.

This module deliberately imports **no** cupy and does **no** HDF5 work at module
load time, so it can be imported and unit-tested anywhere: inside the solver hot
path, the ledger writer, the validation pipeline, and (read-only) MCP tools.
h5py / numpy are imported lazily, only inside the functions that touch HDF5.

It is the single source of truth for:
  * the discriminator identity tuple that distinguishes incompatible runs,
  * variant_label / affect_topology / affect_strength derivation from params,
  * the compatibility key used to gate ranking of two runs,
  * collision-free provenance filename derivation,
  * the k=0 zero-mode projection used by the affect-field solver.

See docs/DATA_CONTRACT.md and docs/OUTPUT_HIERARCHY.md.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

# Mirror of orchestrator.contracts version strings.  Imported when available so
# there is one true definition; literals are a safe fallback if this module is
# loaded outside the package (e.g. a standalone MCP tool).
try:  # pragma: no cover - exercised indirectly
    from orchestrator.contracts import (
        SOLVER_CONTRACT_VERSION as _LOCAL_RHO_CONTRACT,
        CAUSAL_AFFECT_CONTRACT_VERSION as _CAUSAL_AFFECT_CONTRACT,
        ADDITIVE_POT_CONTRACT_VERSION as _ADDITIVE_POT_CONTRACT,
    )
except Exception:  # pragma: no cover
    _LOCAL_RHO_CONTRACT = "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1"
    _CAUSAL_AFFECT_CONTRACT = "IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1"
    _ADDITIVE_POT_CONTRACT = "IRER-SNCGL-ADDITIVE-POT-ETDRK4-v1"

CONTRACT_DC_VERSION = "DC-v1.0"

# Ordered identity fields every artifact / ledger row / provenance file must carry.
REQUIRED_IDENTITY_FIELDS: Tuple[str, ...] = (
    "hunt_name",
    "run_id",
    "utc_start",
    "generation",
    "config_hash",
    "seed",
    "solver_contract_version",
    "variant_label",
    "affect_topology",
    "affect_strength",
    "git_commit",
    "N_grid",
    "dt",
    "T_steps",
    "gpu_backend",
)

# Fields that MUST match before two runs may be ranked against each other.
# Two runs with different values here test different physics or different
# spectral resolutions and are therefore not comparable by SSE alone.
COMPATIBILITY_FIELDS: Tuple[str, ...] = (
    "solver_contract_version",
    "variant_label",
    "affect_topology",
    "N_grid",
)


# ---------------------------------------------------------------------------
# Variant / topology derivation (params -> labels)
# ---------------------------------------------------------------------------

def affect_strength_for_params(params: Dict[str, Any]) -> float:
    """Coupling strength gamma_A.  0.0 == baseline (A field passive)."""
    try:
        return float(params.get("param_affect_coupling", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def affect_topology_for_params(params: Dict[str, Any]) -> str:
    """
    Returns one of: "none", "vacuum_ref", "additive_potential".

    gamma_A == 0 always yields "none" regardless of the declared topology so the
    baseline cannot be mislabelled.  When gamma_A != 0 the topology comes from
    param_affect_topology (default vacuum_ref, the theory-primary channel).
    """
    if affect_strength_for_params(params) == 0.0:
        return "none"
    topo = str(params.get("param_affect_topology", "vacuum_ref")).strip().lower()
    if topo in ("vacuum_ref", "vacuum_reference", "vacuum-reference", ""):
        return "vacuum_ref"
    if topo in ("additive", "additive_potential", "additive-potential"):
        return "additive_potential"
    return topo


def variant_label_for_params(params: Dict[str, Any]) -> str:
    return {
        "none": "LOCAL-RHO",
        "vacuum_ref": "CAUSAL-AFFECT",
        "additive_potential": "ADDITIVE-POT",
    }.get(affect_topology_for_params(params), "UNKNOWN")


def solver_contract_version_for_params(params: Dict[str, Any]) -> str:
    return {
        "none": _LOCAL_RHO_CONTRACT,
        "vacuum_ref": _CAUSAL_AFFECT_CONTRACT,
        "additive_potential": _ADDITIVE_POT_CONTRACT,
    }.get(affect_topology_for_params(params), "UNKNOWN")


# ---------------------------------------------------------------------------
# Runtime probes
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(repo_root: Optional[str] = None) -> str:
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def detect_backend() -> str:
    """'cupy' on a GPU box, 'numpy' otherwise.  Never raises."""
    try:
        import cupy  # noqa: F401
        return "cupy"
    except Exception:
        return "numpy"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Identity construction
# ---------------------------------------------------------------------------

def build_identity(
    *,
    config_hash: str,
    seed: int,
    generation: Optional[int],
    N_grid: int,
    dt: float,
    T_steps: int,
    params: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    hunt_name: Optional[str] = None,
    utc_start: Optional[str] = None,
    gpu_backend: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the canonical identity dict for one run.  All discriminator fields are
    derived here so every write path (HDF5, ledger, provenance, audit) agrees.
    """
    params = params or {}
    return {
        "hunt_name": str(hunt_name or ""),
        "run_id": str(run_id or ""),
        "utc_start": utc_start or utc_now_iso(),
        "generation": int(generation) if generation is not None else -1,
        "config_hash": str(config_hash or ""),
        "seed": int(seed) if seed is not None else 0,
        "solver_contract_version": solver_contract_version_for_params(params),
        "variant_label": variant_label_for_params(params),
        "affect_topology": affect_topology_for_params(params),
        "affect_strength": affect_strength_for_params(params),
        "git_commit": git_commit(repo_root),
        "N_grid": int(N_grid),
        "dt": float(dt),
        "T_steps": int(T_steps),
        "gpu_backend": str(gpu_backend or detect_backend()),
        "contract_dc_version": CONTRACT_DC_VERSION,
    }


def missing_identity_fields(identity: Dict[str, Any]) -> Tuple[str, ...]:
    """Required identity fields that are absent or empty."""
    missing = []
    for field in REQUIRED_IDENTITY_FIELDS:
        if field not in identity:
            missing.append(field)
            continue
        value = identity[field]
        if value is None or (isinstance(value, str) and value == ""):
            missing.append(field)
    return tuple(missing)


# ---------------------------------------------------------------------------
# Compatibility gate
# ---------------------------------------------------------------------------

def compatibility_key(identity: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(str(identity.get(field, "")) for field in COMPATIBILITY_FIELDS)


def are_rankable(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True only if two runs share solver contract, variant, topology, and grid."""
    return compatibility_key(a) == compatibility_key(b)


# ---------------------------------------------------------------------------
# HDF5 /identity group I/O  (h5py imported lazily)
# ---------------------------------------------------------------------------

def write_identity_group(h5file, identity: Dict[str, Any]) -> None:
    """Write the identity dict into an /identity HDF5 group (datasets + attrs)."""
    import numpy as np

    grp = h5file.require_group("identity")
    for key, value in identity.items():
        # Remove a stale dataset if re-writing.
        if key in grp:
            del grp[key]
        if isinstance(value, bool):
            grp.create_dataset(key, data=np.int64(1 if value else 0))
        elif isinstance(value, int):
            grp.create_dataset(key, data=np.int64(value))
        elif isinstance(value, float):
            grp.create_dataset(key, data=np.float64(value))
        else:
            grp.create_dataset(key, data=np.bytes_(str(value).encode("utf-8")))
        # Mirror as an attribute so generic HDF5 browsers (h5dump/HDFView) show it.
        try:
            grp.attrs[key] = value
        except Exception:
            grp.attrs[key] = str(value)


def read_identity_group(h5file) -> Dict[str, Any]:
    """Read an /identity group back to a plain dict (bytes decoded to str)."""
    out: Dict[str, Any] = {}
    if "identity" not in h5file:
        return out
    grp = h5file["identity"]
    for key in grp.keys():
        value = grp[key][()]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif hasattr(value, "item"):
            value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Collision-free provenance naming
# ---------------------------------------------------------------------------

def provenance_filename(
    config_hash: str,
    seed: Optional[int] = None,
    run_id: Optional[str] = None,
    utc_date: Optional[str] = None,
) -> str:
    """
    Collision-free provenance filename.

    Falls back to the legacy ``provenance_{config_hash}.json`` only when no
    discriminator is available, so historical artifacts still resolve.
    """
    if seed is None and not run_id:
        return f"provenance_{config_hash}.json"
    parts = [f"provenance_{config_hash}"]
    if seed is not None:
        parts.append(f"seed{int(seed)}")
    if utc_date:
        parts.append(str(utc_date).replace("-", "").replace(":", ""))
    if run_id:
        parts.append(str(run_id)[:8])
    return "_".join(parts) + ".json"


def provenance_path_for_artifact(output_dir: str, artifact_path: Optional[str], config_hash: str) -> str:
    """
    Derive the provenance path for an artifact.  If the artifact carries an
    /identity group, the seed / run_id / utc_start are folded into the filename
    so multi-seed and re-run results never overwrite each other.  Both the
    writer (validation_pipeline) and readers (result_processor) call this so the
    name is always consistent.
    """
    seed: Optional[int] = None
    run_id: Optional[str] = None
    utc_date: Optional[str] = None
    try:
        import h5py

        if artifact_path and os.path.exists(artifact_path):
            with h5py.File(artifact_path, "r") as f:
                ident = read_identity_group(f)
                if ident:
                    if ident.get("seed") is not None:
                        seed = int(ident["seed"])
                    run_id = (ident.get("run_id") or None)
                    utc_date = (str(ident.get("utc_start") or "")[:10] or None)
                    config_hash = ident.get("config_hash") or config_hash
    except Exception:
        pass
    return os.path.join(output_dir, provenance_filename(config_hash, seed, run_id, utc_date))


# ---------------------------------------------------------------------------
# Solver helper: k=0 zero-mode projection (array-library agnostic)
# ---------------------------------------------------------------------------

def zero_dc_mode(field_k):
    """
    Project out the k=0 (DC) Fourier mode in place and return the array.

    Works on numpy or cupy arrays (only uses ``[0, 0, 0]`` indexing).  Used to
    neutralize the affect field's secular runaway: at k=0 the wave operator has
    no restoring force (c^2 k^2 = 0), so a non-zero source there integrates to an
    unbounded quadratic-in-time zero mode.  The constant mode is pure gauge.
    """
    field_k[0, 0, 0] = 0
    return field_k
