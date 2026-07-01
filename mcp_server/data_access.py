"""
mcp_server.data_access — pure read-only query layer for the IRER stack.

Every function takes explicit paths (so it is trivially testable) and returns
plain JSON-able dicts matching the output schemas in docs/MCP_TOOLS_SPEC.md.
No MCP SDK, no cupy, no GPU.  All access is read-only: nothing here opens a DB
for writing, deletes a file, or mutates an artifact.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import statistics
from typing import Any, Dict, List, Optional

try:
    from orchestrator.run_identity import read_identity_group, compatibility_key
except Exception:  # pragma: no cover
    read_identity_group = None
    compatibility_key = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _connect_ro(db_path: str) -> Optional[sqlite3.Connection]:
    if not db_path or not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.OperationalError:
        return set()


def _find_artifact(config_hash: str, artifact_roots: List[str]) -> Optional[str]:
    if not config_hash:
        return None
    for root in artifact_roots or []:
        if not os.path.isdir(root):
            continue
        for pattern in (f"rho_history_{config_hash}*.h5", f"**/*{config_hash}*.h5"):
            hits = glob.glob(os.path.join(root, pattern), recursive=True)
            if hits:
                return os.path.abspath(hits[0])
    return None


def _find_provenance(config_hash: str, provenance_dir: str) -> Optional[str]:
    if not config_hash or not os.path.isdir(provenance_dir):
        return None
    hits = sorted(glob.glob(os.path.join(provenance_dir, f"provenance_{config_hash}*.json")))
    return os.path.abspath(hits[0]) if hits else None


# ---------------------------------------------------------------------------
# 1. get_run_status
# ---------------------------------------------------------------------------

def get_run_status(
    db_path: str,
    config_hash: str,
    seed: Optional[int] = None,
    hunt_name: Optional[str] = None,
    provenance_dir: Optional[str] = None,
    artifact_roots: Optional[List[str]] = None,
) -> Dict[str, Any]:
    base = {
        "config_hash": config_hash, "seed": seed, "hunt_name": hunt_name,
        "generation": None, "status": "NOT_FOUND", "fitness": None,
        "solver_contract_version": None, "variant_label": None,
        "refinement_status": None, "artifact_path": None,
        "provenance_path": None, "utc_timestamp": None,
    }
    conn = _connect_ro(db_path)
    if conn is None:
        return base
    try:
        rcols = _table_columns(conn, "runs")
        if "config_hash" not in rcols:
            return base
        where = ["r.config_hash LIKE ?"]
        params: List[Any] = [f"{config_hash}%"]
        if seed is not None and "seed" in rcols:
            where.append("r.seed = ?")
            params.append(int(seed))
        if hunt_name and "hunt_name" in rcols:
            where.append("r.hunt_name = ?")
            params.append(hunt_name)
        order = "r.fitness ASC" if "fitness" in rcols else "r.config_hash"
        sql = (
            "SELECT r.*, m.refinement_status AS m_refinement_status "
            "FROM runs r LEFT JOIN metrics m ON r.config_hash = m.config_hash "
            f"WHERE {' AND '.join(where)} ORDER BY {order} LIMIT 1"
        )
        row = conn.execute(sql, params).fetchone()
        if row is None:
            return base
        rd = dict(row)
        out = dict(base)
        out.update({
            "config_hash": rd.get("config_hash"),
            "seed": rd.get("seed"),
            "hunt_name": rd.get("hunt_name"),
            "generation": rd.get("generation"),
            "status": rd.get("status") or "NOT_FOUND",
            "fitness": rd.get("fitness"),
            "solver_contract_version": rd.get("solver_contract_version"),
            "variant_label": rd.get("variant_label"),
            "refinement_status": rd.get("m_refinement_status"),
            "utc_timestamp": rd.get("utc_start") or rd.get("timestamp"),
        })
        ch = out["config_hash"]
        out["artifact_path"] = _find_artifact(ch, artifact_roots or [])
        if provenance_dir:
            out["provenance_path"] = _find_provenance(ch, provenance_dir)
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2. query_ledger
# ---------------------------------------------------------------------------

def query_ledger(
    db_path: str,
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
    out = {"rows": [], "total_matched": 0, "compatibility_warning": None}
    conn = _connect_ro(db_path)
    if conn is None:
        return out
    try:
        rcols = _table_columns(conn, "runs")
        if "config_hash" not in rcols:
            return out
        limit = max(1, min(int(limit or 50), 500))
        where: List[str] = []
        params: List[Any] = []
        if hunt_name and "hunt_name" in rcols:
            where.append("r.hunt_name = ?"); params.append(hunt_name)
        if generation_min is not None:
            where.append("r.generation >= ?"); params.append(int(generation_min))
        if generation_max is not None:
            where.append("r.generation <= ?"); params.append(int(generation_max))
        if solver_contract_version and "solver_contract_version" in rcols:
            where.append("r.solver_contract_version = ?"); params.append(solver_contract_version)
        if variant_label and "variant_label" in rcols:
            where.append("r.variant_label = ?"); params.append(variant_label)
        if status and status.lower() != "any":
            where.append("r.status = ?"); params.append(status.upper())
        if sse_max is not None:
            where.append("m.log_prime_sse <= ?"); params.append(float(sse_max))

        order_map = {
            "log_prime_sse": "m.log_prime_sse ASC",
            "generation": "r.generation DESC",
            "timestamp": "r.timestamp DESC",
        }
        order_sql = order_map.get(order_by, "m.log_prime_sse ASC")
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        count_sql = f"SELECT COUNT(*) FROM runs r LEFT JOIN metrics m ON r.config_hash = m.config_hash{where_sql}"
        total = conn.execute(count_sql, params).fetchone()[0]

        sql = (
            "SELECT r.config_hash, r.seed, r.generation, r.status, "
            "m.log_prime_sse AS log_prime_sse, r.solver_contract_version, "
            "r.variant_label, m.refinement_status, r.origin "
            "FROM runs r LEFT JOIN metrics m ON r.config_hash = m.config_hash"
            f"{where_sql} ORDER BY {order_sql} LIMIT {limit}"
        )
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        out["rows"] = rows
        out["total_matched"] = int(total)

        contracts = {r.get("solver_contract_version") for r in rows if r.get("solver_contract_version")}
        variants = {r.get("variant_label") for r in rows if r.get("variant_label")}
        if len(contracts) > 1 or len(variants) > 1:
            out["compatibility_warning"] = (
                f"Results span multiple variants/contracts (contracts={sorted(contracts)}, "
                f"variants={sorted(variants)}); they are NOT comparable by SSE. "
                "Filter by solver_contract_version + variant_label to rank."
            )
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. read_audit_log
# ---------------------------------------------------------------------------

def read_audit_log(
    audit_log_path: str,
    stage: Optional[str] = None,
    config_hash: Optional[str] = None,
    hunt_name: Optional[str] = None,
    since_utc: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    out = {"events": [], "total_lines_scanned": 0}
    if not audit_log_path or not os.path.exists(audit_log_path):
        return out
    limit = max(1, min(int(limit or 100), 1000))
    matched: List[Dict[str, Any]] = []
    scanned = 0
    with open(audit_log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            scanned += 1
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if stage and ev.get("stage") != stage:
                continue
            if config_hash and not str(ev.get("config_hash") or "").startswith(config_hash):
                continue
            if hunt_name:
                details = ev.get("details") or {}
                if ev.get("hunt_name") != hunt_name and details.get("hunt_name") != hunt_name:
                    continue
            if since_utc and str(ev.get("timestamp") or "") < since_utc:
                continue
            matched.append(ev)
    out["events"] = matched[-limit:]
    out["total_lines_scanned"] = scanned
    return out


# ---------------------------------------------------------------------------
# 4. read_provenance
# ---------------------------------------------------------------------------

def read_provenance(
    provenance_dir: str,
    config_hash: Optional[str] = None,
    seed: Optional[int] = None,
    provenance_path: Optional[str] = None,
) -> Dict[str, Any]:
    out = {
        "found": False, "provenance_path": None, "schema_version": None,
        "solver_contract_version": None, "variant_label": None,
        "spectral_fidelity": None, "falsifiability": None, "full_payload": None,
    }
    path = provenance_path
    if not path and config_hash:
        path = _find_provenance(config_hash, provenance_dir)
        if seed is not None:
            seeded = sorted(glob.glob(os.path.join(provenance_dir, f"provenance_{config_hash}_seed{int(seed)}*.json")))
            if seeded:
                path = os.path.abspath(seeded[0])
    if not path or not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return out
    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    contract = payload.get("solver_contract", {}) if isinstance(payload, dict) else {}
    out.update({
        "found": True,
        "provenance_path": os.path.abspath(path),
        "schema_version": meta.get("schema_version"),
        "solver_contract_version": (contract or {}).get("solver_contract_version"),
        "variant_label": (meta.get("run_metadata") or {}).get("variant_label"),
        "spectral_fidelity": payload.get("spectral_fidelity"),
        "falsifiability": payload.get("falsifiability"),
        "full_payload": payload,
    })
    return out


# ---------------------------------------------------------------------------
# 5. list_artifacts
# ---------------------------------------------------------------------------

def list_artifacts(
    artifact_roots: List[str],
    hunt_name: Optional[str] = None,
    generation: Optional[int] = None,
    solver_contract_version: Optional[str] = None,
    variant_label: Optional[str] = None,
    include_legacy: bool = False,
    limit: int = 100,
) -> Dict[str, Any]:
    import h5py

    limit = max(1, min(int(limit or 100), 2000))
    artifacts: List[Dict[str, Any]] = []
    seen = set()
    for root in artifact_roots or []:
        if not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "**", "*.h5"), recursive=True):
            ap = os.path.abspath(path)
            if ap in seen:
                continue
            seen.add(ap)
            entry = {
                "path": ap, "size_bytes": None, "has_identity_group": False,
                "hunt_name": None, "generation": None, "config_hash": None,
                "seed": None, "solver_contract_version": None,
                "variant_label": None, "utc_start": None, "is_legacy": True,
            }
            try:
                entry["size_bytes"] = os.path.getsize(ap)
                with h5py.File(ap, "r") as f:
                    ident = read_identity_group(f) if read_identity_group else {}
                    if ident:
                        entry.update({
                            "has_identity_group": True, "is_legacy": False,
                            "hunt_name": ident.get("hunt_name"),
                            "generation": ident.get("generation"),
                            "config_hash": ident.get("config_hash"),
                            "seed": ident.get("seed"),
                            "solver_contract_version": ident.get("solver_contract_version"),
                            "variant_label": ident.get("variant_label"),
                            "utc_start": ident.get("utc_start"),
                        })
            except Exception:
                pass

            if entry["is_legacy"] and not include_legacy:
                continue
            if hunt_name and entry.get("hunt_name") != hunt_name:
                continue
            if generation is not None and entry.get("generation") != generation:
                continue
            if solver_contract_version and entry.get("solver_contract_version") != solver_contract_version:
                continue
            if variant_label and entry.get("variant_label") != variant_label:
                continue
            artifacts.append(entry)

    total = len(artifacts)
    return {"artifacts": artifacts[:limit], "total_found": total}


# ---------------------------------------------------------------------------
# 6. inspect_hdf5_schema
# ---------------------------------------------------------------------------

def inspect_hdf5_schema(artifact_path: str) -> Dict[str, Any]:
    import h5py
    import numpy as np  # noqa: F401

    out = {
        "path": os.path.abspath(artifact_path) if artifact_path else None,
        "identity": None, "datasets": [], "solver_contract_json": None,
        "sentinel_code": None, "sentinel_reason": None, "is_fail_artifact": False,
    }
    if not artifact_path or not os.path.exists(artifact_path):
        return out
    with h5py.File(artifact_path, "r") as f:
        datasets: List[Dict[str, Any]] = []

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append({
                    "name": "/" + name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "compression": obj.compression,
                    "size_bytes": int(obj.nbytes) if hasattr(obj, "nbytes") else None,
                })

        f.visititems(_visit)
        out["datasets"] = datasets

        if read_identity_group:
            ident = read_identity_group(f)
            out["identity"] = ident or None

        if "solver_contract" in f:
            try:
                raw = f["solver_contract"][0]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                out["solver_contract_json"] = json.loads(raw)
            except Exception:
                out["solver_contract_json"] = None

        if "sentinel_code" in f:
            try:
                out["sentinel_code"] = float(f["sentinel_code"][0])
                out["is_fail_artifact"] = True
            except Exception:
                pass
        if "sentinel_reason" in f:
            try:
                sr = f["sentinel_reason"][0]
                out["sentinel_reason"] = sr.decode("utf-8") if isinstance(sr, bytes) else str(sr)
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# 7. summarise_generation
# ---------------------------------------------------------------------------

def summarise_generation(
    db_path: str,
    hunt_name: str,
    generation: int,
    solver_contract_version: str,
    variant_label: str,
) -> Dict[str, Any]:
    out = {
        "hunt_name": hunt_name, "generation": generation,
        "solver_contract_version": solver_contract_version, "variant_label": variant_label,
        "total_runs": 0, "succeeded": 0, "failed": 0,
        "sentinel_breakdown": {"math_explosion": 0, "physics_drift": 0, "geometry_sanity": 0},
        "sse_stats": {"min": None, "median": None, "p25": None, "p75": None,
                      "golden_count": 0, "silver_count": 0},
        "new_champion": None, "pareto_front_size": 0, "degenerate_geometry_count": 0,
    }
    conn = _connect_ro(db_path)
    if conn is None:
        return out
    try:
        rcols = _table_columns(conn, "runs")
        if "config_hash" not in rcols:
            return out
        where = ["r.generation = ?"]
        params: List[Any] = [int(generation)]
        if "hunt_name" in rcols:
            where.append("r.hunt_name = ?"); params.append(hunt_name)
        if "solver_contract_version" in rcols:
            where.append("r.solver_contract_version = ?"); params.append(solver_contract_version)
        if "variant_label" in rcols:
            where.append("r.variant_label = ?"); params.append(variant_label)
        where_sql = " AND ".join(where)

        rows = [dict(r) for r in conn.execute(
            "SELECT r.config_hash, r.seed, r.status, r.fitness, m.log_prime_sse AS sse "
            "FROM runs r LEFT JOIN metrics m ON r.config_hash = m.config_hash "
            f"WHERE {where_sql}", params,
        ).fetchall()]
        out["total_runs"] = len(rows)
        out["succeeded"] = sum(1 for r in rows if (r.get("status") or "").upper() == "SUCCESS")
        out["failed"] = sum(1 for r in rows if (r.get("status") or "").upper() == "FAIL")

        sses = [float(r["sse"]) for r in rows if r.get("sse") is not None and float(r["sse"]) < 999.0]
        if sses:
            sses_sorted = sorted(sses)
            out["sse_stats"]["min"] = sses_sorted[0]
            out["sse_stats"]["median"] = statistics.median(sses_sorted)
            out["sse_stats"]["p25"] = sses_sorted[max(0, int(0.25 * (len(sses_sorted) - 1)))]
            out["sse_stats"]["p75"] = sses_sorted[max(0, int(0.75 * (len(sses_sorted) - 1)))]
            out["sse_stats"]["golden_count"] = sum(1 for s in sses if s < 1.0)
            out["sse_stats"]["silver_count"] = sum(1 for s in sses if s < 3.0)
            best = min(rows, key=lambda r: (float(r["sse"]) if r.get("sse") is not None else 999.0))
            out["new_champion"] = {"config_hash": best.get("config_hash"),
                                   "seed": best.get("seed"), "log_prime_sse": best.get("sse")}

        # degenerate geometry count (param_rho_vac < 0.05) if parameters table present
        if "param_rho_vac" in _table_columns(conn, "parameters"):
            chs = [r["config_hash"] for r in rows]
            if chs:
                qmarks = ",".join("?" for _ in chs)
                deg = conn.execute(
                    f"SELECT COUNT(*) FROM parameters WHERE config_hash IN ({qmarks}) AND param_rho_vac < 0.05",
                    chs,
                ).fetchone()[0]
                out["degenerate_geometry_count"] = int(deg)
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 8. audit_data_contract
# ---------------------------------------------------------------------------

def audit_data_contract(
    db_path: str,
    config_hash: Optional[str] = None,
    hunt_name: Optional[str] = None,
    sample_size: int = 20,
    provenance_dir: Optional[str] = None,
    artifact_roots: Optional[List[str]] = None,
) -> Dict[str, Any]:
    import h5py

    out = {"total_checked": 0, "compliant": 0, "non_compliant": 0, "issues": [], "summary": ""}
    conn = _connect_ro(db_path)
    if conn is None:
        out["summary"] = "ledger not found"
        return out
    try:
        rcols = _table_columns(conn, "runs")
        where: List[str] = []
        params: List[Any] = []
        if config_hash:
            where.append("config_hash LIKE ?"); params.append(f"{config_hash}%")
        if hunt_name and "hunt_name" in rcols:
            where.append("hunt_name = ?"); params.append(hunt_name)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        sample_size = max(1, min(int(sample_size or 20), 500))
        rows = [dict(r) for r in conn.execute(
            f"SELECT * FROM runs{where_sql} LIMIT {sample_size}", params
        ).fetchall()]
    finally:
        conn.close()

    for rd in rows:
        ch = rd.get("config_hash")
        issues: List[str] = []
        artifact = _find_artifact(ch, artifact_roots or [])
        if artifact:
            try:
                with h5py.File(artifact, "r") as f:
                    if "identity" not in f:
                        issues.append("missing /identity group")
                    else:
                        ident = read_identity_group(f) if read_identity_group else {}
                        hcv = ident.get("solver_contract_version")
                        lcv = rd.get("solver_contract_version")
                        if hcv and lcv and hcv != lcv:
                            issues.append("solver_contract_version mismatch between HDF5 and ledger")
                    if "A_dot_final" in f and "A_dot_k_final" not in f:
                        issues.append("A_dot_final in spectral space (legacy label; expected A_dot_k_final)")
            except Exception as exc:
                issues.append(f"artifact unreadable: {type(exc).__name__}")
        else:
            issues.append("artifact not found")

        if provenance_dir:
            prov_hits = glob.glob(os.path.join(provenance_dir, f"provenance_{ch}*.json")) if ch else []
            if ch and not prov_hits:
                issues.append("no provenance file found")

        if rd.get("solver_contract_version") is None:
            issues.append("ledger discriminator solver_contract_version not populated")

        out["total_checked"] += 1
        if issues:
            out["non_compliant"] += 1
            out["issues"].append({"config_hash": ch, "artifact_path": artifact, "issues": issues})
        else:
            out["compliant"] += 1

    out["summary"] = (
        f"{out['compliant']}/{out['total_checked']} runs compliant; "
        f"{out['non_compliant']} with issues."
    )
    return out
