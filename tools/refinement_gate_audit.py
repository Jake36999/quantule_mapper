#!/usr/bin/env python3
"""
tools/refinement_gate_audit.py

Prints a snapshot of the refinement promotion gate state from the active ledger:
  - Current solver contract version
  - Counts of each refinement_status / certification state
  - Pending refinement groups
  - Champion-eligible candidates (REFINEMENT_STABLE)
"""
import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.contracts import SOLVER_CONTRACT_VERSION, PROVISIONAL_SSE_THRESHOLD
from config_utils import resolve_active_session_paths, ActiveRunPointerError


def _query(conn: sqlite3.Connection, sql: str, params=()):
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()


def main():
    try:
        paths = resolve_active_session_paths(require_exists=True)
        db_path = str(paths["db_file"])
    except ActiveRunPointerError as exc:
        print(f"[ERROR] No active session: {exc}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Refinement Gate Audit")
    print(f"  DB: {db_path}")
    print(f"{'='*60}")
    print(f"  Contract version : {SOLVER_CONTRACT_VERSION}")
    print(f"  SSE threshold    : {PROVISIONAL_SSE_THRESHOLD}")
    print()

    with sqlite3.connect(db_path) as conn:
        # Refinement status distribution
        rows = _query(conn, """
            SELECT COALESCE(m.refinement_status, 'NULL') AS status, COUNT(*) AS cnt
            FROM runs r
            LEFT JOIN metrics m ON r.config_hash = m.config_hash
            WHERE r.status = 'completed'
            GROUP BY status
            ORDER BY cnt DESC
        """)
        print("  Refinement status (completed runs):")
        for r in rows:
            print(f"    {r['status']:<30} {r['cnt']:>6}")
        print()

        # Solver contract version distribution
        rows = _query(conn, """
            SELECT COALESCE(m.solver_contract_version, 'NULL') AS cv, COUNT(*) AS cnt
            FROM runs r
            LEFT JOIN metrics m ON r.config_hash = m.config_hash
            WHERE r.status = 'completed'
            GROUP BY cv
            ORDER BY cnt DESC
        """)
        print("  Solver contract versions (completed runs):")
        for r in rows:
            certified = " ✓" if r['cv'] == SOLVER_CONTRACT_VERSION else " ✗"
            print(f"    {r['cv']:<45} {r['cnt']:>5}{certified}")
        print()

        # Pending refinement groups
        try:
            rows = _query(conn, """
                SELECT baseline_config_hash, variant_label, status
                FROM refinement_variants
                WHERE status = 'PENDING'
                ORDER BY baseline_config_hash, variant_label
            """)
            print(f"  Pending refinement variants: {len(rows)}")
            for r in rows:
                print(f"    {r['baseline_config_hash'][:12]}  {r['variant_label']:<20}  {r['status']}")
            print()
        except sqlite3.OperationalError:
            print("  refinement_variants table not found (schema migration pending)\n")

        # Champion-eligible candidates
        rows = _query(conn, """
            SELECT r.config_hash, r.generation, m.log_prime_sse, m.refinement_status, m.solver_contract_version
            FROM runs r
            JOIN metrics m ON r.config_hash = m.config_hash
            WHERE r.status = 'completed'
              AND m.refinement_status = 'REFINEMENT_STABLE'
            ORDER BY m.log_prime_sse ASC
            LIMIT 10
        """)
        print(f"  Champion-eligible (REFINEMENT_STABLE, top 10 by SSE):")
        if rows:
            for r in rows:
                print(f"    {r['config_hash'][:12]}  gen={r['generation']}  sse={r['log_prime_sse']:.6f}")
        else:
            print("    (none)")
        print()

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
