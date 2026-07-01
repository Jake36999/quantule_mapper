"""
mcp_server.config — resolves the data locations the read-only tools query, and
enforces a read whitelist so a tool can never read outside the project root.

No MCP SDK or GPU dependency; importable and testable anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_DB_NAME = "simulation_ledger.db"


@dataclass
class McpConfig:
    root: str = ""
    db_path: str = ""
    provenance_dir: str = ""
    audit_log: str = ""
    artifact_roots: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = os.path.abspath(self.root or os.environ.get("QM_ROOT") or os.getcwd())
        self.db_path = self.db_path or os.environ.get("ASTE_LEDGER_DB") or os.path.join(self.root, DEFAULT_DB_NAME)
        self.provenance_dir = self.provenance_dir or os.path.join(self.root, "provenance_reports")
        self.audit_log = self.audit_log or os.path.join(self.root, "runtime_logs", "run_lifecycle_audit.jsonl")
        if not self.artifact_roots:
            self.artifact_roots = [
                os.path.join(self.root, "simulation_data"),
                os.path.join(self.root, "runs"),
            ]

    def is_path_allowed(self, path: str) -> bool:
        """True only if `path` resolves inside the project root (read whitelist)."""
        try:
            ap = os.path.abspath(path)
            root = os.path.abspath(self.root)
            return os.path.commonpath([ap, root]) == root
        except Exception:
            return False


def default_config() -> McpConfig:
    return McpConfig()
