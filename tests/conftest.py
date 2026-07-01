import json
from pathlib import Path

import pytest

from config_utils import create_session_workspace, write_active_run_pointer_atomic
from orchestrator.schema_utils import initialize_ledger_schema


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)

    for relative in (
        "simulation_data",
        "provenance_reports",
        "runtime_logs",
        "archive",
        "GIFS",
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)

    for file_name, payload in (
        ("backlog_queue.json", []),
        ("result_queue.json", []),
        ("backlog_queue.json.workers.json", {}),
        ("backlog_queue.json.claims.json", {}),
    ):
        with open(tmp_path / file_name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    db_path = tmp_path / "simulation_ledger.db"
    initialize_ledger_schema(str(db_path))

    session = create_session_workspace("TEST_HUNT")
    write_active_run_pointer_atomic(
        {
            "hunt_name": session["hunt_name"],
            "run_id": session["run_id"],
            "session_dir": session["session_dir"],
            "db_file": session["db_file"],
            "provenance_dir": session["provenance_dir"],
            "created_at": session["created_at"],
        }
    )
    initialize_ledger_schema(session["db_file"])
    return tmp_path
