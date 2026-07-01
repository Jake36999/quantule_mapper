import asyncio
from pathlib import Path

import pytest

import app as app_module


def _drain_telemetry_queue() -> list[dict]:
    drained: list[dict] = []
    while not app_module.telemetry_queue.empty():
        try:
            item = app_module.telemetry_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if isinstance(item, dict):
            drained.append(item)
    return drained


def test_log_tailer_uses_eof_baseline_and_emits_only_new_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    log_path = tmp_path / "orchestrator.log"
    log_path.write_text("old historical line\n", encoding="utf-8")

    monkeypatch.setattr(
        app_module,
        "DEBUG_LOG_SOURCES",
        [{"id": "orchestrator", "name": "Orchestrator", "path": log_path}],
    )

    file_positions = app_module._initialize_log_tail_positions()
    assert file_positions["orchestrator"] == log_path.stat().st_size

    _drain_telemetry_queue()
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write("new live line\n")

    asyncio.run(app_module._tail_logs_once(file_positions))
    events = _drain_telemetry_queue()

    assert events
    assert all(evt.get("type") == "terminal_log" for evt in events)
    assert any("new live line" in str(evt.get("line", "")) for evt in events)
    assert all("old historical line" not in str(evt.get("line", "")) for evt in events)


def test_log_tailer_handles_truncation_and_continues_streaming(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    log_path = tmp_path / "worker.log"
    log_path.write_text("line one\nline two\n", encoding="utf-8")

    monkeypatch.setattr(
        app_module,
        "DEBUG_LOG_SOURCES",
        [{"id": "worker", "name": "Worker GPU0", "path": log_path}],
    )

    file_positions = app_module._initialize_log_tail_positions()
    _drain_telemetry_queue()

    # Simulate rotation/truncation then fresh writes.
    log_path.write_text("post-rotate line\n", encoding="utf-8")

    asyncio.run(app_module._tail_logs_once(file_positions))
    events = _drain_telemetry_queue()

    assert events
    assert any("post-rotate line" in str(evt.get("line", "")) for evt in events)
    assert all(evt.get("feed_id") == "worker" for evt in events)
