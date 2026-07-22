import os
import shutil
import time
import threading
import json
from pathlib import Path

import h5py  # type: ignore
from fastapi.testclient import TestClient
from app import app

GIFS_DIR = "GIFS"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _append_payload(received: list[dict], payload) -> None:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                received.append(item)
    elif isinstance(payload, dict):
        received.append(payload)


def _clear_gifs_dir(path: str) -> None:
    root = Path(path)
    root.mkdir(exist_ok=True)
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                pass

def test_websocket_gif_update(tmp_path):
    # Setup: ensure GIFS dir is clean
    _clear_gifs_dir(GIFS_DIR)

    with TestClient(app) as client:
        # Start WebSocket in a thread
        received_payloads = []

        def ws_thread():
            with client.websocket_connect("/ws/telemetry") as ws:
                try:
                    for _ in range(4):
                        _append_payload(received_payloads, ws.receive_json())
                        if any(payload.get("type") == "gif_update" for payload in received_payloads):
                            break
                except Exception as e:
                    import logging
                    logging.error(f"WebSocket Test Error: {e}")

        t = threading.Thread(target=ws_thread)
        t.start()
        time.sleep(1)  # Give the server time to start

        # Simulate adding a GIF
        gif_path = os.path.join(GIFS_DIR, "new_best.gif")
        with open(gif_path, "wb") as f:
            f.write(os.urandom(1024))
        time.sleep(2)  # Allow watchdog to trigger

        t.join(timeout=5)
        assert received_payloads, "No payload received over WebSocket."
        payload = next(item for item in received_payloads if item.get("type") == "gif_update")
        assert payload["type"] == "gif_update"
        assert payload["new_path"].endswith("/static/gifs/new_best.gif")


def _write_pde_telemetry_h5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        telemetry = handle.create_group("telemetry")
        telemetry.create_dataset("step", data=[0.0])
        telemetry.create_dataset("energy", data=[1.0])
        telemetry.create_dataset("C_invariant", data=[0.5])
        telemetry.create_dataset("max_amplitude", data=[2.0])


def test_websocket_emits_pde_history_contract(tmp_path):
    _clear_gifs_dir(GIFS_DIR)
    expected_frame = json.loads((FIXTURE_DIR / "pde_history_frame.json").read_text(encoding="utf-8"))

    artifact_path = tmp_path / "gen_0" / "rho_history_fixture.h5"
    _write_pde_telemetry_h5(artifact_path)

    render_meta = {
        "new_sse": 0.123,
        "pcs": 0.77,
        "ic": 0.66,
        "updated_at": "2026-03-12T00:00:00Z",
        "tier": "SILVER",
        "artifact_path": str(artifact_path),
    }
    Path(GIFS_DIR).mkdir(exist_ok=True)
    (Path(GIFS_DIR) / "render_meta.json").write_text(json.dumps(render_meta), encoding="utf-8")

    with TestClient(app) as client:
        received_payloads = []

        def ws_thread():
            with client.websocket_connect("/ws/telemetry") as ws:
                try:
                    for _ in range(8):
                        _append_payload(received_payloads, ws.receive_json())
                        if any(payload.get("type") == "pde_history" for payload in received_payloads):
                            break
                except Exception as e:
                    import logging
                    logging.error(f"WebSocket Test Error: {e}")

        t = threading.Thread(target=ws_thread)
        t.start()
        time.sleep(1)

        gif_path = Path(GIFS_DIR) / "new_best.gif"
        gif_path.write_bytes(os.urandom(1024))
        time.sleep(2)

        t.join(timeout=5)
        assert received_payloads, "No payload received over WebSocket."

        pde_payloads = [payload for payload in received_payloads if payload.get("type") == "pde_history"]
        assert pde_payloads, "Expected pde_history payload was not emitted."
        assert pde_payloads[0]["payload"] == expected_frame["payload"]
        assert pde_payloads[0]["severity"] == "INFO"
