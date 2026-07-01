from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module


def _write_pointcloud_artifact(target_dir: Path, config_hash: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / f"rho_history_{config_hash}.h5"

    psi = np.zeros((4, 4, 4), dtype=np.complex64)
    psi[1, 1, 1] = 2.0 + 0.0j
    psi[2, 2, 2] = 1.0 + 1.0j
    stage = np.zeros((4, 4, 4), dtype=np.complex128)
    stage[1, 1, 1] = 3.0 + 4.0j

    with h5py.File(artifact_path, "w") as h5f:
        h5f.create_dataset("psi_final", data=psi)
        h5f.create_dataset("A_final", data=np.abs(psi).astype(np.float32))
        h5f.create_dataset("N_a_stage", data=stage)

    return artifact_path


@pytest.fixture
def pointcloud_client(monkeypatch: pytest.MonkeyPatch, sandbox: Path):
    monkeypatch.setattr(app_module, "DATA_DIR", str(sandbox / "simulation_data"))
    with TestClient(app_module.app) as client:
        yield client, sandbox


def test_pointcloud_hash_endpoint_streams_valid_binary(pointcloud_client):
    client, sandbox = pointcloud_client
    config_hash = "a" * 64
    _write_pointcloud_artifact(sandbox / "simulation_data", config_hash)

    response = client.get(f"/api/data/pointcloud/hash/{config_hash}/psi_final?threshold=0.05")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert len(response.content) > 0
    assert len(response.content) % 16 == 0


def test_pointcloud_hash_endpoint_supports_stage_datasets(pointcloud_client):
    client, sandbox = pointcloud_client
    config_hash = "b" * 64
    _write_pointcloud_artifact(sandbox / "simulation_data", config_hash)

    response = client.get(f"/api/data/pointcloud/hash/{config_hash}/N_a_stage?threshold=0.01")

    assert response.status_code == 200
    assert len(response.content) % 16 == 0


def test_pointcloud_hash_endpoint_rejects_invalid_hash(pointcloud_client):
    client, _ = pointcloud_client

    response = client.get("/api/data/pointcloud/hash/not-a-hash/psi_final")

    assert response.status_code == 400


def test_pointcloud_hash_endpoint_rejects_unlisted_dataset(pointcloud_client):
    client, sandbox = pointcloud_client
    config_hash = "c" * 64
    _write_pointcloud_artifact(sandbox / "simulation_data", config_hash)

    response = client.get(f"/api/data/pointcloud/hash/{config_hash}/__internal")

    assert response.status_code == 404


def test_pointcloud_hash_endpoint_returns_404_for_missing_artifact(pointcloud_client):
    client, _ = pointcloud_client
    config_hash = "d" * 64

    response = client.get(f"/api/data/pointcloud/hash/{config_hash}/psi_final")

    assert response.status_code == 404


def test_pointcloud_hash_endpoint_clamps_threshold(pointcloud_client):
    client, sandbox = pointcloud_client
    config_hash = "e" * 64
    _write_pointcloud_artifact(sandbox / "simulation_data", config_hash)

    response = client.get(f"/api/data/pointcloud/hash/{config_hash}/psi_final?threshold=2.0")

    assert response.status_code == 200
    assert len(response.content) % 16 == 0