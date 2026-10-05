from pathlib import Path

import h5py
import numpy as np
from fastapi.testclient import TestClient

import app as app_module
from quantule_viz.visual_analysis import (
    CompletedRunAdapter,
    RenderRequest,
    VisualRunRegistry,
    discover_runs,
    render_manifest,
)


def _sphere_field(n: int = 8) -> np.ndarray:
    coords = np.indices((n, n, n), dtype=np.float32)
    center = (n - 1) / 2.0
    r2 = ((coords[0] - center) ** 2 + (coords[1] - center) ** 2 + (coords[2] - center) ** 2)
    return np.exp(-r2 / 4.0).astype(np.float32)


def _write_final_h5(path: Path, config_hash: str = "a" * 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rho = _sphere_field()
    psi = np.sqrt(rho).astype(np.complex64) * np.exp(1j * rho).astype(np.complex64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("psi_final", data=psi)
        handle.create_dataset("omega_sq_final", data=rho * 2.0)
        telemetry = handle.create_group("telemetry")
        telemetry.create_dataset("step", data=np.array([0, 10, 20], dtype=np.int64))
        telemetry.create_dataset("energy", data=np.array([1.0, 1.01, 1.02], dtype=np.float64))
        telemetry.create_dataset("C_invariant", data=np.array([0.5, 0.51, 0.52], dtype=np.float64))
        identity = handle.create_group("identity")
        identity.create_dataset("run_id", data=np.array(["run-final-1"], dtype="S32"))
        identity.create_dataset("config_hash", data=np.array([config_hash], dtype="S80"))
        identity.create_dataset("seed", data=np.array([1042], dtype=np.int64))
        identity.create_dataset("git_commit", data=np.array(["abc123"], dtype="S16"))
    return path


def _write_history_h5(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.stack([_sphere_field(6), _sphere_field(6) * 0.5], axis=0)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("rho_history", data=frames)
    return path


def _write_frames_npz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.stack([_sphere_field(5), _sphere_field(5) * 0.25], axis=0)
    np.savez(path, frames=frames.astype(np.float32))
    return path


def test_adapter_maps_final_hdf5_fields_and_metrics(tmp_path: Path):
    artifact = _write_final_h5(tmp_path / "rho_history_" / "rho_history_aaaaaaaaaaaaaaaa.h5")
    adapter = CompletedRunAdapter(artifact)

    record = adapter.inspect()
    assert record is not None
    assert record.run_id == "run-final-1"
    assert record.config_hash == "a" * 64
    assert record.seed == 1042
    assert set(record.available_fields) == {"rho", "phase", "omega2"}
    assert record.frame_count == 1

    rho, meta = adapter.read_field("rho", 0)
    assert rho.shape == (8, 8, 8)
    assert meta["source_dataset"] == "/psi_final"

    phase, _ = adapter.read_field("phase", 0)
    assert phase.shape == (8, 8, 8)

    metrics = adapter.read_metrics()
    assert len(metrics["series"]) == 3
    assert metrics["series"][1]["step"] == 10


def test_adapter_maps_history_hdf5_and_npz_frame_counts(tmp_path: Path):
    history = CompletedRunAdapter(_write_history_h5(tmp_path / "rho_history_bbbbbbbbbbbbbbbb.h5")).inspect()
    bundle = CompletedRunAdapter(_write_frames_npz(tmp_path / "frames_cccccccccccccccc.npz")).inspect()

    assert history is not None
    assert history.frame_count == 2
    assert history.available_fields == ["rho"]
    assert bundle is not None
    assert bundle.frame_count == 2
    assert bundle.available_fields == ["rho"]


def test_registry_ingests_immutable_run_instances(tmp_path: Path):
    registry = VisualRunRegistry(tmp_path / "registry.sqlite")
    first = _write_final_h5(tmp_path / "one" / "rho_history_" / "rho_history_aaaaaaaaaaaaaaaa.h5")
    second = _write_final_h5(tmp_path / "two" / "rho_history_" / "rho_history_aaaaaaaaaaaaaaaa.h5")

    # Remove explicit identity run_id from the second artifact so path identity is used.
    with h5py.File(second, "a") as handle:
        del handle["identity"]["run_id"]

    runs = discover_runs(registry, [tmp_path])
    matching = [row for row in runs if row["config_hash"] == "a" * 64]

    assert len(matching) == 2
    assert len({row["run_id"] for row in matching}) == 2


def test_render_cache_emits_volume_isosurface_and_slices(tmp_path: Path):
    registry = VisualRunRegistry(tmp_path / "registry.sqlite")
    artifact = _write_final_h5(tmp_path / "rho_history_aaaaaaaaaaaaaaaa.h5")
    discover_runs(registry, [tmp_path])
    run = registry.get("run-final-1")
    assert run is not None

    volume = render_manifest(run, RenderRequest(tile="density_volume", resolution=8), tmp_path / "cache")
    assert volume["assets"]["volume"].endswith("volume.f32")
    assert volume["shape"] == [8, 8, 8]

    surface = render_manifest(run, RenderRequest(tile="density_isosurface", resolution=8, threshold=0.4), tmp_path / "cache")
    assert surface["assets"]["points"].endswith("points.f32")
    assert surface["assets"]["polys"].endswith("polys.u32")

    slices = render_manifest(run, RenderRequest(tile="orthogonal_slices", resolution=8), tmp_path / "cache")
    assert slices["assets"]["slices"].endswith("slices.json")


def test_visual_api_lists_renders_metrics_and_serves_cache(monkeypatch, sandbox: Path):
    data_dir = sandbox / "simulation_data"
    artifact = _write_final_h5(data_dir / ("rho_history_" + "d" * 64 + ".h5"), "d" * 64)
    monkeypatch.setattr(app_module, "DATA_DIR", str(data_dir))

    with TestClient(app_module.app) as client:
        runs_response = client.get("/api/visual/runs")
        assert runs_response.status_code == 200
        runs = runs_response.json()["runs"]
        run = next(row for row in runs if row["artifact_path"] == str(artifact.resolve()))

        detail_response = client.get(f"/api/visual/runs/{run['run_id']}")
        assert detail_response.status_code == 200

        metrics_response = client.get(f"/api/visual/runs/{run['run_id']}/metrics")
        assert metrics_response.status_code == 200
        assert len(metrics_response.json()["series"]) == 3

        render_response = client.post(
            f"/api/visual/runs/{run['run_id']}/render",
            json={"tile": "density_volume", "field": "rho", "resolution": 8},
        )
        assert render_response.status_code == 200
        manifest = render_response.json()["manifest"]

        asset_response = client.get(manifest["assets"]["volume"])
        assert asset_response.status_code == 200
        assert asset_response.headers["content-type"].startswith("application/octet-stream")
        assert len(asset_response.content) == np.prod(manifest["shape"]) * 4


def test_visual_api_rejects_bad_field_and_unknown_hash(monkeypatch, sandbox: Path):
    data_dir = sandbox / "simulation_data"
    _write_final_h5(data_dir / ("rho_history_" + "e" * 64 + ".h5"), "e" * 64)
    monkeypatch.setattr(app_module, "DATA_DIR", str(data_dir))

    with TestClient(app_module.app) as client:
        run = next(row for row in client.get("/api/visual/runs").json()["runs"] if row["config_hash"] == "e" * 64)

        bad_render = client.post(
            f"/api/visual/runs/{run['run_id']}/render",
            json={"tile": "density_volume", "field": "missing"},
        )
        assert bad_render.status_code == 422

        missing_rerun = client.post("/api/visual/rerun", json={"config_hash": "f" * 64})
        assert missing_rerun.status_code == 404
