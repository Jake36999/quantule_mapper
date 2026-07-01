import json
import os
import subprocess
import sys
from pathlib import Path

import h5py


def test_worker_cupy_forced_nan_writes_fail_open_sentinel(tmp_path: Path):
    params_path = tmp_path / "params.json"
    output_path = tmp_path / "sentinel_output.h5"

    payload = {
        "simulation": {
            "N_grid": 16,
            "L_domain": 10.0,
            "T_steps": 2,
            "dt": 0.001,
        },
        "global_seed": 42,
        "param_D": 1.0,
        "param_eta": 0.1,
        "param_rho_vac": 0.0,
        "param_a": 0.0,
        "param_s": 0.0,
        "param_f": 0.0,
        "collapse_threshold": 1e10,
    }
    params_path.write_text(json.dumps(payload), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["ASTE_FORCE_NAN_STEP"] = "0"

    completed = subprocess.run(
        [
            sys.executable,
            "worker_cupy.py",
            "--params",
            str(params_path),
            "--output",
            str(output_path),
        ],
        cwd=str(repo_root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert output_path.exists(), "Expected worker_cupy.py to emit an HDF5 artifact"

    with h5py.File(output_path, "r") as handle:
        assert "sentinel_code" in handle
        assert float(handle["sentinel_code"][()]) == 1002.0
        assert "sentinel_reason" in handle
        assert handle["sentinel_reason"][0].decode("utf-8") == "math_explosion"
