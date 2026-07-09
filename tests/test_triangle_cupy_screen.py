import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "triangle_cupy_screen.py"
spec = importlib.util.spec_from_file_location("triangle_cupy_screen", MODULE_PATH)
screen = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = screen
spec.loader.exec_module(screen)


def test_smoke_parser_defaults_are_short_and_small():
    args = screen.build_arg_parser().parse_args(["smoke"])

    assert args.N == 32
    assert args.steps == 50
    assert args.spacing == 0.32
    assert args.require_gpu_name == "NVIDIA GeForce GTX 1080"


def test_screen_parser_requires_explicit_command_but_has_phase_c_spacing_defaults():
    args = screen.build_arg_parser().parse_args(["screen", "--N", "96", "--steps", "4000"])

    assert args.N == 96
    assert args.steps == 4000
    assert args.spacings == "0.28,0.32,0.36,0.40,0.45,0.49,0.53"
    assert args.require_gpu_name == "NVIDIA GeForce GTX 1080"


def test_build_case_config_uses_phase_c_feb_params_and_simulation_schema():
    cfg = screen.build_case_config(case_id="unit", N=32, steps=25, spacing=0.36, dt=0.005, L=10.0)

    assert cfg["case_id"] == "unit"
    assert cfg["simulation"] == {"n_grid": 32, "t_steps": 25, "dt": 0.005, "l_domain": 10.0}
    assert cfg["triangle_ic"]["spacing_box"] == 0.36
    assert cfg["params"]["param_D"] == screen.FEB_PARAMS["param_D"]
    assert cfg["phase_c_regime"] == "dissipative_geometry_diagnostic_only"
    assert cfg["config_hash"]


def test_output_paths_are_case_scoped(tmp_path):
    paths = screen.case_paths(tmp_path, "triangle_s0.32_N32_T50")

    assert paths["case_dir"] == tmp_path / "triangle_s0.32_N32_T50"
    assert paths["artifact"].name == "triangle_s0.32_N32_T50_smoke.npz"
    assert paths["metadata"].name == "triangle_s0.32_N32_T50_metadata.json"
