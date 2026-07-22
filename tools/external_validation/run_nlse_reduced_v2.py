from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("E:/quantule_mapper_external_data/DATA-NUM-001_nlse_package/extracted/source")
OUT_ROOT = REPO_ROOT / "docs/external_validation/external_data/nlse_reduced_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_project_v2() -> tuple[dict[str, Any], list[dict[str, str]]]:
    result = json.loads((REPO_ROOT / "docs/external_validation/generated_metrics/v2/result.json").read_text(encoding="utf-8"))
    with (REPO_ROOT / "docs/external_validation/generated_metrics/v2/summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return result, rows


def fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 1.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot),
    }


def inspect_package_source(source_root: Path) -> dict[str, Any]:
    setup_text = (source_root / "setup.py").read_text(encoding="utf-8", errors="replace")
    requirements = []
    req_path = source_root / "requirements.txt"
    if req_path.exists():
        requirements = [
            line.strip()
            for line in req_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    license_text = ""
    license_path = source_root / "LICENSE"
    if license_path.exists():
        license_text = license_path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    version = ""
    for marker in ["version='", 'version="']:
        if marker in setup_text:
            tail = setup_text.split(marker, 1)[1]
            version = tail.split(marker[-1], 1)[0]
            break
    return {
        "source_root": str(source_root),
        "package_version_from_setup": version or "UNKNOWN",
        "requirements": requirements,
        "license_file_first_line": license_text or "UNKNOWN",
        "run_mode": "import-from-unpacked-source with local child-process compatibility shims",
    }


def write_shims(shim_dir: Path) -> None:
    write_text(
        shim_dir / "numba.py",
        textwrap.dedent(
            """
            def njit(*args, **kwargs):
                if args and callable(args[0]):
                    return args[0]
                def deco(fn):
                    return fn
                return deco

            def prange(*args):
                return range(*args)
            """
        ).strip()
        + "\n",
      )
    write_text(
        shim_dir / "tqdm.py",
        textwrap.dedent(
            """
            def tqdm(iterable=None, *args, **kwargs):
                return iterable if iterable is not None else []
            """
        ).strip()
        + "\n",
    )
    write_text(
        shim_dir / "cupy.py",
        'raise ImportError("CuPy disabled by reduced-NLS CPU adapter")\n',
    )
    pyfftw_pkg = shim_dir / "pyfftw"
    pyfftw_pkg.mkdir(parents=True, exist_ok=True)
    write_text(
        pyfftw_pkg / "__init__.py",
        textwrap.dedent(
            """
            import numpy as np

            simd_alignment = 16

            class _Config:
                NUM_THREADS = 1
                PLANNER_EFFORT = "FFTW_MEASURE"

            config = _Config()

            class _Cache:
                @staticmethod
                def enable():
                    return None

            class _Interfaces:
                cache = _Cache()

            interfaces = _Interfaces()

            def zeros_aligned(shape, dtype=complex, n=None):
                return np.zeros(shape, dtype=dtype)

            def empty_aligned(shape, dtype=complex, n=None):
                return np.empty(shape, dtype=dtype)

            def import_wisdom(wisdom):
                return None

            def export_wisdom():
                return ()

            class FFTW:
                def __init__(self, input_array, output_array, direction, threads=1, axes=None):
                    self.direction = direction
                    self.axes = axes

                def __call__(self, input_array=None, output_array=None, normalise_idft=False):
                    if self.direction == "FFTW_FORWARD":
                        out = np.fft.fftn(input_array, axes=self.axes)
                    else:
                        out = np.fft.ifftn(input_array, axes=self.axes)
                    output_array[...] = out
                    return output_array
            """
        ).strip()
        + "\n",
    )


def write_external_runner(path: Path) -> None:
    write_text(
        path,
        textwrap.dedent(
            r"""
            import csv
            import json
            import math
            import sys
            from pathlib import Path

            import numpy as np

            shim_dir = Path(sys.argv[1])
            source_root = Path(sys.argv[2])
            out_dir = Path(sys.argv[3])
            config_path = Path(sys.argv[4])
            sys.path.insert(0, str(shim_dir))
            sys.path.insert(1, str(source_root))

            from NLSE import NLSE_1d

            config = json.loads(config_path.read_text())
            rows = []
            histories = {}

            def periodic_centroid(x, rho, L):
                theta = 2.0 * np.pi * x / L
                z = np.sum(rho * np.exp(1j * theta)) / np.sum(rho)
                angle = np.angle(z)
                if angle < 0:
                    angle += 2.0 * np.pi
                return float(angle * L / (2.0 * np.pi))

            def unwrap_positions(pos, L):
                out = [pos[0]]
                for p in pos[1:]:
                    q = p
                    while q - out[-1] > L / 2:
                        q -= L
                    while q - out[-1] < -L / 2:
                        q += L
                    out.append(q)
                return np.array(out, dtype=float)

            for kval in config["k_values"]:
                N = int(config["N"])
                L = float(config["L"])
                dt = float(config["dt"])
                T = float(config["T"])
                # NLSE package linear coefficient is D = 1 / (2*k0). Choose wavelength=4*pi so k0=0.5 and D=1.
                simu = NLSE_1d(
                    alpha=0.0,
                    power=1.0,
                    window=L,
                    n2=1e-12,
                    V=None,
                    L=T,
                    NX=N,
                    Isat=np.inf,
                    wvl=4.0 * np.pi,
                    backend="CPU",
                )
                simu.n2 = 0.0
                simu.delta_z = dt
                simu.propagator = simu._build_propagator()
                x = simu.X.astype(np.float64)
                x0 = 0.25 * L
                sigma = float(config["sigma"])
                psi0 = np.exp(-0.5 * ((x - x0) / sigma) ** 2) * np.exp(1j * kval * x)
                psi0 = psi0.astype(np.complex128)
                samples = []

                def cb(s, A, z_total, i, samples, kval):
                    if i % int(config["sample_every"]) != 0:
                        return
                    rho = np.abs(np.asarray(A)) ** 2
                    samples.append(
                        {
                            "t": float((i + 1) * s.delta_z),
                            "centroid": periodic_centroid(x % L, rho, L),
                            "norm": float(np.sum(rho) * s.delta_X),
                            "rho_max": float(np.max(rho)),
                        }
                    )

                rho0 = np.abs(psi0) ** 2
                samples.append(
                    {
                        "t": 0.0,
                        "centroid": periodic_centroid(x % L, rho0, L),
                        "norm": float(np.sum(rho0) * simu.delta_X),
                        "rho_max": float(np.max(rho0)),
                    }
                )
                out = simu.out_field(
                    psi0,
                    T,
                    verbose=False,
                    plot=False,
                    precision="single",
                    normalize=False,
                    callback=cb,
                    callback_args=(samples, kval),
                )
                rho = np.abs(out) ** 2
                final_t = float(math.ceil(T / dt) * dt)
                samples.append(
                    {
                        "t": final_t,
                        "centroid": periodic_centroid(x % L, rho, L),
                        "norm": float(np.sum(rho) * simu.delta_X),
                        "rho_max": float(np.max(rho)),
                    }
                )
                # Drop duplicate sample times while preserving order.
                dedup = []
                seen = set()
                for item in samples:
                    key = round(item["t"], 12)
                    if key not in seen:
                        seen.add(key)
                        dedup.append(item)
                t = np.array([s["t"] for s in dedup], dtype=float)
                c = unwrap_positions([s["centroid"] for s in dedup], L)
                fit = np.polyfit(t, c, 1)
                v = float(fit[0])
                expected = 2.0 * float(config["D"]) * kval
                norm0 = float(dedup[0]["norm"])
                normf = float(dedup[-1]["norm"])
                rows.append(
                    {
                        "source": "DATA-NUM-001 NLSE package via local CPU FFT compatibility shim",
                        "k": kval,
                        "v_measured": v,
                        "v_expected_2Dk": expected,
                        "v_frac": v / expected if expected else None,
                        "mass_ret": normf / norm0 if norm0 else None,
                        "intercept": float(fit[1]),
                        "samples": len(dedup),
                    }
                )
                histories[str(kval)] = dedup

            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "external_nlse_rows.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            (out_dir / "external_nlse_history.json").write_text(json.dumps(histories, indent=2))
            (out_dir / "external_nlse_runtime.json").write_text(
                json.dumps(
                    {
                        "package_version": "2.3.0",
                        "backend": "CPU",
                        "compatibility_shims": ["pyfftw->numpy FFT", "numba->plain Python decorators"],
                        "D_convention": "D = 1/(2*k0); wavelength=4*pi gives k0=0.5, D=1",
                    },
                    indent=2,
                )
            )
            """
        ).strip()
        + "\n",
    )


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    project_result, project_rows = read_project_v2()
    package_metadata = inspect_package_source(SOURCE_ROOT)
    k_values = [float(r["k"]) for r in project_rows]
    config = {
        "N": 512,
        "L": 10.0,
        "D": 1.0,
        "dt": 0.002,
        "T": 1.0,
        "sigma": 0.45,
        "sample_every": 25,
        "k_values": k_values,
        "source_root": str(SOURCE_ROOT),
    }
    write_json(out / "nlse_adapter_config.json", config)
    shim_dir = out / "runtime_shims"
    write_shims(shim_dir)
    runner = out / "run_external_nlse_case.py"
    write_external_runner(runner)
    cmd = [
        sys.executable,
        str(runner),
        str(shim_dir),
        str(SOURCE_ROOT),
        str(out),
        str(out / "nlse_adapter_config.json"),
    ]
    completed = subprocess.run(cmd, cwd=out, text=True, capture_output=True, timeout=120)
    write_text(out / "commands_run.txt", " ".join(cmd) + "\n\nSTDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr)
    warnings: list[str] = [
        "External NLSE package was run from archived source with local CPU compatibility shims because pyfftw/numba/tqdm were not installed in the active venv; CuPy was shadow-disabled inside the child process to keep the package on its CPU path without pyvkfft.",
        "This is a reduced-equation method sanity check only, not empirical validation and not a full IRER comparison.",
    ]
    if completed.returncode != 0:
        result = {
            "final_decision": "NLSE_REDUCED_V2_BLOCKED",
            "failure_reason": f"External runner exited {completed.returncode}",
            "package_metadata": package_metadata,
            "warnings": warnings,
            "provisional_until_claude_review": True,
        }
        write_json(out / "nlse_reduced_v2_result.json", result)
        write_text(out / "warnings.txt", "\n".join(warnings) + "\n")
        write_text(out / "NLSE_REDUCED_V2_COMPARISON_REPORT.md", report_text(result, [], project_result, warnings))
        return 1
    external_rows = read_csv(out / "external_nlse_rows.csv")
    ext_k = np.array([float(r["k"]) for r in external_rows], dtype=float)
    ext_v = np.array([float(r["v_measured"]) for r in external_rows], dtype=float)
    ext_fit = fit_line(ext_k, ext_v)
    ext_fit["expected_slope_2D"] = 2.0
    ext_fit["slope_percent_error"] = 100.0 * abs(ext_fit["slope"] - 2.0) / 2.0
    project_fit = project_result.get("fit_quality", {})
    comparison_rows = []
    for pr, er in zip(project_rows, external_rows):
        comparison_rows.append(
            {
                "k": pr["k"],
                "project_v_measured": pr["v_measured"],
                "external_v_measured": er["v_measured"],
                "expected_v_2Dk": pr["v_expected_2Dk"],
                "project_v_frac": pr["v_frac"],
                "external_v_frac": er["v_frac"],
                "project_mass_ret": pr["mass_ret"],
                "external_mass_ret": er["mass_ret"],
            }
        )
    summary_rows = [
        {
            "source": "project_C2_V2",
            "slope": project_fit.get("slope"),
            "expected_slope_2D": project_fit.get("expected_slope_2D"),
            "slope_percent_error": project_fit.get("slope_percent_error"),
            "intercept": project_fit.get("intercept"),
            "r_squared": project_fit.get("r_squared"),
            "mass_retention_min": project_fit.get("mass_retention_min"),
        },
        {
            "source": "external_NLSE_reduced",
            "slope": ext_fit["slope"],
            "expected_slope_2D": ext_fit["expected_slope_2D"],
            "slope_percent_error": ext_fit["slope_percent_error"],
            "intercept": ext_fit["intercept"],
            "r_squared": ext_fit["r2"],
            "mass_retention_min": float(np.min([float(r["mass_ret"]) for r in external_rows])),
        },
    ]
    decision = (
        "NLSE_REDUCED_V2_COMPARISON_READY"
        if ext_fit["slope_percent_error"] < 1.0 and summary_rows[1]["mass_retention_min"] > 0.99
        else "NLSE_REDUCED_V2_COMPARISON_READY_WITH_GAPS"
    )
    result = {
        "final_decision": decision,
        "created_utc": utc_now(),
        "project_c2_v2": project_fit,
        "external_nlse_fit": ext_fit,
        "comparison_rows": comparison_rows,
          "summary_rows": summary_rows,
          "package_source": str(SOURCE_ROOT),
          "package_metadata": package_metadata,
          "external_nlse_only_no_quantule_simulations": True,
        "no_production_physics_changed": True,
        "no_verdicts_changed": True,
        "no_physical_correspondence_claim": True,
        "reduced_equation_method_sanity_only": True,
        "provisional_until_claude_review": True,
        "warnings": warnings,
    }
    write_csv(out / "nlse_reduced_v2_summary.csv", summary_rows + comparison_rows)
    write_json(out / "nlse_reduced_v2_result.json", result)
    write_text(out / "warnings.txt", "\n".join(warnings) + "\n")
    write_text(out / "NLSE_REDUCED_V2_COMPARISON_REPORT.md", report_text(result, comparison_rows, project_result, warnings))
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def report_text(result: dict[str, Any], comparison_rows: list[dict[str, Any]], project_result: dict[str, Any], warnings: list[str]) -> str:
    metadata = result.get("package_metadata", {})
    lines = [
        "# NLSE Reduced V2 Comparison Report",
        "",
        "Status: provisional until Claude review.",
        "",
        f"Final decision: `{result.get('final_decision')}`",
        "",
        "## Scope",
        "",
        "This is a reduced-equation numerical-method sanity check only. It is not empirical validation, not a full IRER comparison, and not a physical-correspondence claim.",
        "",
        "## External Package Inspection",
        "",
        f"- Source root: `{metadata.get('source_root')}`",
        f"- Package version from setup.py: `{metadata.get('package_version_from_setup')}`",
        f"- Requirements: `{', '.join(metadata.get('requirements', []))}`",
        f"- License file first line: `{metadata.get('license_file_first_line')}`",
        f"- Run mode: `{metadata.get('run_mode')}`",
        "",
        "## Project C2 V2 Reference",
        "",
        f"- Slope: `{project_result.get('fit_quality', {}).get('slope')}`",
        f"- Expected slope 2D: `{project_result.get('fit_quality', {}).get('expected_slope_2D')}`",
        f"- Slope percent error: `{project_result.get('fit_quality', {}).get('slope_percent_error')}`",
        f"- Minimum mass retention: `{project_result.get('fit_quality', {}).get('mass_retention_min')}`",
        "",
        "## External NLSE Reduced Run",
        "",
        f"- Slope: `{result.get('external_nlse_fit', {}).get('slope')}`",
        f"- Expected slope 2D: `{result.get('external_nlse_fit', {}).get('expected_slope_2D')}`",
        f"- Slope percent error: `{result.get('external_nlse_fit', {}).get('slope_percent_error')}`",
        f"- Intercept: `{result.get('external_nlse_fit', {}).get('intercept')}`",
        f"- R^2: `{result.get('external_nlse_fit', {}).get('r2')}`",
        "",
        "## Side-by-Side Rows",
        "",
        "| k | project v | external v | expected v | project mass ret | external mass ret |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['k']} | {row['project_v_measured']} | {row['external_v_measured']} | "
            f"{row['expected_v_2Dk']} | {row['project_mass_ret']} | {row['external_mass_ret']} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        *[f"- {w}" for w in warnings],
        "",
        "## Required Statements",
        "",
        "- External NLSE only; no Quantule Mapper simulations run.",
        "- No production physics changed.",
        "- No verdicts changed.",
        "- No physical-correspondence claims added.",
        "- Result is a reduced-equation method sanity check only.",
        "- Provisional until Claude review.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated external NLSE reduced-NLS V2 sanity comparison.")
    parser.add_argument("--out", default=str(OUT_ROOT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
