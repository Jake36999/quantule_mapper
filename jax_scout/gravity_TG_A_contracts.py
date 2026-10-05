"""TG-A temporal-geometric feedback contracts.

This is a documentation-and-contract generator for the first TG feedback model.
It does not run a spatial node simulation.  It selects one explicit model class,
records the equations, verifies the linear T/G stability condition, validates
the proposed resolution-source nulls on synthetic controls, and writes the
TG-A run artifacts used by the gravity-maturity docs.

Chosen model class:
    explicitly budgeted dissipative response model

The T/G exchange uses a reciprocal quadratic coupling with positive-definite
energy when |kappa_TG| < omega_T * omega_G.  Damping and source work are
accounted explicitly rather than hidden in a conservative claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_state() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception as exc:
        return f"git status failed: {exc}\n"


def command_line() -> str:
    return " ".join([sys.executable, *sys.argv])


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_model_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    omega_t = float(cfg["omega_T"])
    omega_g = float(cfg["omega_G"])
    kappa = float(cfg["kappa_TG"])
    eps_g = float(cfg["epsilon_G"])
    stable_margin = omega_t * omega_g - abs(kappa)
    return {
        "stage": "TG-A",
        "model_class": "explicitly_budgeted_dissipative_response",
        "rationale": (
            "The first formal model keeps damping and boundary/export losses in an explicit ledger. "
            "The T/G exchange itself is reciprocal and uses a positive-definite quadratic energy "
            "when |kappa_TG| < omega_T * omega_G."
        ),
        "fields": ["phi", "T", "G"],
        "no_separate_radiation_field": True,
        "positive_maps": {
            "N_t(T)": "not applied to phi in TG-B1 pilot; T is response/chronology-load telemetry",
            "A_s(G)": f"exp(-{eps_g} * G)",
        },
        "spatial_pilot_equations": {
            "phi_t": "Pi",
            "Pi_t": "d_x(A_s(G) d_x phi) - m^2 phi - gamma_abs(x) Pi",
            "T_t": "V_T",
            "V_T_t": "c_T^2 T_xx - omega_T^2 T - gamma_T V_T + alpha_T R_res - kappa_TG G - gamma_abs(x) V_T",
            "G_t": "V_G",
            "V_G_t": "c_G^2 G_xx - omega_G^2 G - gamma_G V_G - kappa_TG T - gamma_abs(x) V_G",
        },
        "reduced_TG_energy_density": (
            "0.5 V_T^2 + 0.5 c_T^2 |grad T|^2 + 0.5 omega_T^2 T^2 + "
            "0.5 V_G^2 + 0.5 c_G^2 |grad G|^2 + 0.5 omega_G^2 G^2 + kappa_TG T G"
        ),
        "ledger_identity": "dE_TG/dt = integral alpha_T R_res V_T dV - integral gamma_T V_T^2 dV - integral gamma_G V_G^2 dV - boundary_absorption + numerical_residual",
        "linear_stability_condition": "|kappa_TG| < omega_T * omega_G",
        "linear_stability_margin": stable_margin,
        "linear_stability_pass": stable_margin > 0.0,
        "flat_state": "phi=0, Pi=0, T=0, V_T=0, G=0, V_G=0",
        "coupling_off_limits": {
            "source_off": "R_res=0 leaves T/G at homogeneous null if initialized at null",
            "temporal_off": "alpha_T=0 removes T forcing and downstream G response",
            "geometric_off": "kappa_TG=0 or G disabled retains T response but removes G-mediated phi feedback",
            "feedback_off": "evolve T/G but set epsilon_G=0 in A_s(G)",
        },
        "parameters": cfg,
        "config_hash": config_hash(cfg),
    }


def make_source_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "TG-A",
        "primary_source": {
            "name": "R_coh",
            "definition": "R_coh = max(-(K_phase(t)-K_phase(t-dt))/dt, 0)",
            "K_phase": "integral w_core rho |grad theta|^2 / integral w_core rho, implemented locally as j^2/(rho+eps)",
            "meaning": "temporal load is generated when phase-gradient cost decreases, i.e. a coherence-transition proxy",
            "non_tautology": "does not use outgoing flux, T, G, or desired pulse timing",
        },
        "secondary_source": {
            "name": "R_thr",
            "definition": "sigmoid((P-P_c)/delta_P) * max((P(t)-P(t-dt))/dt, 0)",
            "P": "bounded PAS proxy from core-weighted density/energy concentration",
            "P_c": cfg["P_c"],
            "delta_P": cfg["delta_P"],
            "interpretation_boundary": "threshold-source positives cannot support the loop unless R_coh or an action-derived source reproduces the qualitative sequence",
        },
        "required_nulls": [
            "static_node",
            "flat_field",
            "phase_scrambled_density_matched",
            "source_removed",
            "source_sign_reversed_where_meaningful",
        ],
    }


def finite_difference_source(values: np.ndarray, dt: float, mode: str) -> np.ndarray:
    deriv = np.zeros_like(values)
    deriv[1:] = (values[1:] - values[:-1]) / dt
    if mode == "coh":
        return np.maximum(-deriv, 0.0)
    if mode == "thr":
        pc = 0.5
        delta = 0.08
        gate = 1.0 / (1.0 + np.exp(-(values - pc) / delta))
        return gate * np.maximum(deriv, 0.0)
    raise ValueError(mode)


def source_null_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    dt = float(cfg["source_dt"])
    t = np.arange(0.0, float(cfg["source_T"]) + 0.5 * dt, dt)
    k_static = np.full_like(t, 0.25)
    k_flat = np.zeros_like(t)
    k_coherence_gain = 0.25 + 0.75 * np.exp(-t / 3.0)
    # Same density envelope but a static scrambled phase field: no coherence
    # transition in time, so the source should remain null.
    k_scrambled = np.full_like(t, 0.55)
    p_threshold = 1.0 - np.exp(-t / 3.0)
    cases = [
        ("flat_field", k_flat, "coh", True),
        ("static_node", k_static, "coh", True),
        ("coherence_transition", k_coherence_gain, "coh", False),
        ("phase_scrambled_density_matched", k_scrambled, "coh", True),
        ("threshold_transition", p_threshold, "thr", False),
        ("source_removed", np.zeros_like(t), "coh", True),
    ]
    rows: list[dict[str, Any]] = []
    for name, values, mode, expected_null in cases:
        source = finite_difference_source(values, dt, mode)
        rows.append(
            {
                "case": name,
                "source_family": "R_coh" if mode == "coh" else "R_thr",
                "expected_null": expected_null,
                "integrated_source": float(np.trapezoid(source, t)),
                "peak_source": float(np.max(source)),
                "null_pass": bool((np.max(source) < 1e-12) if expected_null else (np.max(source) > 1e-6)),
            }
        )
    return rows


def linear_mode_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    omega_t = float(cfg["omega_T"])
    omega_g = float(cfg["omega_G"])
    kappa = float(cfg["kappa_TG"])
    # For q=(T,G), K=[[omega_T^2,kappa],[kappa,omega_G^2]].
    K = np.array([[omega_t**2, kappa], [kappa, omega_g**2]], dtype=float)
    eig = np.linalg.eigvalsh(K)
    rows = []
    for i, lam in enumerate(eig):
        rows.append(
            {
                "mode": i,
                "omega_squared": float(lam),
                "omega": float(math.sqrt(lam)) if lam > 0 else float("nan"),
                "stable": bool(lam > 0),
            }
        )
    return rows


def tool_audit_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "tool": "numpy",
            "purpose": "TG-A contract checks, synthetic source nulls, TG-B0 ODE integration",
            "version": np.__version__,
            "selected": "yes",
            "reason": "small analytic arrays and deterministic reduced models do not require GPU",
            "analytic_validation_case": "linear T/G eigenvalues and ledger residuals",
        },
        {
            "tool": "jax",
            "purpose": "TG-B1 full field pilot only",
            "version": "recorded by TG-B1 preflight",
            "selected": "yes-for-field-evolution",
            "reason": "full field evolution should use the established GPU mirror discipline",
            "analytic_validation_case": "flat KG dispersion and source-off null",
        },
        {
            "tool": "scipy/sympy",
            "purpose": "optional future symbolic derivation/eigensolver checks",
            "version": "not required for this TG-A run",
            "selected": "no-for-current-contract",
            "reason": "closed-form T/G stability is available from a 2x2 stiffness matrix",
            "analytic_validation_case": "not used",
        },
        {
            "tool": "existing C3 KG scripts",
            "purpose": "node lineage and baseline KG/Q-ball reference",
            "version": "repo-local",
            "selected": "reference-only",
            "reason": "TG-B1 pilot uses a reduced 1D KG node, but references C3 as the validated KG substrate family",
            "analytic_validation_case": "C3 energy/charge conservation and Q-ball transport reports",
        },
    ]
    return rows


def write_docs(outdir: Path, model_spec: dict[str, Any], source_spec: dict[str, Any], source_rows: list[dict[str, Any]], mode_rows: list[dict[str, Any]]) -> None:
    docdir = ROOT / "docs" / "gravity_maturity"
    docdir.mkdir(parents=True, exist_ok=True)
    label = "TG_A_ACTION_AND_SOURCE_AUDIT_CLOSED" if model_spec["linear_stability_pass"] and all(r["null_pass"] for r in source_rows) else "TG_ACTION_MODEL_NOT_WELL_POSED"
    results_text = "\n".join(
            [
                "# TG-A Action And Source Audit Results",
                "",
                "Timestamp: 2026-07-14.",
                f"Run directory: `{outdir.as_posix()}`.",
                f"Status: `{label}`.",
                "",
                "## Model Class",
                "",
                "Selected: explicitly budgeted dissipative response model with reciprocal T/G exchange.",
                "",
                "Energy ledger:",
                "",
                "```text",
                str(model_spec["ledger_identity"]),
                "```",
                "",
                "The T/G quadratic energy is positive in the tested regime because "
                f"`|kappa_TG| < omega_T omega_G` with margin `{model_spec['linear_stability_margin']:.6g}`.",
                "",
                "## Resolution Sources",
                "",
                "- Primary continuous source: `R_coh = [-d_t K_phase]_+`.",
                "- Secondary threshold source: `R_thr = sigmoid((P-P_c)/delta_P)[d_t P]_+`.",
                "- Neither source uses outgoing flux, T/G pulse timing, or node outcome as input.",
                "",
                "## Source Nulls",
                "",
                "| case | family | integrated source | peak source | pass |",
                "| --- | --- | ---: | ---: | --- |",
                *[
                    f"| {r['case']} | {r['source_family']} | {r['integrated_source']:.6e} | {r['peak_source']:.6e} | {r['null_pass']} |"
                    for r in source_rows
                ],
                "",
                "## Linear Modes",
                "",
                "| mode | omega^2 | omega | stable |",
                "| --- | ---: | ---: | --- |",
                *[
                    f"| {r['mode']} | {r['omega_squared']:.6e} | {r['omega']:.6e} | {r['stable']} |"
                    for r in mode_rows
                ],
                "",
                "## Gate Decision",
                "",
                "TG-A permits TG-B0 if the status is `TG_A_ACTION_AND_SOURCE_AUDIT_CLOSED`.",
            ]
        )
    (docdir / "TG_A_RESULTS.md").write_text(results_text, encoding="utf-8")
    (docdir / "TG_A_ACTION_AND_SOURCE_AUDIT.md").write_text(results_text, encoding="utf-8")
    write_json(docdir / "TG_A_MODEL_SPEC.json", model_spec)
    write_json(
        docdir / "TG_A_SUMMARY.json",
        {
            "status": label,
            "run_directory": str(outdir),
            "linear_stability_pass": model_spec["linear_stability_pass"],
            "source_nulls_pass": all(r["null_pass"] for r in source_rows),
            "model_class": model_spec["model_class"],
            "primary_source": source_spec["primary_source"]["name"],
            "secondary_source": source_spec["secondary_source"]["name"],
        },
    )
    (docdir / "TG_A_DOCUMENTATION_INPUTS.md").write_text(
        "\n".join(
            [
                "# TG-A Documentation Inputs",
                "",
                "- Final bounded label: `TG_A_ACTION_AND_SOURCE_AUDIT_CLOSED` if accepted after review.",
                "- This is a model audit, not a spatial feedback result.",
                "- TG-A does not validate gravity, photons, objective time dilation, or FMIA wires.",
                "- Use `TG-B0` next; do not jump directly to a full 3D PDE.",
                f"- Artifacts: `{outdir.as_posix()}`.",
                "- Key caveat: explicitly dissipative model, not a full conservative IRER action.",
            ]
        ),
        encoding="utf-8",
    )
    (docdir / "TG_TOOL_AND_METHOD_AUDIT.md").write_text(
        "\n".join(
            [
                "# TG Tool And Method Audit",
                "",
                "Timestamp: 2026-07-14.",
                "",
                "| tool | purpose | version | selected | reason | analytic validation case |",
                "| --- | --- | --- | --- | --- | --- |",
                *[
                    f"| {r['tool']} | {r['purpose']} | {r['version']} | {r['selected']} | {r['reason']} | {r['analytic_validation_case']} |"
                    for r in tool_audit_rows()
                ],
            ]
        ),
        encoding="utf-8",
    )


def artifact_hashes(outdir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.csv":
            rows.append({"path": str(path.relative_to(outdir)), "sha256": sha256_file(path)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--omega-T", type=float, default=1.25)
    parser.add_argument("--omega-G", type=float, default=0.85)
    parser.add_argument("--kappa-TG", type=float, default=0.35)
    parser.add_argument("--gamma-T", type=float, default=0.08)
    parser.add_argument("--gamma-G", type=float, default=0.06)
    parser.add_argument("--alpha-T", type=float, default=0.4)
    parser.add_argument("--epsilon-G", type=float, default=0.08)
    parser.add_argument("--P-c", type=float, default=0.5)
    parser.add_argument("--delta-P", type=float, default=0.08)
    parser.add_argument("--source-dt", type=float, default=0.01)
    parser.add_argument("--source-T", type=float, default=12.0)
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_A_CONTRACTS_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "omega_T": args.omega_T,
        "omega_G": args.omega_G,
        "kappa_TG": args.kappa_TG,
        "gamma_T": args.gamma_T,
        "gamma_G": args.gamma_G,
        "alpha_T": args.alpha_T,
        "epsilon_G": args.epsilon_G,
        "P_c": args.P_c,
        "delta_P": args.delta_P,
        "source_dt": args.source_dt,
        "source_T": args.source_T,
    }
    env = {
        "stage": "TG-A",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "git_commit": git_commit(),
        "command_line": command_line(),
    }
    model_spec = make_model_spec(cfg)
    source_spec = make_source_spec(cfg)
    source_rows = source_null_rows(cfg)
    mode_rows = linear_mode_rows(cfg)
    label = "TG_A_ACTION_AND_SOURCE_AUDIT_CLOSED" if model_spec["linear_stability_pass"] and all(r["null_pass"] for r in source_rows) else "TG_ACTION_MODEL_NOT_WELL_POSED"
    write_json(outdir / "environment_versions.json", env)
    (outdir / "git_state_before.txt").write_text(git_state(), encoding="utf-8")
    write_json(outdir / "model_spec.json", model_spec)
    write_json(outdir / "source_spec.json", source_spec)
    write_json(outdir / "contracts_summary.json", {"status": label, "linear_modes": mode_rows, "source_nulls": source_rows})
    write_csv(outdir / "tool_audit.csv", tool_audit_rows())
    write_csv(outdir / "linear_modes.csv", mode_rows)
    write_csv(outdir / "source_nulls.csv", source_rows)
    (outdir / "TECHNICAL_HANDOFF.md").write_text(
        f"# TG-A Technical Handoff\n\nStatus: `{label}`.\n\nProceed to TG-B0 only if accepted after review.\n",
        encoding="utf-8",
    )
    (outdir / "OPEN_QUESTIONS.md").write_text(
        "\n".join(
            [
                "# TG-A Open Questions",
                "",
                "- Can a conservative action-derived `R_res` be constructed later?",
                "- Does `R_coh` remain meaningful on a validated KG/Q-ball node rather than synthetic controls?",
                "- What tolerance should be used for the TG-B1 energy ledger under absorbing boundaries?",
            ]
        ),
        encoding="utf-8",
    )
    write_docs(outdir, model_spec, source_spec, source_rows, mode_rows)
    write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
    print(json.dumps({"status": label, "outdir": str(outdir), "linear_stability_pass": model_spec["linear_stability_pass"], "source_nulls_pass": all(r["null_pass"] for r in source_rows)}, indent=2))


if __name__ == "__main__":
    main()
