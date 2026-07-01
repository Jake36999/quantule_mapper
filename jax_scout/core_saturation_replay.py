"""
Row-targeted Phase C replay / validation helper.

Replays a saved Phase C candidate (or the feb56dc7 reference control) with the same IC family,
solver core, and classifier used by core_saturation_search.py, while stamping the replay contract
into a fresh output directory.

WSL2 jax venv example:
  python /mnt/f/quantule_mapper/jax_scout/core_saturation_replay.py \
    --csv /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_004605/all_evals.csv \
    --idx 663 --N 96 --T 6000 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_VALIDATE_.../idx_663
"""
import argparse
import csv
import json
import math
import os
import shlex
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jax_scout import core_saturation_search as css


def resolve_candidate(csv_path, idx=None, ref=None):
    if ref is not None:
        if ref != "feb56dc7":
            raise ValueError(f"Unsupported reference '{ref}'")
        return {
            "idx": None,
            "label": "ref_feb56dc7",
            "ic_blobs": 6,
            "ic_norm": css.IC_NORM_PER_BLOB_FIXED,
            "target_initial_mass": None,
            "ic_seed": css.SEED,
            "params": dict(css.FEB),
            "source": {"kind": "reference", "name": ref, "source_resolution_N": None},
        }
    if csv_path is None or idx is None:
        raise ValueError("Either --ref or both --csv and --idx are required")
    csv_path = Path(csv_path)
    source_resolution_N = None
    summary_path = csv_path.parent / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            source_resolution_N = int(summary.get("N")) if summary.get("N") is not None else None
        except Exception:
            source_resolution_N = None
    with Path(csv_path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["idx"]) != int(idx):
                continue
            return {
                "idx": int(row["idx"]),
                "label": row.get("label", "rand"),
                "ic_blobs": int(row["ic_blobs"]),
                "ic_norm": row.get("ic_norm") or css.IC_NORM_PER_BLOB_FIXED,
                "target_initial_mass": float(row["target_initial_mass"]) if row.get("target_initial_mass") not in (None, "", "None") else None,
                "ic_seed": int(row["ic_seed"]) if row.get("ic_seed") not in (None, "", "None") else css.SEED,
                "params": {name: float(row[name]) for name in css.order},
                "source": {
                    "kind": "csv_row",
                    "csv": str(csv_path),
                    "idx": int(row["idx"]),
                    "source_resolution_N": source_resolution_N,
                },
            }
    raise ValueError(f"Could not find idx={idx} in {csv_path}")


def ensure_outdir(outdir, overwrite=False):
    out = Path(outdir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_replay_ic_seed(candidate, ic_seed_override=None):
    if ic_seed_override is not None:
        return int(ic_seed_override)
    return int(candidate.get("ic_seed", css.SEED))


def resolve_replay_params(candidate, param_overrides=None):
    saved_params = dict(candidate["params"])
    replay_params = dict(saved_params)
    for name, value in (param_overrides or {}).items():
        if name not in replay_params:
            raise ValueError(f"Unknown parameter override '{name}'")
        replay_params[name] = float(value)
    return replay_params, saved_params


def parse_param_override_specs(specs):
    overrides = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Parameter override must be NAME=VALUE, got '{spec}'")
        name, value = spec.split("=", 1)
        name = name.strip()
        if name not in css.order:
            raise ValueError(f"Unknown parameter override '{name}'")
        overrides[name] = float(value)
    return overrides


def build_replay_target_metadata(candidate, replay_target_initial_mass, replay_resolution_N):
    saved_target_initial_mass = candidate.get("target_initial_mass")
    source_resolution_N = None
    if isinstance(candidate.get("source"), dict):
        source_resolution_N = candidate["source"].get("source_resolution_N")
    dx_source = None if source_resolution_N in (None, 0) else float(css.L_ / float(source_resolution_N))
    dx_replay = float(css.L_ / float(replay_resolution_N))
    target_integral_mass = None
    target_raw_mass_replay = None if replay_target_initial_mass is None else float(replay_target_initial_mass)
    if saved_target_initial_mass is not None and dx_source is not None:
        target_integral_mass = float(saved_target_initial_mass) * (dx_source ** 3)

    override_used = replay_target_initial_mass is not None and saved_target_initial_mass is not None and not math.isclose(
        float(replay_target_initial_mass),
        float(saved_target_initial_mass),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    if override_used and target_integral_mass is not None:
        expected_replay_raw = target_integral_mass / (dx_replay ** 3)
        if math.isclose(float(replay_target_initial_mass), expected_replay_raw, rel_tol=1e-9, abs_tol=1e-9):
            mass_scaling_mode = "resolution_scaled_raw_target"
        else:
            mass_scaling_mode = "explicit_raw_target_override"
    else:
        mass_scaling_mode = "exact_saved_raw_target"

    return {
        "saved_target_initial_mass": None if saved_target_initial_mass is None else float(saved_target_initial_mass),
        "replay_target_initial_mass": None if replay_target_initial_mass is None else float(replay_target_initial_mass),
        "mass_scaling_mode": mass_scaling_mode,
        "source_resolution_N": source_resolution_N,
        "replay_resolution_N": int(replay_resolution_N),
        "dx_source": dx_source,
        "dx_replay": dx_replay,
        "target_integral_mass": target_integral_mass,
        "target_raw_mass_replay": target_raw_mass_replay,
    }


def build_replay_contract_metadata(
    candidate,
    *,
    replay_target_meta,
    saved_ic_seed,
    replay_ic_seed,
    saved_params,
    replay_params,
    param_overrides,
):
    return {
        "saved_ic_seed": int(saved_ic_seed),
        "replay_ic_seed": int(replay_ic_seed),
        "saved_params": dict(saved_params),
        "replay_params": dict(replay_params),
        "param_overrides": dict(param_overrides),
        **replay_target_meta,
    }


def resolve_replay_target_initial_mass(candidate, *, replay_resolution_N, target_initial_mass=None, target_initial_mass_override=None):
    if target_initial_mass_override is not None:
        return float(target_initial_mass_override)
    if target_initial_mass is not None:
        return float(target_initial_mass)

    saved_target = candidate.get("target_initial_mass")
    source_resolution_N = None
    if isinstance(candidate.get("source"), dict):
        source_resolution_N = candidate["source"].get("source_resolution_N")
    if (
        saved_target is not None
        and source_resolution_N not in (None, int(replay_resolution_N))
    ):
        raise ValueError(
            "Cross-resolution raw-target replay requires --target-initial-mass-override "
            f"(saved target from N={source_resolution_N}, replay requested at N={int(replay_resolution_N)})."
        )
    return saved_target


def save_replay(outdir, candidate, result, *, N, T, command, replay_contract_meta):
    np.savez_compressed(
        outdir / "probe_data.npz",
        psi0=result["psi0"],
        psi_mid=result["psi_mid"],
        psi_fin=result["psi_fin"],
        energy=result["energy"],
        er=result["er"],
        max_amp=result["max_amp"],
    )
    payload = {
        "candidate": candidate,
        "klass": result["klass"],
        "metrics": result["metrics"],
        "finite": result["finite"],
        "N": N,
        "T": T,
        "L": css.L_,
        "dt": css.DT,
        "ic_seed": candidate["ic_seed"],
        "git_commit": css.get_git_commit(),
        "git_dirty": css.get_git_dirty(),
        "classifier": css.classifier_spec(),
        "parameter_rng_seed": css.PARAM_SEED,
        "ic": css.ic_descriptor([candidate["ic_blobs"]], candidate["ic_seed"], candidate["ic_norm"], candidate["target_initial_mass"]),
        "ic_stats": result["ic_stats"],
        "replay_kind": "RESOLUTION_SCALED_TARGET_REPLAY"
        if replay_contract_meta["mass_scaling_mode"] == "resolution_scaled_raw_target"
        else "EXACT_ROW_REPLAY",
        "saved_target_initial_mass": replay_contract_meta["saved_target_initial_mass"],
        "replay_target_initial_mass": replay_contract_meta["replay_target_initial_mass"],
        "mass_scaling_mode": replay_contract_meta["mass_scaling_mode"],
        "source_resolution_N": replay_contract_meta["source_resolution_N"],
        "replay_resolution_N": replay_contract_meta["replay_resolution_N"],
        "dx_source": replay_contract_meta["dx_source"],
        "dx_replay": replay_contract_meta["dx_replay"],
        "target_integral_mass": replay_contract_meta["target_integral_mass"],
        "target_raw_mass_replay": replay_contract_meta["target_raw_mass_replay"],
        "saved_ic_seed": replay_contract_meta["saved_ic_seed"],
        "replay_ic_seed": replay_contract_meta["replay_ic_seed"],
        "saved_params": replay_contract_meta["saved_params"],
        "replay_params": replay_contract_meta["replay_params"],
        "param_overrides": replay_contract_meta["param_overrides"],
        "replay_target_metadata": replay_contract_meta,
        "command": command,
    }
    with (outdir / "summary.json").open("w") as fh:
        json.dump(payload, fh, indent=2, default=float)


def maybe_save_trace_diagnostic(outdir, candidate, result, *, replay_contract_meta, replay_params, n_snap):
    if n_snap is None or int(n_snap) <= 0:
        return None

    from jax_scout import core_saturation_collapse_diag as diag
    from jax_scout import transfer_diag as td

    N = int(result["N"])
    T = int(result["T"])
    dx = css.L_ / N
    times = np.linspace(0, T, int(n_snap) + 1)
    snaps, finite_last = td.capture_trajectory(
        [replay_params[k] for k in css.order],
        result["psi0"],
        N,
        css.L_,
        css.DT,
        T,
        int(n_snap),
    )
    trace = [diag._snapshot_metrics(np.asarray(psi), replay_params, dx) for psi in snaps]
    diag_candidate = {
        "idx": candidate.get("idx"),
        "run_dir": candidate.get("source", {}).get("csv"),
        "K": candidate["ic_blobs"],
        "klass": result["klass"],
        "ref": candidate.get("label") == "ref_feb56dc7",
    }
    summary = diag._summarize_trace(diag_candidate, times, trace, finite_last=bool(finite_last))
    replay_payload = {
        "label": candidate["label"],
        "ic_blobs": candidate["ic_blobs"],
        "ic_seed": candidate["ic_seed"],
        "ic_norm": candidate["ic_norm"],
        "target_initial_mass": candidate["target_initial_mass"],
        "N": N,
        "T": T,
        "dt": css.DT,
        "initial_mass": result["ic_stats"]["initial_mass"],
        "saved_target_initial_mass": replay_contract_meta["saved_target_initial_mass"],
        "replay_target_initial_mass": replay_contract_meta["replay_target_initial_mass"],
        "mass_scaling_mode": replay_contract_meta["mass_scaling_mode"],
    }
    diagnostic_payload = {
        "candidate": diag_candidate,
        "replay": replay_payload,
        "times": times.tolist(),
        "trace": trace,
        "summary": summary,
    }
    np.savez_compressed(outdir / "frames.npz", psi=np.asarray(snaps, dtype=np.complex64), times=np.asarray(times, dtype=np.float64))
    diag._write_json(outdir / "diagnostic_summary.json", diagnostic_payload)
    return diagnostic_payload


def run_replay(
    *,
    csv_path=None,
    idx=None,
    ref=None,
    N=None,
    T=None,
    outdir=None,
    overwrite=False,
    ic_norm=None,
    target_initial_mass=None,
    target_initial_mass_override=None,
    ic_seed_override=None,
    param_override_specs=None,
    trace_snaps=None,
    command_override=None,
):
    if N is None or T is None or outdir is None:
        raise ValueError("N, T, and outdir are required")
    candidate = resolve_candidate(csv_path, idx=idx, ref=ref)
    outdir_path = ensure_outdir(outdir, overwrite=overwrite)
    replay_ic_norm = ic_norm or candidate["ic_norm"]
    replay_ic_seed = resolve_replay_ic_seed(candidate, ic_seed_override=ic_seed_override)
    param_overrides = parse_param_override_specs(param_override_specs)
    replay_params, saved_params = resolve_replay_params(candidate, param_overrides=param_overrides)
    replay_target_initial_mass = resolve_replay_target_initial_mass(
        candidate,
        replay_resolution_N=N,
        target_initial_mass=target_initial_mass,
        target_initial_mass_override=target_initial_mass_override,
    )
    result = css.run_probe(
        replay_params,
        int(N),
        int(T),
        candidate["ic_blobs"],
        seed=replay_ic_seed,
        ic_norm=replay_ic_norm,
        target_initial_mass=replay_target_initial_mass,
    )
    result["N"] = int(N)
    result["T"] = int(T)
    resolved_candidate = dict(candidate)
    replay_target_meta = build_replay_target_metadata(resolved_candidate, replay_target_initial_mass, int(N))
    replay_contract_meta = build_replay_contract_metadata(
        resolved_candidate,
        replay_target_meta=replay_target_meta,
        saved_ic_seed=int(candidate["ic_seed"]),
        replay_ic_seed=replay_ic_seed,
        saved_params=saved_params,
        replay_params=replay_params,
        param_overrides=param_overrides,
    )
    replay_candidate = dict(resolved_candidate)
    replay_candidate["ic_norm"] = replay_ic_norm
    replay_candidate["target_initial_mass"] = replay_target_initial_mass
    replay_candidate["ic_seed"] = replay_ic_seed
    replay_candidate["params"] = replay_params
    command = command_override or " ".join(sys.argv)
    save_replay(
        outdir_path,
        replay_candidate,
        result,
        N=int(N),
        T=int(T),
        command=command,
        replay_contract_meta=replay_contract_meta,
    )
    diagnostic_payload = maybe_save_trace_diagnostic(
        outdir_path,
        replay_candidate,
        result,
        replay_contract_meta=replay_contract_meta,
        replay_params=replay_params,
        n_snap=trace_snaps,
    )
    return {
        "outdir": outdir_path,
        "candidate": replay_candidate,
        "result": result,
        "diagnostic": diagnostic_payload,
        "replay_contract_meta": replay_contract_meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Path to Phase C all_evals.csv")
    ap.add_argument("--idx", type=int, help="Row index inside --csv")
    ap.add_argument("--ref", choices=["feb56dc7"], help="Replay a built-in reference control")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ic-norm", choices=css.IC_NORM_CHOICES, default=None, help="optional override of saved IC normalization mode")
    ap.add_argument("--target-initial-mass", type=float, default=None, help="optional override for total_mass_fixed target mass")
    ap.add_argument("--target-initial-mass-override", type=float, default=None, help="explicit override of the saved raw target mass")
    ap.add_argument("--ic-seed-override", type=int, default=None, help="optional override of the saved IC seed")
    ap.add_argument("--param-override", action="append", default=[], help="repeatable NAME=VALUE override for a saved parameter")
    ap.add_argument("--trace-snaps", type=int, default=None, help="optional sparse trajectory capture count for diagnostic summary output")
    args = ap.parse_args()

    replay_payload = run_replay(
        csv_path=args.csv,
        idx=args.idx,
        ref=args.ref,
        N=args.N,
        T=args.T,
        outdir=args.out,
        overwrite=args.overwrite,
        ic_norm=args.ic_norm,
        target_initial_mass=args.target_initial_mass,
        target_initial_mass_override=args.target_initial_mass_override,
        ic_seed_override=args.ic_seed_override,
        param_override_specs=args.param_override,
        trace_snaps=args.trace_snaps,
    )
    replay_candidate = replay_payload["candidate"]
    result = replay_payload["result"]
    print(
        f"{replay_candidate['label']} -> {result['klass']} | "
        f"n_fin={result['metrics']['n_fin']} slope={result['metrics']['late_slope']:+.3e} "
        f"er_fin={result['metrics']['er_fin']:.4f}"
    )
    print(replay_payload["outdir"])


if __name__ == "__main__":
    main()
