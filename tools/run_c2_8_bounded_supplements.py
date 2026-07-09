"""Bounded diagnostic supplements for an existing C2.8 two-node run.

Uses the saved isolated soliton profile from the completed run. Diagnostic-only;
does not alter production physics or the canonical C2.8 harness.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_scout.phase_d_c2_8_two_node import FAM, build, classify_collision, evolve_pair, place  # noqa: E402


def run(args: argparse.Namespace) -> None:
    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    phi_iso = np.load(source / "phi_iso.npy")
    N = int(args.N)
    L = float(args.L)
    dt = float(args.dt)
    dtchunk = int(args.dtchunk)
    ops = build(N, L, dt)
    x = np.linspace(-L / 2.0, L / 2.0, N, endpoint=False)
    Xax, _, _ = np.meshgrid(x, x, x, indexing="ij")
    results = {
        "source_run": str(source),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "jax_devices": [str(d) for d in jax.devices()],
        "rules": [
            "diagnostic supplement only",
            "not canonical C2.8",
            "uses saved phi_iso.npy",
            "no production physics changes",
        ],
    }

    # A. Static phase midpoint.
    psi_mid = (place(phi_iso, -3.0, N, L) + np.exp(1j * (np.pi / 2.0)) * place(phi_iso, +3.0, N, L)).astype(np.complex128)
    print("[SUPP A] static pair sep=6 dphi=pi/2, T=20", flush=True)
    r_mid = evolve_pair(psi_mid, ops, N, L, dt, 20.0, dtchunk, "supp_static_phase_pi2", str(out))
    tr_mid = r_mid["traj"]
    seps_mid = [(q["t"], q["sep"]) for q in tr_mid if q["sep"] is not None]
    s0_mid = seps_mid[0][1] if seps_mid else None
    send_mid = seps_mid[-1][1] if seps_mid else None
    if tr_mid[-1]["n_peaks"] == 1:
        trend_mid = "MERGED"
    elif send_mid is not None and s0_mid is not None and send_mid < s0_mid - 0.5:
        trend_mid = "ATTRACT"
    elif send_mid is not None and s0_mid is not None and send_mid > s0_mid + 0.5:
        trend_mid = "REPEL"
    else:
        trend_mid = "HOLD"
    print(f"    -> {trend_mid} | sep {s0_mid} -> {send_mid} n_end={tr_mid[-1]['n_peaks']} mass_end={tr_mid[-1]['mass']}", flush=True)
    results["supp_static_phase_pi2"] = {
        "trend": trend_mid,
        "sep_start": s0_mid,
        "sep_end": send_mid,
        "traj_tail": tr_mid[-5:],
    }

    # B. Non-canonical identity helper: tiny amplitude asymmetry in n=2 head-on.
    n = 2
    D = FAM["D"]
    k = 2.0 * np.pi * n / L
    v = 2.0 * D * k
    t_meet = 10.0 / (2.0 * v)
    Tphys = min(t_meet + 10.0, 40.0)
    psi_asym = (
        1.01 * place(phi_iso, -5.0, N, L) * np.exp(1j * k * (Xax + 5.0))
        + 0.99 * place(phi_iso, +5.0, N, L) * np.exp(-1j * k * (Xax - 5.0))
    ).astype(np.complex128)
    print("[SUPP B] diagnostic head-on n=2 with 1.01/0.99 amplitude asymmetry, T={:.1f}".format(Tphys), flush=True)
    r_asym = evolve_pair(psi_asym, ops, N, L, dt, Tphys, dtchunk, "supp_headon_n2_amp_asym", str(out))
    verdict = classify_collision(r_asym)
    tr_asym = r_asym["traj"]
    seps_asym = [q["sep"] for q in tr_asym if q["sep"] is not None]
    print(
        f"    -> {verdict} | min_sep={min(seps_asym) if seps_asym else 'na'} "
        f"n_end={tr_asym[-1]['n_peaks']} mass_end={tr_asym[-1]['mass']}",
        flush=True,
    )
    results["supp_headon_n2_amp_asym"] = {
        "neutral_label": "DIAGNOSTIC_ASYMMETRIC_TWO_CORES_SURVIVE_CLOSE_ENCOUNTER"
        if tr_asym[-1]["n_peaks"] == 2
        else "DIAGNOSTIC_ASYMMETRIC_AMBIGUOUS",
        "verdict_from_existing_classifier": verdict,
        "v_each": v,
        "min_sep": min(seps_asym) if seps_asym else None,
        "traj_tail": tr_asym[-5:],
        "caveat": "Non-canonical amplitude asymmetry diagnostic; not a replacement for labelled identity tracking.",
    }

    (out / "supplement_summary.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"C2_8_SUPPLEMENTS_DONE {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--dtchunk", type=int, default=1000)
    run(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
