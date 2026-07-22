"""TG-B2 characterization matrix — F(sep) range and F(mass) scaling of the ROBUST in-phase attraction.

The one hardened two-node result is the in-phase (Δφ=0, zero-relative-momentum) A-well body-force attraction
(Phase R PASS). Everything at Δφ≠0 is motion-confounded (dynamical alignment sweep). So characterize the clean case:

  F(sep):  <F_R_well> vs separation  -> falloff / screening range (is it short-range Yukawa? fit a length).
  F(mass): <F_R_well> vs node mass (via the Q-ball w-family) at fixed sep -> does the mediated force scale with node
           "mass"? (the gravity-relevant scaling: gravity would give F ~ M1*M2; here both nodes equal -> F ~ M^p, fit p.)

All in-phase, body-force observable (frozen `gravity_TG_B2_definitive_force.diag`), reused unchanged. Automated
row-addressable matrix for an ~8h unattended local-GPU run. Mirror-only; frozen modules; no gravity/UFF/IRER claim.
  <wsl python> jax_scout/gravity_TG_B2_characterization.py --T 180
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s  # noqa: E402
from jax_scout import gravity_TG_B2_two_node_awell as b2  # noqa: E402
from jax_scout import gravity_TG_B2_definitive_force as dfn  # noqa: E402
from jax_scout.phase_d_c3_wave import build_kg  # noqa: E402


def wj(p, o):
    p.write_text(json.dumps(o, indent=2, default=float), encoding="utf-8")


def run_well(csv_path, psi, pi, cfgv, refsv, g, a_sign, nchunk, chunk_steps, dt, merge_thresh):
    state = b2.to_dev(psi, pi, g); rows = []
    FULL = jnp.asarray([1.0, 1.0, 1.0, 1.0])
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["t", "F_R", "sep", "charge", "amp"]); wr.writeheader()
        t = 0.0
        for step in range(nchunk + 1):
            if step > 0:
                state = b2.evolve_2n(state, cfgv, refsv, g, FULL, jnp.asarray(a_sign), chunk_steps)
                t += chunk_steps * dt
            d = dfn.diag(state, cfgv, g, jnp.asarray(a_sign, dtype=jnp.float64), jnp.asarray(1.0))
            row = {"t": t, "F_R": float(d["F_R"]), "sep": float(d["sep"]), "charge": float(d["charge"]), "amp": float(d["amp"])}
            rows.append(row); wr.writerow(row); fh.flush()
            if row["sep"] < merge_thresh or not np.isfinite(row["sep"]):
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--T", type=float, default=180.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--sample-dt", type=float, default=0.5)
    ap.add_argument("--f-discard", type=float, default=0.4)
    ap.add_argument("--merge-thresh", type=float, default=1.2)
    # frozen physics
    for k, v in dict(c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1, alpha_T=0.35, omega_T=1.25, omega_G=0.85,
                     gamma_T=0.08, gamma_G=0.06, kappa_TG=0.55, epsilon_G=0.06, cT=0.7, cG=0.55,
                     absorb_width=1.6, absorb_strength=0.02, core_radius=2.0).items():
        ap.add_argument(f"--{k.replace('_','-')}", type=float, default=v)
    args = ap.parse_args()

    # matrix: (sep, w). F(sep) at w=0.964; F(mass) at sep=3.0 over w-family. (3.0,0.964) shared.
    # w-family masses (N64/L16): 0.945->105.9, 0.955->69.9, 0.964->52.1, 0.972->42.8, 0.980->38.7 (monotone, below VK turn)
    matrix = [("Fsep_sep2.5", 2.5, 0.964), ("Fsep_sep3.0", 3.0, 0.964), ("Fsep_sep4.0", 4.0, 0.964),
              ("Fsep_sep5.0", 5.0, 0.964),
              ("Fmass_w0.945", 3.0, 0.945), ("Fmass_w0.955", 3.0, 0.955),
              ("Fmass_w0.972", 3.0, 0.972), ("Fmass_w0.980", 3.0, 0.980)]

    out = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B2_CHARACTERIZATION_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    pf = b2.preflight(); t0 = time.time()
    print(f"=== TG-B2 characterization (F(sep), F(mass)) | {pf['backend']} {pf['devices']} | out={out} ===", flush=True)
    wj(out / "config.json", {"args": vars(args), "matrix": matrix, "preflight": pf,
        "observable": "in-phase (dphi=0) body force <F_R_well>; F(sep) range + F(mass) scaling",
        "boundary": "mirror-only; frozen modules; no gravity/UFF/IRER claim"})
    if not pf["gpu_ok"]:
        wj(out / "RUN_FAILED.json", {"status": "GPU_PREFLIGHT_FAILED", **pf}); print("GPU_PREFLIGHT_FAILED"); return

    chunk_steps = max(1, int(round(args.sample_dt / args.dt)))
    nchunk = max(8, int(round(args.T / args.dt / chunk_steps)))
    rows_out = []
    for rid, sep, w in matrix:
        wj(out / f"ROW_{rid}_STARTED.json", {"sep": sep, "w": w, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        try:
            cfg = vars(args).copy(); cfg["w"] = w
            phi, prof = b1s.solve_qball(cfg)
            op = build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
            refs = b1s.ref_source_norms(phi, cfg, op)
            g = b1s.make_grid(op); cfgv = b1s.cfg_array(cfg); refsv = b1s.refs_array(refs)
            dV = (cfg["L"] / cfg["N"]) ** 3
            node_mass = float(np.sum(np.abs(phi) ** 2) * dV)   # single-node mass (int rho)
            node_amp = float(np.max(np.abs(phi)))
            psi, pi, sc = b2.place_two(phi, cfg, sep, 0.0)     # in-phase
            print(f"[{rid}] sep={sep} w={w} qball_resid={prof['residual']:.1e} node_mass={node_mass:.4f} ...", flush=True)
            well = run_well(out / f"scalars_{rid}.csv", psi, pi, cfgv, refsv, g, +1.0, nchunk, chunk_steps, args.dt, args.merge_thresh)
            t = np.array([r["t"] for r in well]); m = t >= args.f_discard * t[-1]
            fr = np.array([r["F_R"] for r in well])
            q = np.array([r["charge"] for r in well]); q_ok = bool(abs(q[-1]-q[0])/(abs(q[0])+1e-30) < 1e-2)
            sep_min = float(np.min([r["sep"] for r in well]))
            res = {"row": rid, "sep": sep, "w": w, "node_mass": node_mass, "node_amp": node_amp,
                   "F_R_well_avg": float(fr[m].mean()), "F_R_well_std": float(fr[m].std()),
                   "attracts": bool(fr[m].mean() < 0), "charge_ok": q_ok,
                   "sep_min": sep_min, "nodes_distinct": bool(sep_min > args.merge_thresh),
                   "gates_pass": bool(q_ok and sep_min > args.merge_thresh)}
            rows_out.append(res); wj(out / f"ROW_{rid}_COMPLETE.json", res)
            print(f"[{rid}] <F_R>={res['F_R_well_avg']:+.3e}+-{res['F_R_well_std']:.1e} mass={node_mass:.4f} "
                  f"gates={res['gates_pass']}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            wj(out / f"ROW_{rid}_FAILED.json", {"row": rid, "sep": sep, "w": w, "error": str(e), "trace": traceback.format_exc()})
            print(f"[{rid}] FAILED: {e}", flush=True)

    # fits (gate-passing only)
    good = [r for r in rows_out if r["gates_pass"]]
    fsep = [(r["sep"], abs(r["F_R_well_avg"])) for r in good if r["w"] == 0.964]
    fmass = [(r["node_mass"], abs(r["F_R_well_avg"])) for r in good if abs(r["sep"] - 3.0) < 1e-9]
    fits = {}
    if len(fsep) >= 3:
        s = np.array(sorted(fsep));
        # exponential (Yukawa-ish) fit log|F| ~ -sep/lambda + const, and power log|F|~ p*log sep
        cE = np.polyfit(s[:, 0], np.log(s[:, 1]), 1); fits["screening_length"] = float(-1 / cE[0]) if cE[0] < 0 else None
        cP = np.polyfit(np.log(s[:, 0]), np.log(s[:, 1]), 1); fits["Fsep_power"] = float(cP[0])
    if len(fmass) >= 3:
        mm = np.array(sorted(fmass)); cM = np.polyfit(np.log(mm[:, 0]), np.log(mm[:, 1]), 1)
        fits["Fmass_power"] = float(cM[0])   # F ~ mass^p ; gravity-like (F~M^2 for equal masses) => p~2
    elapsed = (time.time()-t0)/3600.0
    wj(out / "summary.json", {"verdict": "TG_B2_CHARACTERIZATION_COMPLETE", "rows": rows_out, "fits": fits,
        "elapsed_hours": elapsed,
        "note": "F(sep) falloff/screening + F(mass) scaling exponent for the in-phase A-well body force. Fmass_power p: F~M^p; gravity(equal masses) would give p~2.",
        "boundary": "mirror-only; frozen modules; no gravity/UFF/IRER claim"})
    wj(out / "RUN_COMPLETE.json", {"verdict": "TG_B2_CHARACTERIZATION_COMPLETE", "fits": fits, "elapsed_hours": elapsed,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"\n[fits] {fits}\n=== TG_B2_CHARACTERIZATION_COMPLETE | {elapsed:.2f}h ===", flush=True)


if __name__ == "__main__":
    main()
