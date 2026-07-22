"""TG-B2 dynamical alignment sweep — does the quasi-static F(Δφ) = a·cos(Δφ)+b structure survive the dynamics?

The quasi-static MC-3 sweep (`gravity_TG_B2_alignment_sweep.py`) found the loop force decomposes into a DC
(alignment-independent, monopole) attraction `b` + a phase-modulated `a·cos(Δφ)`, with the attract→repel crossover
moving from ~anti-phase (near field) toward π/2 (far field). This is the FALSIFIABLE dynamical test: measure the
dynamical Gravity-D body force `<F_R>` (the definitive observable) at fixed separation across Δφ. Prediction: F_R
weakens monotonically with Δφ and crosses to repulsion near anti-phase; if the dynamical force does NOT show this, the
alignment-modulation reframe fails.

Reuses the FROZEN definitive body-force observable (`gravity_TG_B2_definitive_force.diag`) + b2 dynamics, imported
unchanged. Off (A=1) is a built-in null; well/hill antisymmetry is the sign control. GPU (JAX). Mirror-only; no
gravity/UFF/IRER claim; a candidate reframe to confirm or falsify.
  <wsl python> jax_scout/gravity_TG_B2_dynamical_alignment.py --sep 3.0 --dphis 0,0.5,0.75,0.875,1.0 --T 180
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
from jax_scout import gravity_TG_B2_definitive_force as dfn  # noqa: E402 (frozen; reuse diag)
from jax_scout.phase_d_c3_wave import build_kg  # noqa: E402


def wj(p, o):
    p.write_text(json.dumps(o, indent=2, default=float), encoding="utf-8")


def run_arm(csv_path, psi, pi, cfgv, refsv, g, flags, a_sign, feedback, nchunk, chunk_steps, dt, merge_thresh):
    state = b2.to_dev(psi, pi, g)
    rows = []
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["t", "F_R", "P_R", "charge", "sep", "amp"]); wr.writeheader()
        t = 0.0
        d = dfn.diag(state, cfgv, g, jnp.asarray(a_sign, dtype=jnp.float64), jnp.asarray(feedback, dtype=jnp.float64))
        row = {k: float(d[k]) for k in ["F_R", "P_R", "charge", "sep", "amp"]}; row["t"] = t
        rows.append(row); wr.writerow(row); fh.flush()
        for _ in range(nchunk):
            state = b2.evolve_2n(state, cfgv, refsv, g, jnp.asarray(flags, dtype=jnp.float64),
                                 jnp.asarray(a_sign, dtype=jnp.float64), chunk_steps)
            t += chunk_steps * dt
            d = dfn.diag(state, cfgv, g, jnp.asarray(a_sign, dtype=jnp.float64), jnp.asarray(feedback, dtype=jnp.float64))
            row = {k: float(d[k]) for k in ["F_R", "P_R", "charge", "sep", "amp"]}; row["t"] = t
            rows.append(row); wr.writerow(row); fh.flush()
            if row["sep"] < merge_thresh or not np.isfinite(row["sep"]):
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--sep", type=float, default=3.0)
    ap.add_argument("--dphis", default="0,0.5,0.75,0.875,1.0")  # in units of pi
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--T", type=float, default=180.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--sample-dt", type=float, default=0.5)
    ap.add_argument("--f-discard", type=float, default=0.4)
    ap.add_argument("--merge-thresh", type=float, default=1.2)
    for k, v in dict(c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1, w=0.964, alpha_T=0.35, omega_T=1.25, omega_G=0.85,
                     gamma_T=0.08, gamma_G=0.06, kappa_TG=0.55, epsilon_G=0.06, cT=0.7, cG=0.55,
                     absorb_width=1.6, absorb_strength=0.02, core_radius=2.0).items():
        ap.add_argument(f"--{k.replace('_','-')}", type=float, default=v)
    args = ap.parse_args()

    dphis = [float(x) * np.pi for x in args.dphis.split(",") if x.strip()]
    out = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B2_DYN_ALIGN_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    pf = b2.preflight()
    t0 = time.time()
    print(f"=== TG-B2 dynamical alignment sweep | {pf['backend']} {pf['devices']} | sep={args.sep} | out={out} ===", flush=True)
    wj(out / "config.json", {"args": vars(args), "dphis_over_pi": args.dphis, "preflight": pf,
        "observable": "dynamical Gravity-D body force <F_R> time-avg over settled window; off(A=1)=0 null; well/hill antisym",
        "prediction": "F_R weakens with dphi, crosses to repel near anti-phase (from quasi-static a*cos(dphi)+b)",
        "boundary": "mirror-only; frozen modules; no gravity/UFF/IRER claim"})
    if not pf["gpu_ok"]:
        wj(out / "RUN_FAILED.json", {"status": "GPU_PREFLIGHT_FAILED", **pf}); print("GPU_PREFLIGHT_FAILED"); return

    cfg = vars(args).copy()
    phi, prof = b1s.solve_qball(cfg)
    op = build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    refs = b1s.ref_source_norms(phi, cfg, op)
    g = b1s.make_grid(op); cfgv = b1s.cfg_array(cfg); refsv = b1s.refs_array(refs)
    chunk_steps = max(1, int(round(args.sample_dt / args.dt)))
    nchunk = max(8, int(round(args.T / args.dt / chunk_steps)))
    FULL = [1.0, 1.0, 1.0, 1.0]; OFF = [1.0, 1.0, 1.0, 0.0]
    print(f"[setup] resid={prof['residual']:.1e}; sep={args.sep}; dphis/pi={args.dphis}; T={args.T}", flush=True)

    rows_out = []
    for dphi in dphis:
        rid = f"dphi{dphi/np.pi:.3f}pi"
        wj(out / f"ROW_{rid}_STARTED.json", {"dphi": dphi, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        try:
            psi, pi, sc = b2.place_two(phi, cfg, args.sep, dphi)
            arms = {}
            for name, flags, a, fb in (("well", FULL, +1.0, 1.0), ("hill", FULL, -1.0, 1.0)):
                print(f"[{rid}] arm {name} ...", flush=True)
                arms[name] = run_arm(out / f"scalars_{rid}_{name}.csv", psi, pi, cfgv, refsv, g, flags, a, fb,
                                     nchunk, chunk_steps, args.dt, args.merge_thresh)
            n = min(len(v) for v in arms.values())
            t = np.array([r["t"] for r in arms["well"][:n]]); m = t >= args.f_discard * t[-1]
            fw = np.array([r["F_R"] for r in arms["well"][:n]]); fh = np.array([r["F_R"] for r in arms["hill"][:n]])
            avg_w = float(fw[m].mean()); std_w = float(fw[m].std()); avg_h = float(fh[m].mean())
            q = np.array([r["charge"] for r in arms["well"][:n]]); q_ok = bool(abs(q[-1]-q[0])/(abs(q[0])+1e-30) < 1e-2)
            sep_min = min(float(np.min([r["sep"] for r in v[:n]])) for v in arms.values())
            res = {"dphi": dphi, "dphi_over_pi": dphi/np.pi, "F_R_well_avg": avg_w, "F_R_well_std": std_w,
                   "F_R_hill_avg": avg_h, "well_attracts": bool(avg_w < 0),
                   "antisym_rel": float(abs(avg_w+avg_h)/(abs(avg_w)+1e-30)), "charge_ok": q_ok,
                   "sep_min": sep_min, "nodes_distinct": bool(sep_min > args.merge_thresh)}
            rows_out.append(res); wj(out / f"ROW_{rid}_COMPLETE.json", res)
            print(f"[{rid}] <F_R_well>={avg_w:+.3e}+-{std_w:.1e} ({'ATTRACT' if avg_w<0 else 'REPEL'}) "
                  f"antisym={res['antisym_rel']:.2e} Q_ok={q_ok} distinct={res['nodes_distinct']}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            wj(out / f"ROW_{rid}_FAILED.json", {"dphi": dphi, "error": str(e), "trace": traceback.format_exc()})
            print(f"[{rid}] FAILED: {e}", flush=True)

    good = [r for r in rows_out if r["charge_ok"] and r["nodes_distinct"]]
    Fw = [r["F_R_well_avg"] for r in good]
    weakens = bool(len(Fw) >= 2 and Fw[0] < 0 and Fw[-1] > Fw[0])   # attraction weakens as dphi grows
    crosses = bool(any(a < 0 for a in Fw) and any(a > 0 for a in Fw))
    if weakens and crosses:
        verdict = "TG_B2_DYN_ALIGN_CONFIRMS_QUASISTATIC_WEAKEN_AND_CROSSOVER"
    elif weakens:
        verdict = "TG_B2_DYN_ALIGN_WEAKENS_NO_DYNAMICAL_CROSSOVER"
    elif good:
        verdict = "TG_B2_DYN_ALIGN_NO_CLEAR_ALIGNMENT_TREND"
    else:
        verdict = "TG_B2_DYN_ALIGN_NO_GATE_PASSING_ROW"
    elapsed = (time.time()-t0)/3600.0
    wj(out / "summary.json", {"verdict": verdict, "sep": args.sep, "rows": rows_out, "elapsed_hours": elapsed,
        "note": "dynamical body-force <F_R> vs relative phase; tests the quasi-static a*cos(dphi)+b prediction",
        "boundary": "mirror-only; frozen modules; no gravity/UFF/IRER claim"})
    wj(out / "RUN_COMPLETE.json", {"verdict": verdict, "elapsed_hours": elapsed,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"\n=== {verdict} | elapsed {elapsed:.2f}h ===", flush=True)


if __name__ == "__main__":
    main()
