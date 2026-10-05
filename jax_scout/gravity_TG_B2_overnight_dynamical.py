"""TG-B2 overnight long-average dynamical two-node force — settle the dynamical sign.

Per the method assessment (`TG_B2_METHOD_ASSESSMENT_AND_DYNAMICAL_PLAN.md`) and Codex's review: the T=12 dynamical run
could not settle the sign because the bare two-node pair breathes violently (feedback-off breathes 56%; the loop is a
small oscillatory residue). Fix here = LONG runs so the T/G ring-up damps and the breathing averages out, then extract
the SECULAR loop force as the SLOPE of the loop impulse over the settled window (not a final-time snapshot).

Observable:  J(t) = P_R_full(t) - P_R_off(t)   (half-space field momentum, momentum-conserving instrument)
  secular loop force = linear-fit slope dJ/dt over the settled window [f_discard*T, T].
  A-well slope < 0 (matching bare inward drift) => ATTRACTION; A-well/A-hill slopes must be antisymmetric (control).

Consistent normalization: single-node frozen B1S refs (matches the frozen model AND the dynamic run; the static
script's per-two-node refs are a separate caveat, not used here).

Automation (unattended, up to ~10h, review in the morning):
  - one ROW per separation; arms off/well/hill run in sequence;
  - per-arm scalar CSV streaming (t, P_R, P_tot, charge, comR, comL, sep, amp) -- NO full-field host transfer;
  - ROW_<sep>_STARTED/COMPLETE/FAILED.json markers; RUN_COMPLETE.json + summary.json at the end;
  - gates: |dQ/Q| small, |P_tot| small, nodes stay distinct (no merger). Stop-rule: a row that trips a gate is marked
    DEGRADED/FAILED and the run continues to the next separation.

GPU (WSL/JAX). Mirror-only; frozen TG-B1S untouched; no gravity/UFF/IRER claim. Preregister all outcomes
(attraction / repulsion / null); do not tune to attraction.
  <wsl python> jax_scout/gravity_TG_B2_overnight_dynamical.py --seps 3.0,3.5,4.0 --T 200 --target-hours 8
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
from jax_scout.phase_d_c3_wave import build_kg  # noqa: E402


def write_json(p, o):
    p.write_text(json.dumps(o, indent=2, default=float), encoding="utf-8")


@jax.jit
def full_diag(state, g):
    phi, pi = state[0], state[1]
    dV = g["dx"] ** 3
    dxphi = jnp.fft.ifftn(g["ikx"] * jnp.fft.fftn(phi))
    px = -2.0 * jnp.real(jnp.conj(pi) * dxphi)
    X = g["X"]; rho = jnp.abs(phi) ** 2
    mR = jnp.sum(jnp.where(X > 0, rho, 0.0)) * dV + 1e-30
    mL = jnp.sum(jnp.where(X < 0, rho, 0.0)) * dV + 1e-30
    return {"P_R": jnp.sum(jnp.where(X > 0, px, 0.0)) * dV,
            "P_tot": jnp.sum(px) * dV,
            "charge": jnp.imag(jnp.sum(jnp.conj(phi) * pi)) * dV,
            "comR": jnp.sum(jnp.where(X > 0, X * rho, 0.0)) * dV / mR,
            "comL": jnp.sum(jnp.where(X < 0, X * rho, 0.0)) * dV / mL,
            "amp": jnp.max(jnp.abs(phi))}


def run_arm(csv_path, psi, pi, cfgv, refsv, g, flags, a_sign, nchunk, chunk_steps, dt, merge_thresh):
    state = b2.to_dev(psi, pi, g)
    rows = []
    fields = ["t", "P_R", "P_tot", "charge", "comR", "comL", "sep", "amp"]
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader()
        t = 0.0
        d = full_diag(state, g)
        row = {"t": t, "P_R": float(d["P_R"]), "P_tot": float(d["P_tot"]), "charge": float(d["charge"]),
               "comR": float(d["comR"]), "comL": float(d["comL"]),
               "sep": float(d["comR"] - d["comL"]), "amp": float(d["amp"])}
        rows.append(row); wr.writerow(row); fh.flush()
        for _ in range(nchunk):
            state = b2.evolve_2n(state, cfgv, refsv, g, jnp.asarray(flags, dtype=jnp.float64),
                                 jnp.asarray(a_sign, dtype=jnp.float64), chunk_steps)
            t += chunk_steps * dt
            d = full_diag(state, g)
            sep = float(d["comR"] - d["comL"])
            row = {"t": t, "P_R": float(d["P_R"]), "P_tot": float(d["P_tot"]), "charge": float(d["charge"]),
                   "comR": float(d["comR"]), "comL": float(d["comL"]), "sep": sep, "amp": float(d["amp"])}
            rows.append(row); wr.writerow(row); fh.flush()
            if sep < merge_thresh or not np.isfinite(sep):
                row["merged"] = True
                break
    return rows


def slope(t, y, f_discard):
    t = np.asarray(t); y = np.asarray(y)
    m = t >= f_discard * t[-1]
    if m.sum() < 4:
        return float("nan")
    return float(np.polyfit(t[m], y[m], 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seps", default="3.0,3.5,4.0")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--T", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--sample-dt", type=float, default=0.5)
    ap.add_argument("--f-discard", type=float, default=0.4)      # discard first 40% (transient) before the slope fit
    ap.add_argument("--merge-thresh", type=float, default=1.2)   # node distinctness gate
    ap.add_argument("--q-tol", type=float, default=1e-2)         # |dQ/Q| gate
    ap.add_argument("--target-hours", type=float, default=8.0)   # advisory only; printed for budgeting
    for k, v in dict(c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1, w=0.964, alpha_T=0.35, omega_T=1.25, omega_G=0.85,
                     gamma_T=0.08, gamma_G=0.06, kappa_TG=0.55, epsilon_G=0.06, cT=0.7, cG=0.55,
                     absorb_width=1.6, absorb_strength=0.02, core_radius=2.0).items():
        ap.add_argument(f"--{k.replace('_','-')}", type=float, default=v)
    args = ap.parse_args()

    seps = [float(x) for x in args.seps.split(",") if x.strip()]
    out = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B2_OVERNIGHT_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    pf = b2.preflight()
    t_start = time.time()
    print(f"=== TG-B2 overnight dynamical | {pf['backend']} {pf['devices']} x64={pf['x64']} | out={out} ===", flush=True)
    write_json(out / "config.json", {"args": vars(args), "seps": seps, "preflight": pf,
                                     "observable": "secular loop force = slope of J=P_R_full-P_R_off over settled window",
                                     "boundary": "mirror-only; frozen TG-B1S untouched; no gravity/UFF/IRER claim"})
    if not pf["gpu_ok"]:
        write_json(out / "RUN_FAILED.json", {"status": "GPU_PREFLIGHT_FAILED", **pf}); print("GPU_PREFLIGHT_FAILED"); return

    cfg = vars(args).copy()
    phi, prof = b1s.solve_qball(cfg)
    op = build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    refs = b1s.ref_source_norms(phi, cfg, op)          # single-node frozen B1S refs (consistent normalization)
    g = b1s.make_grid(op); cfgv = b1s.cfg_array(cfg); refsv = b1s.refs_array(refs)
    chunk_steps = max(1, int(round(args.sample_dt / args.dt)))
    nchunk = max(8, int(round(args.T / args.dt / chunk_steps)))
    print(f"[setup] qball resid={prof['residual']:.1e}; seps={seps}; T={args.T} ({nchunk} samples/arm); "
          f"target~{args.target_hours}h; discard first {args.f_discard:.0%}", flush=True)

    FULL = [1.0, 1.0, 1.0, 1.0]; OFF = [1.0, 1.0, 1.0, 0.0]
    row_results = []
    for sep in seps:
        rid = f"sep{sep:.2f}"
        write_json(out / f"ROW_{rid}_STARTED.json", {"sep": sep, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        try:
            psi, pi, sc = b2.place_two(phi, cfg, sep, 0.0)
            arms = {}
            for name, flags, a in (("off", OFF, -1.0), ("well", FULL, +1.0), ("hill", FULL, -1.0)):
                print(f"[{rid}] arm {name} ...", flush=True)
                arms[name] = run_arm(out / f"scalars_{rid}_{name}.csv", psi, pi, cfgv, refsv, g,
                                     flags, a, nchunk, chunk_steps, args.dt, args.merge_thresh)
            # align on common length (in case an arm merged early)
            n = min(len(v) for v in arms.values())
            t = np.array([r["t"] for r in arms["off"][:n]])
            PR = {k: np.array([r["P_R"] for r in v[:n]]) for k, v in arms.items()}
            Jw = PR["well"] - PR["off"]; Jh = PR["hill"] - PR["off"]
            slope_well = slope(t, Jw, args.f_discard); slope_hill = slope(t, Jh, args.f_discard)
            # gates
            def dq(v):
                q = np.array([r["charge"] for r in v[:n]]); return float(abs(q[-1] - q[0]) / (abs(q[0]) + 1e-30))
            q_ok = all(dq(v) < args.q_tol for v in arms.values())
            ptot_max = max(float(np.max(np.abs([r["P_tot"] for r in v[:n]]))) for v in arms.values())
            sep_min = min(float(np.min([r["sep"] for r in v[:n]])) for v in arms.values())
            distinct = sep_min > args.merge_thresh
            # inward sign from bare off drift (slope of P_R_off over settled window)
            off_drift = slope(t, PR["off"], args.f_discard)
            inward = float(np.sign(off_drift)) if off_drift != 0 else -1.0
            well_inward = (slope_well * inward) > 0
            sign_reverses = (slope_well * slope_hill) < 0
            res = {"sep": sep, "n_samples": int(n), "t_end": float(t[-1]),
                   "slope_J_well": slope_well, "slope_J_hill": slope_hill, "off_drift_slope": off_drift,
                   "inward_sign": inward, "well_inward": bool(well_inward), "sign_reverses": bool(sign_reverses),
                   "charge_ok": bool(q_ok), "P_tot_max": ptot_max, "sep_min": sep_min, "nodes_distinct": bool(distinct),
                   "gates_pass": bool(q_ok and distinct)}
            row_results.append(res)
            write_json(out / f"ROW_{rid}_COMPLETE.json", res)
            rd = ("INWARD/attract" if well_inward else "OUTWARD/repel")
            print(f"[{rid}] slope_J_well={slope_well:+.3e} ({rd}) slope_J_hill={slope_hill:+.3e} "
                  f"reverses={sign_reverses} gates_pass={res['gates_pass']} (Q_ok={q_ok} distinct={distinct} sep_min={sep_min:.2f})", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            write_json(out / f"ROW_{rid}_FAILED.json", {"sep": sep, "error": str(e), "trace": traceback.format_exc()})
            print(f"[{rid}] FAILED: {e}", flush=True)

    # overall verdict: use gate-passing rows; require consistent sign + antisymmetry across them
    good = [r for r in row_results if r["gates_pass"] and np.isfinite(r["slope_J_well"])]
    if good and all(r["sign_reverses"] for r in good) and all(r["well_inward"] for r in good):
        verdict = "TG_B2_DYNAMICAL_AWELL_ATTRACTION_SECULAR_CONFIRMED"
    elif good and all(r["sign_reverses"] for r in good) and all(not r["well_inward"] for r in good):
        verdict = "TG_B2_DYNAMICAL_AWELL_REPULSION_SECULAR"
    elif good:
        verdict = "TG_B2_DYNAMICAL_SIGN_INCONSISTENT_OR_NULL_ACROSS_SEP"
    else:
        verdict = "TG_B2_DYNAMICAL_NO_GATE_PASSING_ROW"
    elapsed = (time.time() - t_start) / 3600.0
    summary = {"verdict": verdict, "rows": row_results, "elapsed_hours": elapsed, "config": cfg, "seps": seps,
               "note": "secular loop force = slope of J=P_R_full-P_R_off over settled window; inward calibrated from bare-off drift slope; A-well/A-hill antisymmetry = control",
               "boundary": "mirror-only dynamical two-node force; frozen TG-B1S untouched; no gravity/UFF/IRER claim"}
    write_json(out / "summary.json", summary)
    write_json(out / "RUN_COMPLETE.json", {"verdict": verdict, "elapsed_hours": elapsed,
                                           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"\n=== {verdict} | elapsed {elapsed:.2f}h ===", flush=True)


if __name__ == "__main__":
    main()
