"""TG-B2 DEFINITIVE two-node force — dynamical body-force observable to settle the attract/repel sign.

Why: the overnight run's proxies DISAGREE on direction (half-space momentum slope -> repel; COM-separation
differential -> attract) because both are contaminated by the violent 56% breathing. This uses the cleaner
observable Codex and the method assessment call for: the VERIFIED Gravity-D body force evaluated on the LIVE fields,

    F_R(t) = - c^2 * Integral_{x>0} (d_x A(t)) |grad phi(t)|^2 dV        A = exp(a_sign*eps_G*G)

time-averaged over the settled window. F_R < 0 => right node pushed toward centre => ATTRACTION. This
(i) reduces to the static calc when A,phi are frozen, (ii) is NOT the momentum (so not fooled by internal breathing
flux or absorber momentum loss), (iii) has a built-in null: off (A=1) gives F_R = 0 exactly. Sign control: run both
a_sign=+1 (A-well, theory) and -1 (A-hill, frozen); <F_R> must reverse.

Optional light cooling (damped, charge-renormalized preconditioning) can quiet the breathing; default OFF (rely on
long averaging first, the safe/simple path). Reuses frozen TG-B1S dynamics via gravity_TG_B2_two_node_awell.

Colab-fast-lane friendly (self-contained, GPU preflight, standard args). GPU (JAX). Mirror-only; frozen TG-B1S
untouched; no gravity/UFF/IRER claim. Preregister all outcomes (attract/repel/null); do not tune to attraction.
  <python> jax_scout/gravity_TG_B2_definitive_force.py --seps 3.0,4.0 --T 180
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
def diag(state, cfg, g, a_sign, feedback):
    """Primary: dynamical Gravity-D body force on the right node. Plus P_R, sep, charge, amp cross-checks."""
    phi, pi, T, VT, G, VG = state
    c = cfg[1]; eps_G = cfg[12]
    dV = g["dx"] ** 3
    A = jnp.exp(a_sign * eps_G * G * feedback)                     # off: feedback=0 -> A=1 -> F_R=0 (built-in null)
    dxA = jnp.real(jnp.fft.ifftn(g["ikx"] * jnp.fft.fftn(A)))
    gx = jnp.fft.ifftn(g["ikx"] * jnp.fft.fftn(phi))
    gy = jnp.fft.ifftn(g["iky"] * jnp.fft.fftn(phi))
    gz = jnp.fft.ifftn(g["ikz"] * jnp.fft.fftn(phi))
    grad2 = jnp.abs(gx) ** 2 + jnp.abs(gy) ** 2 + jnp.abs(gz) ** 2
    X = g["X"]; rho = jnp.abs(phi) ** 2
    F_R = -c * c * jnp.sum(jnp.where(X > 0, dxA * grad2, 0.0)) * dV     # <0 = toward centre = attraction
    px = -2.0 * jnp.real(jnp.conj(pi) * gx)
    mR = jnp.sum(jnp.where(X > 0, rho, 0.0)) * dV + 1e-30
    mL = jnp.sum(jnp.where(X < 0, rho, 0.0)) * dV + 1e-30
    return {"F_R": F_R,
            "P_R": jnp.sum(jnp.where(X > 0, px, 0.0)) * dV,
            "charge": jnp.imag(jnp.sum(jnp.conj(phi) * pi)) * dV,
            "sep": jnp.sum(jnp.where(X > 0, X * rho, 0.0)) * dV / mR - jnp.sum(jnp.where(X < 0, X * rho, 0.0)) * dV / mL,
            "amp": jnp.max(jnp.abs(phi))}


def cool(state, cfgv, refsv, g, flags, a_sign, gamma, n_chunk, chunk_steps):
    """Optional charge-renormalized damped preconditioning to quiet the breathing (relaxes toward fixed-charge state)."""
    dt = float(cfgv[0]); dV = g["dx"] ** 3
    Q0 = float(jnp.imag(jnp.sum(jnp.conj(state[0]) * state[1])) * dV)

    @jax.jit
    def dstep(s):
        for _ in range(chunk_steps):
            k = b2.rhs_2n(s, cfgv, refsv, g, jnp.asarray(flags, dtype=jnp.float64), jnp.asarray(a_sign, dtype=jnp.float64))
            k = (k[0], k[1] - gamma * s[1], k[2], k[3], k[4], k[5])   # damp phi-acceleration
            s = tuple(y + dt * dy for y, dy in zip(s, k))             # simple Euler is fine for relaxation
        return s
    for _ in range(n_chunk):
        state = dstep(state)
        Q = float(jnp.imag(jnp.sum(jnp.conj(state[0]) * state[1])) * dV)
        lam = (abs(Q0) / (abs(Q) + 1e-30)) ** 0.5
        state = (state[0] * lam, state[1] * lam, state[2], state[3], state[4], state[5])
    return state


def run_arm(csv_path, psi, pi, cfgv, refsv, g, flags, a_sign, feedback, nchunk, chunk_steps, dt, merge_thresh,
            cool_flags, gamma_cool, cool_chunks):
    state = b2.to_dev(psi, pi, g)
    if cool_chunks > 0:
        state = cool(state, cfgv, refsv, g, cool_flags, a_sign, gamma_cool, cool_chunks, chunk_steps)
    rows = []
    fields = ["t", "F_R", "P_R", "charge", "sep", "amp"]
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader()
        t = 0.0
        d = diag(state, cfgv, g, jnp.asarray(a_sign, dtype=jnp.float64), jnp.asarray(feedback, dtype=jnp.float64))
        row = {k: float(d[k]) for k in ["F_R", "P_R", "charge", "sep", "amp"]}; row["t"] = t
        rows.append(row); wr.writerow(row); fh.flush()
        for _ in range(nchunk):
            state = b2.evolve_2n(state, cfgv, refsv, g, jnp.asarray(flags, dtype=jnp.float64),
                                 jnp.asarray(a_sign, dtype=jnp.float64), chunk_steps)
            t += chunk_steps * dt
            d = diag(state, cfgv, g, jnp.asarray(a_sign, dtype=jnp.float64), jnp.asarray(feedback, dtype=jnp.float64))
            row = {k: float(d[k]) for k in ["F_R", "P_R", "charge", "sep", "amp"]}; row["t"] = t
            rows.append(row); wr.writerow(row); fh.flush()
            if row["sep"] < merge_thresh or not np.isfinite(row["sep"]):
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seps", default="3.0,4.0")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--T", type=float, default=180.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--sample-dt", type=float, default=0.5)
    ap.add_argument("--f-discard", type=float, default=0.4)
    ap.add_argument("--merge-thresh", type=float, default=1.2)
    ap.add_argument("--q-tol", type=float, default=1e-2)
    ap.add_argument("--cool-T", type=float, default=0.0)        # 0 = no cooling (default, safe). e.g. 40 to precondition
    ap.add_argument("--gamma-cool", type=float, default=0.3)
    for k, v in dict(c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1, w=0.964, alpha_T=0.35, omega_T=1.25, omega_G=0.85,
                     gamma_T=0.08, gamma_G=0.06, kappa_TG=0.55, epsilon_G=0.06, cT=0.7, cG=0.55,
                     absorb_width=1.6, absorb_strength=0.02, core_radius=2.0).items():
        ap.add_argument(f"--{k.replace('_','-')}", type=float, default=v)
    args = ap.parse_args()

    seps = [float(x) for x in args.seps.split(",") if x.strip()]
    out = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B2_DEFINITIVE_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    pf = b2.preflight()
    t0 = time.time()
    print(f"=== TG-B2 definitive force (body-force observable) | {pf['backend']} {pf['devices']} x64={pf['x64']} | out={out} ===", flush=True)
    write_json(out / "config.json", {"args": vars(args), "seps": seps, "preflight": pf,
               "observable": "PRIMARY: <F_R> = time-avg of -c^2 Int_{x>0} d_x A |grad phi|^2 dV over settled window; F_R<0=attract; off(A=1)=0 null; well/hill antisymmetry control",
               "boundary": "mirror-only; frozen TG-B1S untouched; no gravity/UFF/IRER claim"})
    if not pf["gpu_ok"]:
        write_json(out / "RUN_FAILED.json", {"status": "GPU_PREFLIGHT_FAILED", **pf}); print("GPU_PREFLIGHT_FAILED"); return

    cfg = vars(args).copy()
    phi, prof = b1s.solve_qball(cfg)
    op = build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    refs = b1s.ref_source_norms(phi, cfg, op)          # consistent single-node B1S normalization
    g = b1s.make_grid(op); cfgv = b1s.cfg_array(cfg); refsv = b1s.refs_array(refs)
    chunk_steps = max(1, int(round(args.sample_dt / args.dt)))
    nchunk = max(8, int(round(args.T / args.dt / chunk_steps)))
    cool_chunks = int(round(args.cool_T / args.dt / chunk_steps)) if args.cool_T > 0 else 0
    print(f"[setup] resid={prof['residual']:.1e}; seps={seps}; T={args.T} ({nchunk} samples/arm); "
          f"cool_T={args.cool_T}; discard first {args.f_discard:.0%}", flush=True)

    FULL = [1.0, 1.0, 1.0, 1.0]; OFF = [1.0, 1.0, 1.0, 0.0]
    rows_out = []
    for sep in seps:
        rid = f"sep{sep:.2f}"
        write_json(out / f"ROW_{rid}_STARTED.json", {"sep": sep, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        try:
            psi, pi, sc = b2.place_two(phi, cfg, sep, 0.0)
            arms = {}
            for name, flags, a, fb in (("off", OFF, -1.0, 0.0), ("well", FULL, +1.0, 1.0), ("hill", FULL, -1.0, 1.0)):
                print(f"[{rid}] arm {name} ...", flush=True)
                arms[name] = run_arm(out / f"scalars_{rid}_{name}.csv", psi, pi, cfgv, refsv, g, flags, a, fb,
                                     nchunk, chunk_steps, args.dt, args.merge_thresh, FULL, args.gamma_cool, cool_chunks)
            n = min(len(v) for v in arms.values())
            t = np.array([r["t"] for r in arms["off"][:n]]); m = t >= args.f_discard * t[-1]
            FR = {k: np.array([r["F_R"] for r in v[:n]]) for k, v in arms.items()}
            avg = {k: float(FR[k][m].mean()) for k in arms}
            std = {k: float(FR[k][m].std()) for k in arms}
            def dq(v):
                q = np.array([r["charge"] for r in v[:n]]); return float(abs(q[-1] - q[0]) / (abs(q[0]) + 1e-30))
            q_ok = all(dq(v) < args.q_tol for v in arms.values())
            sep_min = min(float(np.min([r["sep"] for r in v[:n]])) for v in arms.values())
            distinct = sep_min > args.merge_thresh
            off_null = abs(avg["off"]) < 0.1 * max(abs(avg["well"]), 1e-30) or abs(avg["off"]) < 1e-9
            well_attracts = avg["well"] < 0
            sign_reverses = avg["well"] * avg["hill"] < 0
            res = {"sep": sep, "n": int(n), "t_end": float(t[-1]),
                   "F_R_well_avg": avg["well"], "F_R_hill_avg": avg["hill"], "F_R_off_avg": avg["off"],
                   "F_R_well_std": std["well"], "F_R_hill_std": std["hill"],
                   "well_attracts": bool(well_attracts), "sign_reverses": bool(sign_reverses),
                   "off_null_ok": bool(off_null), "charge_ok": bool(q_ok), "sep_min": sep_min, "nodes_distinct": bool(distinct),
                   "gates_pass": bool(q_ok and distinct and off_null)}
            rows_out.append(res)
            write_json(out / f"ROW_{rid}_COMPLETE.json", res)
            print(f"[{rid}] <F_R_well>={avg['well']:+.3e}+-{std['well']:.1e} ({'ATTRACT' if well_attracts else 'REPEL'}) "
                  f"<F_R_hill>={avg['hill']:+.3e} <F_R_off>={avg['off']:+.2e} reverses={sign_reverses} "
                  f"gates_pass={res['gates_pass']} (Q_ok={q_ok} distinct={distinct} off_null={off_null})", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            write_json(out / f"ROW_{rid}_FAILED.json", {"sep": sep, "error": str(e), "trace": traceback.format_exc()})
            print(f"[{rid}] FAILED: {e}", flush=True)

    good = [r for r in rows_out if r["gates_pass"]]
    if good and all(r["sign_reverses"] for r in good) and all(r["well_attracts"] for r in good):
        verdict = "TG_B2_DEFINITIVE_AWELL_ATTRACTION_CONFIRMED"
    elif good and all(r["sign_reverses"] for r in good) and all(not r["well_attracts"] for r in good):
        verdict = "TG_B2_DEFINITIVE_AWELL_REPULSION_CONFIRMED"
    elif good:
        verdict = "TG_B2_DEFINITIVE_SIGN_INCONSISTENT_ACROSS_SEP"
    else:
        verdict = "TG_B2_DEFINITIVE_NO_GATE_PASSING_ROW"
    elapsed = (time.time() - t0) / 3600.0
    write_json(out / "summary.json", {"verdict": verdict, "rows": rows_out, "elapsed_hours": elapsed, "config": cfg, "seps": seps,
               "note": "PRIMARY observable = time-averaged dynamical Gravity-D body force <F_R>; off(A=1) null; well/hill antisymmetry control. F_R<0=attraction.",
               "boundary": "mirror-only; frozen TG-B1S untouched; no gravity/UFF/IRER claim"})
    write_json(out / "RUN_COMPLETE.json", {"verdict": verdict, "elapsed_hours": elapsed,
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"\n=== {verdict} | elapsed {elapsed:.2f}h ===", flush=True)


if __name__ == "__main__":
    main()
