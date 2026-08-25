"""TG-B2 midplane stress-flux force — the independent second estimator (campaign item P2).

WHY. Every force number in the gravity branch (~75 runs) comes from ONE estimator, the Gravity-D
body force

    F_R = -c^2 Int_{x>0} (d_x A) |grad phi|^2 dV                                        (1)

P1 (docs/gravity_maturity/TG_P1_EVIDENCE_RECONCILIATION.md) established that (1) is a half-space
TOTAL BODY FORCE weighted by the GRADIENT ENERGY ONLY - one of four terms in the KG energy density -
and that it has never been cross-checked against anything. This module supplies the check.

THE IDENTITY. The evolved system (gravity_TG_B2_two_node_awell.rhs_2n) is

    d_t phi = pi
    d_t pi  = c^2 div(A grad phi) - m^2 phi + (a rho + s rho^2 + f rho^3) phi,   rho = |phi|^2

which follows from  L = |pi|^2 - c^2 A |grad phi|^2 - m^2 rho + U(rho),
with  U(rho) = a rho^2/2 + s rho^3/3 + f rho^4/4   (so U'(rho) = a rho + s rho^2 + f rho^3).

Momentum density  p_x = -2 Re(pi* d_x phi)  (identical to the existing P_R integrand). Differentiating
p_x in time and substituting the equations of motion gives the exact local conservation law

    d_t p_x = d_x S + (transverse divergences) - c^2 (d_x A) |grad phi|^2                (2)

with the plane-flux function

    S = c^2 A |grad phi|^2 - 2 c^2 A |d_x phi|^2 - |pi|^2 + m^2 rho - U(rho)             (3)

Integrating (2) over the half-space x>0 of the periodic box (transverse divergences integrate away):

    d/dt P_x(x>0) = Int_{far} S dA - Int_{x=0} S dA + F_R                                (4)

So the flux-based estimator of the same force is

    F_flux = dP_x/dt - [ Int_{far} S dA - Int_{x=0} S dA ]                               (5)

F_flux samples a 2-D PLANE and carries |pi|^2, m^2 rho and U(rho) - terms (1) does not contain at all.
It is genuinely independent, not a rearrangement.

WHAT IS TESTED (preregistered, all outcomes describable in advance):

  G1  OFF-arm null. feedback=0 -> A=1 -> F_R = 0 identically. Then (4) must hold with the flux terms
      alone. This validates the stress tensor (3) with NO involvement of A. If G1 fails, the
      derivation or the discretization is wrong and nothing else here is meaningful.
  G2  Ledger residual. |dP_x/dt - (flux + F_R)| small vs the scale of the individual terms, on the
      live well/hill arms.
  G3  Estimator agreement. <F_flux> vs <F_R> over the settled window, well and hill arms.
  G4  Sign control. <F_flux> must reverse between well and hill, like <F_R> does.

  Failure of G3 while G1/G2 pass would mean the two estimators measure different things - which is
  itself the answer P2 was asking for, and would call the body-force reading into question.

DISCRETIZATION NOTE. Two flux variants are computed because they answer slightly different questions:
  plane : S sampled on the grid planes x=0 (index N/2) and x=-L/2 (index 0). This is the physical
          "momentum crossing the midplane", but the mask boundary sits at x=dx/2, so plane placement
          carries an O(dx) offset.
  div   : -Int_{x>0} d_x S dV evaluated spectrally with the same deriv_x used by the solver. This is
          the discretely exact partner of the mask X>0 and has no placement ambiguity.
The difference between them measures the placement error and is reported, not hidden.

Mirror-only. Frozen TG-B1S dynamics untouched - this module only OBSERVES. No gravity/UFF/IRER claim.

  <wsl python> jax_scout/gravity_TG_B2_midplane_stress_flux.py --sep 3.0 --T 40 --N 64 --L 16
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
def stress_diag(state, cfg, g, a_sign, feedback):
    """All three momentum-ledger terms on the live fields, plus the existing body force.

    Returns the pieces of identity (4) so the residual can be formed offline:
        dP_x/dt = (S_out - S_in) + F_R
    """
    phi, pi, T, VT, G, VG = state
    c, m = cfg[1], cfg[2]
    a, s, f = cfg[3], cfg[4], cfg[5]
    eps_G = cfg[12]
    dx = g["dx"]
    dV = dx ** 3
    dA = dx ** 2

    A = jnp.exp(a_sign * eps_G * G * feedback)          # feedback=0 -> A=1 -> F_R=0 exactly
    gx = b1s.deriv_x(phi, g)
    gy = b1s.deriv_y(phi, g)
    gz = b1s.deriv_z(phi, g)
    grad2 = jnp.abs(gx) ** 2 + jnp.abs(gy) ** 2 + jnp.abs(gz) ** 2
    gx2 = jnp.abs(gx) ** 2
    rho = jnp.abs(phi) ** 2
    X = g["X"]

    # --- potential U(rho) with U'(rho) = a rho + s rho^2 + f rho^3 (matches rhs_2n) -------------
    U = 0.5 * a * rho ** 2 + s * rho ** 3 / 3.0 + 0.25 * f * rho ** 4

    # --- plane-flux function S, eq. (3) ---------------------------------------------------------
    S = c * c * A * grad2 - 2.0 * c * c * A * gx2 - jnp.abs(pi) ** 2 + m * m * rho - U

    # --- existing body force, eq. (1) -- recomputed here so both come from the same fields ------
    dxA = jnp.real(b1s.deriv_x(A, g))
    F_R = -c * c * jnp.sum(jnp.where(X > 0, dxA * grad2, 0.0)) * dV

    # --- momentum in the half space, and its density ---------------------------------------------
    px = -2.0 * jnp.real(jnp.conj(pi) * gx)
    P_R = jnp.sum(jnp.where(X > 0, px, 0.0)) * dV

    # --- flux variant 1: S on the mask INTERFACES -----------------------------------------------
    # The region is the mask X>0, i.e. cells n/2+1 .. n-1. Its faces therefore lie at the cell
    # interfaces x=+dx/2 and (wrapping) x=L/2-dx/2, NOT on grid planes. Sampling S at index n/2
    # (x=0) and index 0 (x=-L/2) misplaces both faces by half a cell and the seam by a whole one,
    # which opens the ledger at O(1). Interpolate to the interfaces instead.
    n = S.shape[0]
    S_in = 0.5 * (jnp.sum(S[n // 2, :, :]) + jnp.sum(S[n // 2 + 1, :, :])) * dA   # x = +dx/2
    S_out = 0.5 * (jnp.sum(S[n - 1, :, :]) + jnp.sum(S[0, :, :])) * dA            # x = L/2 - dx/2
    flux_plane = S_out - S_in
    # physical midplane flux, reported for interpretation (exactly on the symmetry plane x=0)
    S_mid = jnp.sum(S[n // 2, :, :]) * dA
    S_far = jnp.sum(S[0, :, :]) * dA

    # --- flux variant 2: spectral divergence over the same mask (no plane-placement error) -------
    # Int_{x>0} d_x S dV = S_out - S_in by the fundamental theorem: same convention as flux_plane.
    dS = jnp.real(b1s.deriv_x(S, g))
    flux_div = jnp.sum(jnp.where(X > 0, dS, 0.0)) * dV

    return {"F_R": F_R, "P_R": P_R,
            "S_mid": S_mid, "S_far": S_far, "S_in": S_in, "S_out": S_out,
            "flux_plane": flux_plane, "flux_div": flux_div,
            "charge": jnp.imag(jnp.sum(jnp.conj(phi) * pi)) * dV,
            "amp": jnp.max(jnp.abs(phi)),
            "A_min": jnp.min(A), "A_max": jnp.max(A)}


FIELDS = ["t", "F_R", "P_R", "S_mid", "S_far", "S_in", "S_out", "flux_plane", "flux_div",
          "charge", "amp", "A_min", "A_max"]


def run_arm(csv_path, psi, pi, cfgv, refsv, g, flags, a_sign, feedback,
            nchunk, chunk_steps, dt):
    state = b2.to_dev(psi, pi, g)
    a_j = jnp.asarray(a_sign, dtype=jnp.float64)
    fb_j = jnp.asarray(feedback, dtype=jnp.float64)
    fl_j = jnp.asarray(flags, dtype=jnp.float64)
    rows = []
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        t = 0.0
        d = stress_diag(state, cfgv, g, a_j, fb_j)
        row = {k: float(d[k]) for k in FIELDS if k != "t"}
        row["t"] = t
        rows.append(row)
        wr.writerow(row)
        for _ in range(nchunk):
            state = b2.evolve_2n(state, cfgv, refsv, g, fl_j, a_j, chunk_steps)
            t += chunk_steps * dt
            d = stress_diag(state, cfgv, g, a_j, fb_j)
            row = {k: float(d[k]) for k in FIELDS if k != "t"}
            row["t"] = t
            rows.append(row)
            wr.writerow(row)
            fh.flush()
            if not np.isfinite(row["F_R"]):
                break
    return rows


def series(rows, f_discard, which="flux_plane"):
    """Per-sample F_flux and F_R on the settled window, aligned to the interior samples.

    Returned so arms can be differenced SAMPLE BY SAMPLE. The absolute F_flux is a ~1700:1
    cancellation between dP/dt and the flux, so its scatter dwarfs its mean; but that scatter is
    almost entirely common mode across arms (identical ICs), and cancels in the difference.
    """
    t = np.array([r["t"] for r in rows])
    P = np.array([r["P_R"] for r in rows])
    F = np.array([r["F_R"] for r in rows])
    Q = np.array([r[which] for r in rows])
    if len(t) < 5:
        return None
    dPdt = (P[2:] - P[:-2]) / (t[2:] - t[:-2])
    ti = t[1:-1]
    msk = ti >= f_discard * ti[-1]
    return {"t": ti[msk], "F_flux": (dPdt - Q[1:-1])[msk], "F_R": F[1:-1][msk],
            "dPdt": dPdt[msk]}


def stat(x):
    x = np.asarray(x, dtype=float)
    n = max(len(x), 1)
    sem = float(x.std() / np.sqrt(n))
    return {"mean": float(x.mean()), "std": float(x.std()), "sem": sem, "n": int(n),
            "sigma": float(abs(x.mean()) / sem) if sem > 0 else None}


def ledger(rows, f_discard, which="flux_plane"):
    """Form the momentum-ledger residual for one arm.

    dP_x/dt is a centred difference of the sampled P_R series, so it is only defined on the
    interior samples; the flux and body-force terms are aligned to those same samples.
    """
    t = np.array([r["t"] for r in rows])
    P = np.array([r["P_R"] for r in rows])
    F = np.array([r["F_R"] for r in rows])
    Q = np.array([r[which] for r in rows])
    if len(t) < 5:
        return None
    dPdt = (P[2:] - P[:-2]) / (t[2:] - t[:-2])          # centred, interior only
    Fi, Qi, ti = F[1:-1], Q[1:-1], t[1:-1]
    pred = Qi + Fi                                       # eq. (4) RHS
    resid = dPdt - pred
    msk = ti >= f_discard * ti[-1]
    scale = float(np.mean(np.abs(np.stack([dPdt[msk], Qi[msk], Fi[msk]]))) + 1e-300)
    # F_flux, eq. (5): the same force, estimated from the flux side
    F_flux = dPdt - Qi
    return {
        "n_samples": int(msk.sum()),
        "dPdt_mean": float(dPdt[msk].mean()),
        "flux_mean": float(Qi[msk].mean()),
        "F_R_mean": float(Fi[msk].mean()),
        "F_R_std": float(Fi[msk].std()),
        "F_flux_mean": float(F_flux[msk].mean()),
        "F_flux_std": float(F_flux[msk].std()),
        "resid_mean": float(resid[msk].mean()),
        "resid_rms": float(np.sqrt(np.mean(resid[msk] ** 2))),
        "resid_rel": float(np.sqrt(np.mean(resid[msk] ** 2)) / scale),
        "term_scale": scale,
        "estimator_ratio": (float(F_flux[msk].mean() / Fi[msk].mean())
                            if abs(Fi[msk].mean()) > 1e-30 else None),
        "estimator_rel_gap": (float(abs(F_flux[msk].mean() - Fi[msk].mean()) / abs(Fi[msk].mean()))
                              if abs(Fi[msk].mean()) > 1e-30 else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--sep", type=float, default=3.0)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--T", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--sample-dt", type=float, default=0.05)
    ap.add_argument("--f-discard", type=float, default=0.4)
    ap.add_argument("--arms", default="off,well,hill")
    ap.add_argument("--reanalyze", action="store_true",
                    help="skip simulation; re-derive summary.json from the CSVs already in --out "
                         "(use after a gate/analysis fix, so a completed run need not be repeated)")
    ap.add_argument("--resid-tol", type=float, default=0.05,
                    help="G1/G2 pass if relative ledger residual is below this")
    ap.add_argument("--agree-tol", type=float, default=0.05,
                    help="G3 pass if |(F_flux-F_flux_off)-F_R|/|F_R| is below this")
    for k, v in dict(c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1, w=0.964, alpha_T=0.35,
                     omega_T=1.25, omega_G=0.85, gamma_T=0.08, gamma_G=0.06, kappa_TG=0.55,
                     epsilon_G=0.06, cT=0.7, cG=0.55, absorb_width=1.6, absorb_strength=0.02,
                     core_radius=2.0).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "sweep_runs" / (
        "TG_B2_MIDPLANE_FLUX_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    pf = b2.preflight()
    t0 = time.time()
    print("=== TG-B2 midplane stress-flux (P2 independent estimator) | "
          f"{pf['backend']} {pf['devices']} x64={pf['x64']} | out={out} ===", flush=True)
    write_json(out / "config.json", {
        "args": vars(args), "preflight": pf,
        "observable": "F_flux = dP_x/dt - Int_plane S dA, S = c^2 A|grad phi|^2 - 2c^2 A|d_x phi|^2 "
                      "- |pi|^2 + m^2 rho - U(rho); compared against F_R = -c^2 Int_{x>0}(d_x A)|grad phi|^2 dV",
        "gates": {"G1": "off arm (A=1, F_R=0): ledger closes on flux alone",
                  "G2": "live arms: |dP/dt-(flux+F_R)| small vs term scale",
                  "G3": "<F_flux> agrees with <F_R>",
                  "G4": "<F_flux> reverses sign between well and hill"},
        "boundary": "mirror-only; frozen TG-B1S dynamics untouched (observation only); "
                    "no gravity/UFF/IRER claim"})
    if not pf["gpu_ok"] and not args.reanalyze:
        write_json(out / "RUN_FAILED.json", {"status": "GPU_PREFLIGHT_FAILED", **pf})
        print("GPU_PREFLIGHT_FAILED")
        return

    cfg = vars(args).copy()
    phi, prof = ((None, {"residual": float("nan")}) if args.reanalyze
                 else b1s.solve_qball(cfg))
    if args.reanalyze:
        op = refs = g = cfgv = refsv = None
    else:
        op = build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
        refs = b1s.ref_source_norms(phi, cfg, op)
        g = b1s.make_grid(op)
        cfgv = b1s.cfg_array(cfg)
        refsv = b1s.refs_array(refs)
    chunk_steps = max(1, int(round(args.sample_dt / args.dt)))
    nchunk = max(8, int(round(args.T / args.dt / chunk_steps)))
    print(f"[setup] resid={prof['residual']:.1e}; sep={args.sep}; T={args.T} "
          f"({nchunk} samples/arm); discard first {args.f_discard:.0%}", flush=True)

    FULL = [1.0, 1.0, 1.0, 1.0]
    OFF = [1.0, 1.0, 1.0, 0.0]
    ARM = {"off": (OFF, -1.0, 0.0), "well": (FULL, +1.0, 1.0), "hill": (FULL, -1.0, 1.0)}
    want = [a.strip() for a in args.arms.split(",") if a.strip()]

    psi, pi, sc = (b2.place_two(phi, cfg, args.sep, 0.0) if not args.reanalyze else (None, None, None))
    results = {}
    series_by_arm = {}
    for name in want:
        flags, a_sign, fb = ARM[name]
        if args.reanalyze:
            path = out / f"stress_{name}.csv"
            with open(path, newline="") as fh:
                rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
            print(f"[arm {name}] re-analysed {len(rows)} samples from {path.name}", flush=True)
        else:
            print(f"[arm {name}] ...", flush=True)
            rows = run_arm(out / f"stress_{name}.csv", psi, pi, cfgv, refsv, g,
                           flags, a_sign, fb, nchunk, chunk_steps, args.dt)
        res = {v: ledger(rows, args.f_discard, v) for v in ("flux_plane", "flux_div")}
        results[name] = res
        series_by_arm[name] = series(rows, args.f_discard, "flux_plane")
        lp = res["flux_plane"]
        if lp:
            gap = lp["estimator_rel_gap"]
            gaps = "n/a (F_R==0)" if gap is None else f"{gap:.3f}"
            print(f"[arm {name}] <F_R>={lp['F_R_mean']:+.4e} <F_flux>={lp['F_flux_mean']:+.4e} "
                  f"gap={gaps} | ledger resid_rel={lp['resid_rel']:.3e} "
                  f"(div variant {res['flux_div']['resid_rel']:.3e})", flush=True)

    # ---------------- gates ----------------
    def rel(name, variant="flux_plane"):
        r = results.get(name, {}).get(variant)
        return r["resid_rel"] if r else float("nan")

    g1 = rel("off") < args.resid_tol if "off" in results else None
    g2 = all(rel(n) < args.resid_tol for n in results if n != "off") or None
    # G3/G4 must compare LIKE WITH LIKE. F_R is a DIFFERENTIAL quantity: it is identically zero
    # when A=1, so it measures only the A-mediated force. F_flux is a TOTAL: it also contains any
    # bare KG two-body force, which F_R cannot see. The correct comparison is therefore
    #     F_flux(arm) - F_flux(off)   against   F_R(arm)
    # -- the same "isolate the loop force as F_full - F_off" convention the two-node harness uses.
    # Differencing is done SAMPLE BY SAMPLE: the arms share initial conditions, so the large
    # dP/dt and flux terms are common mode and cancel, which is what makes the differential
    # resolvable at all (the absolute F_flux is not).
    live = [n for n in ("well", "hill") if n in results]
    diff_stats = {}
    if "off" in results and series_by_arm.get("off") is not None:
        so = series_by_arm["off"]
        for n in live:
            sn = series_by_arm[n]
            k = min(len(so["F_flux"]), len(sn["F_flux"]))
            d = stat(sn["F_flux"][:k] - so["F_flux"][:k])
            fr = stat(sn["F_R"][:k])
            d["F_R_mean"] = fr["mean"]
            d["rel_gap"] = abs(d["mean"] - fr["mean"]) / (abs(fr["mean"]) + 1e-300)
            d["sem_frac"] = d["sem"] / (abs(d["mean"]) + 1e-300)
            d["gap_within_error"] = bool(d["rel_gap"] <= max(d["sem_frac"], 1e-12))
            d["corr_per_sample"] = float(np.corrcoef(sn["F_flux"][:k] - so["F_flux"][:k],
                                                     sn["F_R"][:k])[0, 1])
            diff_stats[n] = d
            results[n]["differential"] = d

    g3 = (all(diff_stats[n]["rel_gap"] < args.agree_tol for n in live)
          if diff_stats and len(diff_stats) == len(live) else None)
    g4 = (bool(diff_stats["well"]["mean"] * diff_stats["hill"]["mean"] < 0)
          if {"well", "hill"} <= set(diff_stats) else None)

    # Is the BARE (A-independent) two-body force resolved at this run length? Report honestly.
    bare = None
    if "off" in results and series_by_arm.get("off") is not None:
        b = stat(series_by_arm["off"]["F_flux"])
        b["resolved_at_2sigma"] = bool(b["sigma"] is not None and b["sigma"] >= 2.0)
        b["note"] = ("absolute F_flux is a ~1700:1 cancellation between dP/dt and the plane flux; "
                     "its per-sample scatter is orders of magnitude above its mean, so a small "
                     "|sigma| here means NOT MEASURED, not 'measured to be zero'")
        bare = {"bare_kg_two_body_force": b}

    if g1 is False:
        verdict = "TG_B2_MIDPLANE_FLUX_STRESS_TENSOR_INVALID__OFF_ARM_LEDGER_OPEN"
    elif g1 and g2 and g3 and g4:
        verdict = "TG_B2_MIDPLANE_FLUX_CONFIRMS_BODY_FORCE"
    elif g1 and g2 and g3 is False:
        verdict = "TG_B2_MIDPLANE_FLUX_DISAGREES_WITH_BODY_FORCE__ESTIMATORS_MEASURE_DIFFERENT_QUANTITIES"
    elif g1 and g2 is False:
        verdict = "TG_B2_MIDPLANE_FLUX_LEDGER_OPEN_ON_LIVE_ARMS"
    else:
        verdict = "TG_B2_MIDPLANE_FLUX_INCONCLUSIVE"

    summary = {
        "verdict": verdict,
        "gates": {"G1_off_ledger_closes": g1, "G2_live_ledger_closes": g2,
                  "G3_estimators_agree": g3, "G4_flux_sign_reverses": g4},
        "tolerances": {"resid_rel": args.resid_tol, "estimator_rel_gap": args.agree_tol},
        "arms": results,
        "force_decomposition": bare,
        "differential": diff_stats,
        "sep": args.sep, "N": args.N, "L": args.L, "T": args.T, "dt": args.dt,
        "elapsed_hours": (time.time() - t0) / 3600.0,
        "config": {k: v for k, v in vars(args).items()},
        "note": "F_flux is an INDEPENDENT estimator: it samples a 2-D plane and carries |pi|^2, "
                "m^2 rho and U(rho), none of which appear in the body force F_R. Agreement is a "
                "genuine cross-check; disagreement means the two measure different quantities.",
        "boundary": "mirror-only; frozen TG-B1S dynamics untouched; no gravity/UFF/IRER claim",
    }
    write_json(out / "summary.json", summary)
    write_json(out / "RUN_COMPLETE.json", {"outdir": str(out), "status": verdict})
    print(f"\n{verdict}\n  gates: {summary['gates']}\n  elapsed {summary['elapsed_hours']:.3f} h",
          flush=True)


if __name__ == "__main__":
    main()
