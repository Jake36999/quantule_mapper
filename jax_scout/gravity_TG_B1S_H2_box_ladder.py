"""H2 -- is the long-time drift numerical, physical, or an artefact of the box?

WHY THIS EXISTS. `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED` has blocked promotion of the TG
sector since July. The plan framed it as a binary -- numerical (RK4 is not structure-preserving)
versus physical phase-detuning -- and proposed dt-halving and grid-doubling to decide it.

THAT PART IS ALREADY ANSWERED, by the existing D4 rows, and the answer is NEITHER:

    dt halved            delta_omega_inf changes by 4 parts in 1e12
    grid N 48 -> 56      changes by 4 parts in 1e9
    absorber x1.25       changes by 7 parts in 1e7
    box L 10 -> 12       changes by a FACTOR OF 2.54

RK4 is fourth order, so a time-integration artefact would fall ~16x when dt is halved. It moves in
the twelfth digit. The drift is numerically converged in both dt and dx. The one knob that moves it
is the size of the periodic box -- and `boundary_flux_proxy_max` falls 3.6x in the larger box while
the drift falls 2.54x, so the drift tracks the boundary rather than the dynamics.

Two points in L are not a scaling law. This harness measures the ladder.

WHAT IT DOES NOT DO. It does not touch the frozen dynamics. It imports the D4/D5 closure module and
calls exactly the functions that produced the rows above -- solve_phi_refs,
dec.run_pair_decomposition, dec.asymptotic_frequency_row, boundary_flux -- so a difference between
this ladder and the D4 rows can only come from the configuration, never from the method.
Mirror-only, observation-only. No gravity / UFF / IRER claim.

READING THE RESULT

    |delta_omega_inf| falls toward zero as L grows  -> finite-box artefact; the "drift" is the
                                                      boundary, and the blocker dissolves.
    it flattens to a non-zero constant              -> a real property of the model; the L=10
                                                      value was contaminated but the effect is
                                                      genuine.
    it grows, or moves without pattern              -> neither; report that and stop.

Usage (WSL jax_irer venv):
    python jax_scout/gravity_TG_B1S_H2_box_ladder.py --out sweep_runs/TG_H2_BOX_LADDER \
        --boxes 10:48,12:48,14:48,16:48 --periods 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import jax_scout.gravity_TG_B1S_D4_D5_closure_gpu as clo
import jax_scout.gravity_TG_B1S_drift_decomposition_gpu as dec
from jax_scout.provenance import flat_stamp

PARAMS = ("c", "m", "a", "s", "f", "w", "dt", "T", "alpha_T", "omega_T", "omega_G",
          "gamma_T", "gamma_G", "kappa_TG", "epsilon_G", "cT", "cG",
          "absorb_width", "absorb_strength", "core_radius")


def d4_defaults():
    """Read the physics defaults out of the D4 harness's own parser, by parsing its source.

    Copying them here by hand is how a ladder stops being comparable to the rows it is meant to
    extend: the copy is correct on the day it is written and silently wrong after the next edit.
    The whole value of this harness is that the ONLY difference from a D4 row is L and N, so the
    defaults are read from the one place that defines them, and a missing one is a hard failure
    rather than a plausible-looking substitute.
    """
    import ast
    src = Path(clo.__file__).read_text(encoding="utf-8")
    found = {}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        flag = getattr(node.args[0], "value", None)
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        name = flag[2:].replace("-", "_")
        if name not in PARAMS:
            continue
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            try:
                # literal_eval, not isinstance(Constant): a negative default such as s=-0.5
                # is a UnaryOp, and treating it as unreadable would silently drop two parameters.
                found[name] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                pass
    missing = [p for p in PARAMS if p not in found]
    if missing:
        raise SystemExit("cannot read D4 defaults for %s from %s -- refusing to guess, because a "
                         "guessed parameter makes this ladder incomparable to the D4 rows"
                         % (", ".join(missing), clo.__file__))
    return found


def parse_boxes(spec):
    """'10:48,12:58' -> [(10.0, 48), (12.0, 58)].

    L and N are named together on purpose: the whole question is which of the two the drift
    responds to, so neither may be left implicit.
    """
    out = []
    for part in spec.split(","):
        L, N = part.split(":")
        out.append((float(L), int(N)))
    return out


def main():
    ap = argparse.ArgumentParser(description="H2 box ladder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxes", default="10:48,12:48,14:48,16:48",
                    help="comma-separated L:N pairs")
    ap.add_argument("--periods", type=float, default=50.0)
    ap.add_argument("--sample-dt", type=float, default=1.0)   # the D4 row value
    # every physics parameter keeps its D4 default, so the ladder is comparable by construction
    defaults = d4_defaults()
    for name in PARAMS:
        ap.add_argument("--" + name.replace("_", "-"), type=float, default=defaults[name])
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    clo.install_basin_initial_state_patch()

    base_ns = argparse.Namespace(N=48, L=10.0, sample_dt=args.sample_dt,
                                 **{k: getattr(args, k) for k in PARAMS})
    cfg0 = clo.default_config(base_ns)

    boxes = parse_boxes(args.boxes)
    (outdir / "config.json").write_text(json.dumps(
        {"boxes": boxes, "periods": args.periods, "base_config": cfg0,
         "provenance": flat_stamp(),
         "note": ("H2 box ladder -- observation only; frozen dynamics imported from the "
                  "D4 closure module")},
        indent=2, default=str), encoding="utf-8")

    rows = []
    for L, N in boxes:
        case = "H2_L%.4g_N%d" % (L, N)
        cfg = dict(cfg0)
        cfg["L"], cfg["N"] = float(L), int(N)
        t0 = time.time()
        print("[%s] L=%.4g N=%d dx=%.4f ..." % (case, L, N, L / N), flush=True)
        phi, _prof, refs, base = clo.solve_phi_refs(cfg)
        summary, pair_rows, _f, _o = dec.run_pair_decomposition(
            case, args.periods, 1.0, cfg, phi, refs, args.sample_dt)
        asym = dec.asymptotic_frequency_row(case, pair_rows)
        bnd = clo.boundary_flux(case, pair_rows)
        row = {
            "case": case, "L": L, "N": N, "dx": L / N,
            "delta_omega_infty": asym.get("delta_omega_infty"),
            "early_late_relative_difference": asym.get("early_late_relative_difference"),
            "fit_rmse": asym.get("fit_rmse"),
            "constant_slope_supported": asym.get("constant_slope_supported"),
            "boundary_flux_proxy_max": bnd.get("shell_flux_proxy_max"),
            "ledger_residual_abs": summary.get("ledger_residual_abs"),
            "profile_overlap": summary.get("full_profile_overlap"),
            "baseline_status": base.get("status"),
            "wall_s": round(time.time() - t0, 1),
        }
        rows.append(row)
        print("   d_omega_inf = %.6e   boundary_flux = %.4e   (%.0f s)"
              % (row["delta_omega_infty"], row["boundary_flux_proxy_max"], row["wall_s"]),
              flush=True)
        # written after every row, so a killed run keeps everything it has already measured
        clo.write_csv(outdir / "box_ladder.csv", rows)

    # The reading is stated by the harness, so it cannot be quietly reinterpreted later.
    good = [r for r in rows if r["delta_omega_infty"] is not None]
    verdict, extra = "H2_INSUFFICIENT_POINTS", {}
    if len(good) >= 3:
        Ls = np.array([r["L"] for r in good], dtype=float)
        ds = np.abs(np.array([r["delta_omega_infty"] for r in good], dtype=float))
        # a straight power law in L is the falsifiable form: |delta_omega| ~ L**p
        p = float(np.polyfit(np.log(Ls), np.log(ds), 1)[0])
        shrink = float(ds[-1] / ds[0])

        # "It shrank a lot" does NOT mean "it goes to zero". The first version of this test used
        # only shrink and a log-log slope, and called a 3.4x decrease a finite-box effect when the
        # sequence was in fact levelling off at ~30% of its starting value. Decaying TO ZERO and
        # decaying TO A NON-ZERO ASYMPTOTE look identical to a power-law slope and are opposite
        # physics: the first makes the drift an artefact, the second makes it real.
        #
        # So fit both and let the residuals decide.
        # Fit |d| = a + b*exp(-c L) DETERMINISTICALLY. A seeded optimiser is the wrong tool here:
        # three parameters against four points, and curve_fit with a plausible-looking p0 landed in
        # a local minimum giving c = 37.6 and a 58% residual, against 0.9% for the true optimum.
        # For FIXED c the model is linear in (a, b), so scan c and solve (a, b) exactly. One
        # dimension, no starting guess, no local minima.
        asym, asym_rel_rms = None, None
        if len(good) >= 4:
            best = None
            for c_try in np.linspace(0.02, 5.0, 20000):
                X = np.column_stack([np.ones_like(Ls), np.exp(-c_try * Ls)])
                coef, *_ = np.linalg.lstsq(X, ds, rcond=None)
                rr = float(np.sqrt(np.mean((ds - X @ coef) ** 2)) / np.mean(ds))
                if best is None or rr < best[0]:
                    best = (rr, c_try, float(coef[0]), float(coef[1]))
            asym_rel_rms, c_fit, asym, b_fit = best
            extra_fit = {"asymptote_decay_rate_c": c_fit, "asymptote_b": b_fit}
        pure = ds - np.exp(np.polyval(np.polyfit(np.log(Ls), np.log(ds), 1), np.log(Ls)))
        pure_rel_rms = float(np.sqrt(np.mean(pure ** 2)) / np.mean(ds))

        if asym is not None and asym_rel_rms < 0.5 * pure_rel_rms and asym > 0.05 * ds[0]:
            verdict = "H2_DRIFT_CONVERGES_TO_A_NONZERO_ASYMPTOTE__PARTLY_BOX_PARTLY_REAL"
        elif shrink < 0.5 and p < -0.5:
            verdict = "H2_DRIFT_IS_FINITE_BOX_EFFECT"
        elif 0.8 < shrink < 1.25:
            verdict = "H2_DRIFT_BOX_INDEPENDENT__NOT_A_BOUNDARY_EFFECT"
        else:
            verdict = "H2_DRIFT_BOX_DEPENDENT_NO_CLEAN_POWER_LAW"
        extra = {"power_law_exponent_in_L": p, "shrink_factor_largest_over_smallest": shrink,
                 "asymptote_L_to_inf": asym, "asymptote_fit_rel_rms": asym_rel_rms,
                 "decay_to_zero_fit_rel_rms": pure_rel_rms,
                 "asymptote_fraction_of_smallest_box": (asym / ds[0]) if asym else None}
        extra.update(locals().get("extra_fit", {}))
    clo.write_json(outdir / "summary.json",
                   dict(verdict=verdict, rows=rows, provenance=flat_stamp(), **extra))
    clo.write_json(outdir / "RUN_COMPLETE.json", {"verdict": verdict, "n_rows": len(rows)})
    print("")
    print("VERDICT: %s" % verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
