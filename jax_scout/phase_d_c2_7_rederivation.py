"""Phase D / C2.7 — re-derivation campaign on the FIXED substrate (true geometry-off, post-C2.6).
Stages (checkpointed, one bounded run):
  R0  CFL/dt re-baseline: the old stability windows were measured at D_eff~D/151; with full dispersion restored,
      re-establish which (N, dt) pairs are stable for conservative geometry-off evolution (feb D=2.73 + D=1.0).
  R2  feb/a* TRUE pure-NLS mini-scout: the C2.1 "native soliton" map was measured on the bugged substrate
      (and its geometry-ON variant separately); does ANYTHING localize for feb coefficients on the true flat
      substrate? (Prediction: mostly DISPERSE — box/binding argument: g_max=0.23 << mu_box ~ 16D/L^2 = 0.44.)
  R3  Moving-family confirmation at N=96: Petviashvili (a=0.8, s=-0.2, f=0, D=1.0, mu=0.2), hold, winding-boost
      ladder n=1,2 -> v vs 2Dk, mass. CLEAN_TRANSPORT_N96 if v_frac > 0.9 and mass_ret > 0.98.
(R1 = family_scout Tier-1/1b reruns are chained separately — the scout now uses the fixed ops automatically.)

  wsl:  python jax_scout/phase_d_c2_7_rederivation.py [--out DIR]
"""
import os, sys, json, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css, physics, transfer_diag as td
from jax_scout.phase_d_c1_transport import _evolve_chunk
from jax_scout.phase_d_c2_soliton_scout import gaussian_ic, occ
from jax_scout.phase_d_c2_2_loss_source import _axis_grid, _circ_angle, _velocity
from jax_scout.phase_d_c2_3_exact_soliton import petviashvili, momentum_x

L = css.L_


def _ops_flat(N, dt, D=None, a=None, s=None, f=None):
    p = {**css.FEB, "param_a": a if a is not None else float(css.FEB["param_a"]) * 1.15,
         "kinetic_mode": "conservative", "param_a_coupling": 0.0, "param_geom_off": True}
    if D is not None: p["param_D"] = D
    if s is not None: p["param_s"] = s
    if f is not None: p["param_f"] = f
    return physics.build_operators(N, L, dt, p)


def _evolve(psi, ops, steps, dt_chunk):
    M0 = float(np.sum(np.abs(psi) ** 2))
    pk = physics.initial_psi_k(jnp.asarray(psi), ops); cur = psi
    for c in range(steps // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return None, np.nan
    return cur, float(np.sum(np.abs(cur) ** 2)) / M0


def r0_cfl(out, quick=False):
    print(f"=== R0: CFL/dt re-baseline (true substrate, full dispersion){' [QUICK]' if quick else ''} ===", flush=True)
    rows = []
    nd_pairs = [(48, 1e-3), (96, 5e-4)] if quick else [(48, 1e-3), (48, 5e-4), (96, 1e-3), (96, 5e-4), (96, 2.5e-4)]
    for (N, dt) in nd_pairs:
        for (Dv, tag) in [(2.7329, "feb-D"), (1.0, "family-D")]:
            ops = _ops_flat(N, dt, D=Dv)
            psi = gaussian_ic(1.0, 0.15, N)
            steps = int(round(1.0 / dt))
            cur, mret = _evolve(psi, ops, steps, min(1000, steps))
            ok = cur is not None
            stiff = Dv * (np.pi * N / L) ** 2 * dt
            rows.append({"N": N, "dt": dt, "D": Dv, "stable": ok, "mass_ret_T1": mret,
                         "linear_stiffness_rad_per_step": round(stiff, 2)})
            print(f"  N={N} dt={dt} D={Dv} ({tag}): {'STABLE' if ok else 'COLLAPSE'} "
                  f"mass_ret={mret if not np.isnan(mret) else 'nan'} |Dk2dt|max={stiff:.2f}", flush=True)
    json.dump(rows, open(os.path.join(out, "r0_cfl.json"), "w"), indent=2, default=float)
    return rows


def r2_feb_scout(out, dt=5e-4, quick=False):
    print(f"=== R2: feb/a* TRUE pure-NLS mini-scout (N=48){' [QUICK 2-cell]' if quick else ''} ===", flush=True)
    N = 48
    ops = _ops_flat(N, dt)                                   # feb coefficients, D=2.7329, a*=x1.15
    As = (1.0, 2.0) if quick else (0.5, 1.0, 2.0)
    sigs = (0.15,) if quick else (0.083, 0.15)
    T_phys = 2.0 if quick else 4.0
    rows = []
    for A in As:
        for sig in sigs:
            psi = gaussian_ic(A, sig, N)
            occ0 = occ(np.abs(psi) ** 2)
            cur, mret = _evolve(psi, ops, int(round(T_phys / dt)), 1000)
            if cur is None:
                k = "COLLAPSE"; occr = np.nan; ampf = np.nan
            else:
                rho = np.abs(cur) ** 2
                occr = occ(rho) / (occ0 + 1e-30); ampf = float(np.abs(cur).max()) / A
                k = ("LOCALIZED" if (occr < 2.0 and mret > 0.6 and 0.4 < ampf) else
                     "DISPERSE" if occr > 3.0 or ampf < 0.3 else "MARGINAL")
            rows.append({"A": A, "sigma": sig, "klass": k, "mass_ret": mret, "occ_ratio": occr, "amp_ret": ampf})
            print(f"  A={A} sig={sig}: {k} mass_ret={mret} occ_ratio={occr} amp_ret={ampf}", flush=True)
    loc = [r for r in rows if r["klass"] == "LOCALIZED"]
    verdict = "FEB_TRUE_NLS_HAS_LOCALIZED" if loc else "FEB_TRUE_NLS_NO_LOCALIZED"
    print(f"  => {verdict} ({len(loc)}/6)", flush=True)
    json.dump({"verdict": verdict, "rows": rows}, open(os.path.join(out, "r2_feb.json"), "w"), indent=2, default=float)
    return verdict


def r3_family_n96(out, dt=2.5e-4):
    print("=== R3: moving-family confirmation at N=96 (a=0.8 s=-0.2 f=0 D=1.0 mu=0.2) ===", flush=True)
    N = 96; D = 1.0
    ops = _ops_flat(N, dt, D=D, a=0.8, s=-0.2, f=0.0)
    Xax = _axis_grid(N)
    psi_c, prof = petviashvili(1.0, 0.08, ops, N, 0.2)
    if psi_c is None or prof["residual"] > 1e-6:
        print(f"  petviashvili N=96 FAILED: {prof}", flush=True)
        json.dump({"verdict": "R3_PETVIASHVILI_FAILED", "prof": prof},
                  open(os.path.join(out, "r3_n96.json"), "w"), indent=2, default=float)
        return "R3_PETVIASHVILI_FAILED"
    print(f"  petviashvili: residual={prof['residual']:.2e} amp={prof['amp']:.3f} occ={prof['occ']:.4f}", flush=True)
    np.save(os.path.join(out, "r3_soliton_n96.npy"), psi_c)
    cur, mret = _evolve(psi_c, ops, int(round(3.0 / dt)), 2000)
    print(f"  hold T=3: mass_ret={mret:.4f}", flush=True)
    boosts = []
    for n in (1, 2):
        k = 2 * np.pi * n / L
        psi = (psi_c * np.exp(1j * k * Xax)).astype(np.complex128)
        M0 = float(np.sum(np.abs(psi) ** 2))
        pk = physics.initial_psi_k(jnp.asarray(psi), ops)
        ang, tt = [], []
        cur = psi
        steps = int(round(3.0 / dt)); chunk = 1000
        for c in range(steps // chunk):
            pk = _evolve_chunk(pk, ops, chunk); cur = np.asarray(jnp.fft.ifftn(pk))
            if not np.isfinite(cur).all():
                break
            ang.append(_circ_angle(np.abs(cur) ** 2, Xax)); tt.append((c + 1) * chunk * dt)
        v, r2, pos = _velocity(ang, tt)
        vf = v / (2 * D * k)
        m = float(np.sum(np.abs(cur) ** 2)) / M0
        boosts.append({"n": n, "k": k, "v": v, "v_pred": 2 * D * k, "v_frac": vf, "r2": r2, "mass_ret": m})
        print(f"  boost n={n}: v={v:+.4f} (pred {2*D*k:+.4f}, frac={vf:+.4f}) r2={r2:.3f} mass={m:.4f}", flush=True)
    ok = all(b["v_frac"] > 0.9 and b["mass_ret"] > 0.98 for b in boosts)
    verdict = "CLEAN_TRANSPORT_N96_CONFIRMED" if ok else "R3_TRANSPORT_DEGRADED_AT_N96"
    print(f"  => {verdict}", flush=True)
    json.dump({"verdict": verdict, "profile": prof, "hold_mass_ret": mret, "boosts": boosts},
              open(os.path.join(out, "r3_n96.json"), "w"), indent=2, default=float)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="bounded smoke: R0 CFL + R2 feb pure-NLS scout only; SKIP the long N=96 R3 hold/boost")
    a_ = ap.parse_args()
    tag = "PHASE_D_C2_7_REDERIVE" + ("_QUICK" if a_.quick else "")
    out = a_.out or os.path.join(ROOT, "sweep_runs", f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    print(f"=== C2.7 RE-DERIVATION CAMPAIGN (fixed substrate){' [QUICK]' if a_.quick else ''} | out={out} ===", flush=True)
    t0 = time.time()
    r0 = r0_cfl(out, quick=a_.quick)
    v2 = r2_feb_scout(out, quick=a_.quick)
    if a_.quick:
        print(f"\n=== C2.7 QUICK DONE | R2={v2} (R3 N=96 hold skipped: see docs/PHASE_D_C2_7_REDERIVATION_RESULTS.md "
              f"for the finalized CLEAN_TRANSPORT_N96_CONFIRMED) | {round((time.time()-t0)/60,1)}m ===", flush=True)
        return
    v3 = r3_family_n96(out)
    print(f"\n=== C2.7 DONE | R2={v2} R3={v3} | {round((time.time()-t0)/60,1)}m ===", flush=True)


if __name__ == "__main__":
    main()
