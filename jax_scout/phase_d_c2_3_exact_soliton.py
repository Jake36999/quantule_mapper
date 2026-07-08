"""Phase D / C2.3 — EXACT NLS soliton (imaginary-time ground state) boost test + paired N-dt convergence.
C2.2 showed the kick-associated loss is dominantly dt-numerical (converges 0.205->0.132->0.090, extrap ~0.03) and
NOT geometry. Two loose ends: (1) the residual could be the boosted object being a relaxed Gaussian rather than the
true NLS eigenstate; (2) N-convergence was blocked by CFL. This harness:
  - relaxes the IC to the EXACT ground state by imaginary-time propagation (split-step, mass-renormalized,
    geometry-off pure NLS: i psi_t = -D lap psi - (a rho + s rho^2 + f rho^3) psi, focusing cubic / defocusing septic);
  - verifies stationarity (residual ||H psi - mu psi|| / ||mu psi||);
  - boosts the exact state (kick ladder) and tracks mass, centroid velocity AND total momentum P(t)=Im sum psi* grad psi.
MOMENTUM DIAGNOSTIC: Galilean invariance predicts v = 2*D*k exactly for the boosted eigenstate; C2.1/C2.2 measured
v/k ~ 0.038 = 0.7% of 2D=5.47 — either momentum is destroyed numerically (a momentum sink would also source the k^2
losses) or the Gaussian-relaxed compound soaked the kick into its halo. P(t) settles which.
Mirror only; Phase C default untouched; conservative branch via build_operators only; no clipping; no matter claims.

  wsl:  python jax_scout/phase_d_c2_3_exact_soliton.py --N 96 --dt 0.00025 --kicks 0,1,2 --Tphys 6.0 [--out DIR]
"""
import os, sys, json, time, argparse
from functools import partial
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
from jax_scout.phase_d_c2_2_loss_source import _ops, _axis_grid, _circ_angle, _velocity, _f

L = css.L_


# ---------------- imaginary-time ground state (pure NLS, geometry off) ----------------
@partial(jax.jit, static_argnames=("n_iters",))
def _relax_chunk(psi_k, ops, M0k, dtau, n_iters):
    """Split-step imaginary time: exact linear decay exp(-dtau*D*k^2), explicit nonlinear gain, renormalize mass."""
    lin = jnp.exp(dtau * ops.D_diff * ops.minus_k_sq)          # minus_k_sq = -k^2  ->  exp(-dtau D k^2)

    def body(pk, _):
        pk = pk * lin
        psi = jnp.fft.ifftn(pk)
        rho = jnp.real(psi) ** 2 + jnp.imag(psi) ** 2
        psi = psi * jnp.exp(dtau * (ops.a * rho + ops.s * rho ** 2 + ops.f * rho ** 3))
        pk = jnp.fft.fftn(psi) * ops.dealias_mask
        pk = pk * jnp.sqrt(M0k / jnp.sum(jnp.abs(pk) ** 2))    # Parseval: mass_k = Ncells * mass_x
        return pk, None

    pk, _ = jax.lax.scan(body, psi_k, None, length=n_iters)
    return pk


@partial(jax.jit, static_argnames=("n_iters",))
def _petviashvili_chunk(pk, ops, mu, gamma, n_iters):
    """Petviashvili iteration for the stationary state mu*psi = D lap psi + g(rho) psi at FIXED mu.
    (Imaginary time is the wrong tool here: it flows to the max-mu state = the UNIFORM condensate — the
    localized soliton is metastable, not the global ground state. Petviashvili targets the solitary wave.)"""
    denom = mu - ops.D_diff * ops.minus_k_sq               # = mu + D k^2 > 0

    def body(pk, _):
        psi = jnp.fft.ifftn(pk)
        rho = jnp.real(psi) ** 2 + jnp.imag(psi) ** 2
        Nk = jnp.fft.fftn((ops.a * rho + ops.s * rho ** 2 + ops.f * rho ** 3) * psi) * ops.dealias_mask
        num = jnp.sum(jnp.real(jnp.conj(pk) * (denom * pk)))
        den = jnp.sum(jnp.real(jnp.conj(pk) * Nk))
        S = num / jnp.where(jnp.abs(den) < 1e-300, 1e-300, den)
        pk_new = (jnp.abs(S) ** gamma) * Nk / denom
        return pk_new, jnp.abs(S)

    pk, Ss = jax.lax.scan(body, pk, None, length=n_iters)
    return pk, Ss[-1]


def petviashvili(A, sig, ops, N, mu, max_iters=600, chunk=50, gamma=1.5, tol=1e-10):
    """Run Petviashvili at fixed mu from a Gaussian seed; returns (psi, profile) or (None, reason).
    NOTE physics constraint: g(rho)=a rho+s rho^2+f rho^3 saturates at g_max~0.23 (rho~0.6) for the feb/a*
    coefficients, so stationary branches only exist for mu < ~0.23; the uniform condensate branch is a strong
    attractor of the iteration for wide seeds (use narrow seeds to reach the localized branch, if it exists)."""
    psi0 = gaussian_ic(A, sig, N)
    pk = jnp.fft.fftn(jnp.asarray(psi0)) * ops.dealias_mask
    prev = np.asarray(jnp.fft.ifftn(pk))
    for it in range(0, max_iters, chunk):
        pk, S = _petviashvili_chunk(pk, ops, mu, gamma, chunk)
        cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return None, {"mu": mu, "gamma": gamma, "fail": "NONFINITE"}
        if float(np.max(np.abs(cur))) < 1e-6:
            return None, {"mu": mu, "gamma": gamma, "fail": "COLLAPSED_TO_ZERO"}
        drel = float(np.linalg.norm(cur - prev) / (np.linalg.norm(cur) + 1e-300))
        prev = cur
        if drel < tol:
            break
    mu_chk, res = stationarity(cur, ops)
    rho = np.abs(cur) ** 2
    prof = {"mu": mu, "gamma": gamma, "mu_check": mu_chk, "residual": res, "S_final": float(S),
            "drel": drel, "mass": float(rho.sum()), "amp": float(np.max(np.abs(cur))), "occ": occ(rho),
            "iters": it + chunk}
    return cur, prof


def stationarity(psi, ops):
    """Residual of H psi = mu psi with H = D lap + (a rho + s rho^2 + f rho^3) (imag-time generator)."""
    pk = np.fft.fftn(psi)
    lap = np.fft.ifftn(np.asarray(ops.minus_k_sq) * pk)
    rho = np.abs(psi) ** 2
    D, a, s, f = (float(np.asarray(v)) for v in (ops.D_diff, ops.a, ops.s, ops.f))
    Hpsi = D * lap + (a * rho + s * rho ** 2 + f * rho ** 3) * psi
    mu = float(np.real(np.vdot(psi, Hpsi)) / np.real(np.vdot(psi, psi)))
    res = float(np.linalg.norm(Hpsi - mu * psi) / (abs(mu) * np.linalg.norm(psi) + 1e-300))
    return mu, res


def relax(A, sig, ops, N, dtau, max_iters, chunk, tol):
    psi0 = gaussian_ic(A, sig, N)
    M0 = float(np.sum(np.abs(psi0) ** 2)); M0k = M0 * psi0.size
    pk = jnp.fft.fftn(jnp.asarray(psi0)) * ops.dealias_mask
    pk = pk * jnp.sqrt(M0k / jnp.sum(jnp.abs(pk) ** 2))
    prev = np.asarray(jnp.fft.ifftn(pk)); hist = []
    for it in range(0, max_iters, chunk):
        pk = _relax_chunk(pk, ops, M0k, dtau, chunk)
        cur = np.asarray(jnp.fft.ifftn(pk))
        drel = float(np.linalg.norm(cur - prev) / (np.linalg.norm(cur) + 1e-300))
        mu, res = stationarity(cur, ops)
        hist.append({"iter": it + chunk, "drel": drel, "mu": mu, "residual": res})
        print(f"   [relax] iter={it+chunk:5d} drel={drel:.2e} mu={mu:+.4f} residual={res:.2e}", flush=True)
        prev = cur
        if drel < tol:
            break
    rho = np.abs(prev) ** 2
    prof = {"mass": M0, "amp": float(np.max(np.abs(prev))), "occ": occ(rho),
            "mu": hist[-1]["mu"], "residual": hist[-1]["residual"], "iters": hist[-1]["iter"]}
    return prev, prof, hist


# ---------------- real-time boost with momentum tracking ----------------
def momentum_x(psi, ikx):
    """P_x = Im sum psi* dpsi/dx (grid units)."""
    dx_psi = np.fft.ifftn(np.asarray(ikx) * np.fft.fftn(psi))
    return float(np.sum(np.imag(np.conj(psi) * dx_psi)))


def _peak_x(rho, N):
    """x-index of the density peak (grid units)."""
    return int(np.unravel_index(int(np.argmax(rho)), rho.shape)[0])


def settle(A, sig, ops, N, dt, Tphys, dt_chunk):
    """Real-time settle of the Gaussian IC (the C2.1/C2.2 protocol) -> dynamically-settled quasi-soliton."""
    psi = gaussian_ic(A, sig, N)
    M0 = float(np.sum(np.abs(psi) ** 2)); occ0 = occ(np.abs(psi) ** 2)
    pk = physics.initial_psi_k(jnp.asarray(psi), ops); cur = psi
    steps = int(round(Tphys / dt))
    for c in range(steps // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return None, {"fail": "COLLAPSE"}
    rho = np.abs(cur) ** 2
    prof = {"mass_ret": float(rho.sum()) / M0, "amp": float(np.max(np.abs(cur))),
            "occ_ratio": occ(rho) / (occ0 + 1e-30), "n": len(td.detect_nodes(cur, L / N))}
    return cur, prof


def boost_exact(settled, ops, Xax, n_kick, N, dt, T_steps, dt_chunk):
    dx = L / N
    k = 2 * np.pi * n_kick / L
    psi = (settled * np.exp(1j * k * Xax)).astype(np.complex128)
    M0 = float(np.sum(np.abs(psi) ** 2)); P0 = momentum_x(psi, ops.ikx)
    pk = physics.initial_psi_k(jnp.asarray(psi), ops)
    ang, tt, ptraj, mtraj, pk_idx = [], [], [], [], []
    cur = psi
    for c in range(T_steps // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return {"n": n_kick, "k": k, "v": np.nan, "v_pred_2Dk": 2 * float(np.asarray(ops.D_diff)) * k,
                    "r2": np.nan, "mass_ret": np.nan, "P0": P0, "P_ret_traj": ptraj, "n_fin": 0, "collapsed": True}
        rho = np.abs(cur) ** 2
        ang.append(_circ_angle(rho, Xax)); tt.append((c + 1) * dt_chunk * dt)
        ptraj.append(momentum_x(cur, ops.ikx) / (P0 if abs(P0) > 1e-12 else 1.0))
        mtraj.append(float(rho.sum()) / M0)
        pk_idx.append(_peak_x(rho, N))
    v, r2, pos = _velocity(ang, tt) if len(tt) >= 3 else (np.nan, np.nan, [0, 0])
    # peak velocity: unwrap the peak index (periodic, N cells), linear fit -> physical units
    v_peak, r2_peak = np.nan, np.nan
    if len(pk_idx) >= 3:
        raw = np.asarray(pk_idx, dtype=float)
        unwrapped = raw.copy()
        for i in range(1, len(raw)):
            d = raw[i] - raw[i - 1]
            d = d - N * round(d / N)                       # minimal-image step
            unwrapped[i] = unwrapped[i - 1] + d
        tarr = np.asarray(tt)
        Amat = np.vstack([tarr, np.ones_like(tarr)]).T
        (m, b), resid, *_ = np.linalg.lstsq(Amat, unwrapped * dx, rcond=None)
        ss = np.sum((unwrapped * dx - (unwrapped * dx).mean()) ** 2)
        v_peak = float(m); r2_peak = float(1.0 - (resid[0] / ss if resid.size and ss > 0 else 0.0))
    D = float(np.asarray(ops.D_diff))
    return {"n": n_kick, "k": k, "v": float(v), "v_peak": v_peak, "r2_peak": r2_peak,
            "v_pred_2Dk": 2 * D * k,
            "v_frac_of_galilean": float(v / (2 * D * k)) if k else np.nan,
            "v_peak_frac_of_galilean": float(v_peak / (2 * D * k)) if (k and np.isfinite(v_peak)) else np.nan,
            "disp": float((pos[-1] - pos[0]) / L), "r2": float(r2),
            "mass_ret": mtraj[-1] if mtraj else np.nan, "P0": P0,
            "P_ret_traj": [round(p, 4) for p in ptraj], "mass_traj": [round(m, 4) for m in mtraj],
            "n_fin": len(td.detect_nodes(cur, dx)), "collapsed": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=96); ap.add_argument("--dt", type=float, default=0.00025)
    ap.add_argument("--A", type=float, default=1.0); ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--kicks", default="0,1,2"); ap.add_argument("--Tphys", type=float, default=6.0)
    ap.add_argument("--mus", default="0.10,0.15,0.20,0.22", help="Petviashvili mu scan (branches only exist mu<~0.23)")
    ap.add_argument("--seeds", default="1.0:0.08,1.5:0.06,0.8:0.12", help="narrow seeds 'A:sig,...' to avoid the uniform attractor")
    ap.add_argument("--settle-Tphys", type=float, default=12.0, help="fallback dynamical settle time (phys units)")
    ap.add_argument("--dtchunk", type=int, default=2000); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kicks = [int(x) for x in a.kicks.split(",")]
    T_steps = int(round(a.Tphys / a.dt))
    out = a.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C2_3_EXACT_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    ops = _ops(False, a.N, a.dt)                                # geometry OFF = pure NLS (the clean-transport question)
    Xax = _axis_grid(a.N)
    D = float(np.asarray(ops.D_diff))
    print(f"=== C2.3 EXACT-SOLITON | N={a.N} dt={a.dt} Tphys={a.Tphys} ({T_steps} steps) kicks={kicks} "
          f"seed A={a.A} sig={a.sigma} | D={D} (Galilean v=2Dk) | out={out} ===", flush=True)

    # Petviashvili (seed x mu x gamma) scan -> pick the best-converged localized stationary state
    seeds = [tuple(float(y) for y in c.split(":")) for c in a.seeds.split(",")]
    cands = []
    for (As, sigs) in seeds:
        for mu in (float(x) for x in a.mus.split(",")):
            for gamma in (1.5, 2.0):
                t0 = time.time()
                psi_c, prof_c = petviashvili(As, sigs, ops, a.N, mu, gamma=gamma)
                if psi_c is None:
                    print(f"[petviashvili] seed A={As} sig={sigs} mu={mu} g={gamma} -> FAIL {prof_c.get('fail')}", flush=True)
                    continue
                loc = prof_c["occ"] < 0.5 and 0.1 <= prof_c["amp"] <= 3.0     # localized, sane amplitude
                print(f"[petviashvili] seed A={As} sig={sigs} mu={mu} g={gamma} residual={prof_c['residual']:.2e} "
                      f"S={prof_c['S_final']:.6f} amp={prof_c['amp']:.3f} occ={prof_c['occ']:.4f} "
                      f"mass={prof_c['mass']:.1f} {'LOCALIZED' if loc else 'not-localized'} "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)
                if loc and prof_c["residual"] < 1e-6:
                    cands.append((psi_c, prof_c))
                if loc and prof_c["residual"] < 1e-9:
                    break                                          # good enough; skip the gamma retry
    exact_found = bool(cands)
    if exact_found:
        psi_exact, prof = min(cands, key=lambda c: c[1]["residual"])
        obj_src = "PETVIASHVILI_EXACT"
        print(f"[selected] mu={prof['mu']} residual={prof['residual']:.2e} amp={prof['amp']:.3f} "
              f"occ={prof['occ']:.4f} mass={prof['mass']:.1f}", flush=True)
    else:
        # No exact stationary state (Petviashvili: uniform below mu~0.15, no fixed point 0.2-0.22, no branch
        # above saturation ~0.23; imaginary time drains to uniform) -> the C2.1 object is a long-lived
        # QUASI-soliton. Fall back to the dynamically-settled object; the boost diagnostics (P(t), v_peak)
        # still answer the loss-source / velocity-anomaly questions.
        print("[fallback] no exact stationary state -> dynamically-settled quasi-soliton "
              f"(settle Tphys={a.settle_Tphys})", flush=True)
        t0 = time.time()
        psi_exact, prof = settle(a.A, a.sigma, ops, a.N, a.dt, a.settle_Tphys, a.dtchunk)
        if psi_exact is None:
            print("\n=== C2_3_SETTLE_COLLAPSED (CFL?) ===", flush=True)
            json.dump({"N": a.N, "dt": a.dt, "verdict": "C2_3_SETTLE_COLLAPSED"},
                      open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
            print(f"C2_3_DONE {out}", flush=True)
            return
        obj_src = "DYNAMICALLY_SETTLED_QUASISOLITON"
        mu_chk, res = stationarity(psi_exact, ops)
        prof = {**prof, "mu_check": mu_chk, "residual": res}
        print(f"[settled] mass_ret={prof['mass_ret']:.4f} amp={prof['amp']:.3f} occ_ratio={prof['occ_ratio']:.3f} "
              f"n={prof['n']} | stationarity mu={mu_chk:+.4f} residual={res:.2e} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)

    boosts = []
    for nk in kicks:
        t0 = time.time()
        b = boost_exact(psi_exact, ops, Xax, nk, a.N, a.dt, T_steps, a.dtchunk)
        boosts.append(b)
        pret = b["P_ret_traj"][-1] if b.get("P_ret_traj") else np.nan
        print(f"[boost] n={nk} k={b['k']:.3f} v_cent={_f(b.get('v'), '+.5f')} v_peak={_f(b.get('v_peak'), '+.5f')} "
              f"(pred 2Dk={b['v_pred_2Dk']:+.3f}; frac cent={_f(b.get('v_frac_of_galilean'))} "
              f"peak={_f(b.get('v_peak_frac_of_galilean'))}) r2={_f(b.get('r2'), '.2f')}/{_f(b.get('r2_peak'), '.2f')} "
              f"mass={_f(b.get('mass_ret'), '.4f')} P_ret={_f(pret)} ({(time.time()-t0)/60:.1f}m)", flush=True)

    bs = {b["n"]: b for b in boosts if not b.get("collapsed")}
    summary = {"N": a.N, "dt": a.dt, "Tphys": a.Tphys, "D": D, "object_source": obj_src,
               "exact_stationary_found": exact_found, "profile": prof, "boosts": boosts}
    if 0 in bs:
        ctrl = bs[0]["mass_ret"]; summary["control_mass"] = ctrl
        summary["kick_loss"] = {n: round(ctrl - bs[n]["mass_ret"], 4) for n in bs if n != 0}
    kl2 = summary.get("kick_loss", {}).get(2, np.nan)
    ctrl = summary.get("control_mass", np.nan)
    pcons = all(b["P_ret_traj"] and abs(b["P_ret_traj"][-1] - 1.0) < 0.05 for n, b in bs.items() if n != 0) if len(bs) > 1 else False
    if np.isfinite(ctrl) and ctrl > 0.97 and np.isfinite(kl2) and kl2 < 0.02:
        verdict = "C2_PURE_NLS_CLEAN_TRANSPORT_SUPPORTED"
    elif np.isfinite(kl2) and kl2 < 0.045:
        verdict = "C2_EXACT_PROFILE_REDUCES_LOSS_PARTIAL"
    else:
        verdict = "C2_RESIDUAL_NOT_PROFILE_SOURCED"
    if not exact_found:
        verdict += "_QUASISOLITON"
    summary["momentum_conserved"] = bool(pcons)
    summary["verdict"] = verdict
    print(f"\n=== {verdict} | object={obj_src} control={_f(ctrl, '.4f')} kick_loss={summary.get('kick_loss')} "
          f"momentum_conserved={pcons} ===", flush=True)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"C2_3_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
