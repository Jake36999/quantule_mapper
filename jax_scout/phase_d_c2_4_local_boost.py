"""Phase D / C2.4 — LOCAL boost (no net winding): the decisive Galilean-transport test.
C2.3 showed the integer-n periodic boost e^{ikx} imposes a GLOBAL topological winding: momentum parks in a
delocalized ring current (which moves no density), the quasi-soliton core creeps by drag (v=0.037k), and the
measured mobility is a drag coefficient, not soliton ballistics. Decisive question: with a LOCAL boost — a smooth
phase ramp whose gradient is ~uniform (= k_loc) across the core and compensated in the far background so the total
ring winding is exactly ZERO — does the quasi-soliton move at the Galilean velocity v = 2*D*k_loc?

  H1 (protocol artifact): v ~ 2*D*k_loc  -> substrate fully transport-capable; C2.3's tiny mu was the winding.
  H2 (physical pinning):  v ~ 0.04*k_loc -> the creep is intrinsic; pinning is physical.
  H3 (partial): core sheds the local phase into background current and ends in between (report the fraction).

Construction: grad_phi(x) = k_loc * [ pgauss(x-x_c, w_up) - c * pgauss(x-x_c-L/2, w_down) ] with c chosen so
sum(grad_phi) = 0 exactly (zero winding); phi = spectral antiderivative; psi -> psi * exp(i*phi(x)). The up-bump
(w_up=2.0) covers the core (radius ~1.2-1.5) so it sees ~uniform k_loc; the compensating down-bump sits at the
antipode in low-density background. Loads the saved settled object (no re-settle). Pure NLS geometry-off; mirror
only; Phase C untouched; no clipping; no matter claims.

  wsl:  python jax_scout/phase_d_c2_4_local_boost.py --object-npy sweep_runs/C23_N96_k2/object_psi.npy \
            --klocs 0,0.314,0.628 --Tphys 6.0 [--out DIR]
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
from jax_scout.phase_d_c2_soliton_scout import occ
from jax_scout.phase_d_c2_2_loss_source import _ops, _axis_grid, _circ_angle, _velocity, _f
from jax_scout.phase_d_c2_3_exact_soliton import momentum_x, stationarity

L = css.L_


def _pgauss(x, c, w):
    """Periodic-wrapped 1D Gaussian bump."""
    d = x - c
    d = d - L * np.round(d / L)
    return np.exp(-0.5 * (d / w) ** 2)


def local_phase(k_loc, N, x_c=0.0, w_up=2.0, w_down=2.0):
    """1D phase profile phi(x): grad phi ~= k_loc across the core, compensated at the antipode, ZERO net winding.
    Returns (phi_1d, diag) with diag = {winding, grad_at_core, grad_min}."""
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    up = _pgauss(x, x_c, w_up)
    dn = _pgauss(x, x_c + L / 2, w_down)
    g = up - (up.sum() / dn.sum()) * dn                       # exact zero mean -> zero winding
    g = g * (k_loc / g.max()) if k_loc != 0 else g * 0.0      # peak local wavenumber = k_loc at the core
    kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    gk = np.fft.fft(g)
    phik = np.zeros_like(gk)
    nz = kx != 0
    phik[nz] = gk[nz] / (1j * kx[nz])
    phi = np.real(np.fft.ifft(phik))
    diag = {"winding": float(g.sum() * (L / N)), "grad_at_core": float(g[N // 2]) if k_loc else 0.0,
            "grad_min": float(g.min())}
    return phi, diag


def core_fraction(rho, N, r_core=2.5):
    """Mass fraction within minimal-image radius r_core of the density peak (core vs background condensate)."""
    ip = np.unravel_index(int(np.argmax(rho)), rho.shape)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    d = []
    for ax in range(3):
        dd = x - x[ip[ax]]
        dd = dd - L * np.round(dd / L)
        d.append(dd)
    R2 = d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2
    return float(rho[R2 < r_core ** 2].sum() / rho.sum())


def run_case(psi0, ops, Xax, k_loc, N, dt, T_steps, dt_chunk):
    dx = L / N
    phi1d, pdiag = local_phase(k_loc, N)
    phi3d = phi1d[:, None, None]
    psi = (psi0 * np.exp(1j * phi3d)).astype(np.complex128)
    M0 = float(np.sum(np.abs(psi) ** 2)); P0 = momentum_x(psi, ops.ikx)
    pk = physics.initial_psi_k(jnp.asarray(psi), ops)
    ang, tt, ptraj, mtraj = [], [], [], []
    cur = psi
    for c in range(T_steps // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return {"k_loc": k_loc, "collapsed": True, "P0": P0, "phase_diag": pdiag}
        rho = np.abs(cur) ** 2
        ang.append(_circ_angle(rho, Xax)); tt.append((c + 1) * dt_chunk * dt)
        ptraj.append(momentum_x(cur, ops.ikx))
        mtraj.append(float(rho.sum()) / M0)
    v, r2, pos = _velocity(ang, tt) if len(tt) >= 3 else (np.nan, np.nan, [0, 0])
    D = float(np.asarray(ops.D_diff))
    v_pred = 2 * D * k_loc
    return {"k_loc": k_loc, "v": float(v), "r2": float(r2), "v_pred_2Dk": v_pred,
            "v_frac_of_galilean": float(v / v_pred) if k_loc else np.nan,
            "disp_box": float((pos[-1] - pos[0]) / L), "mass_ret": mtraj[-1] if mtraj else np.nan,
            "core_frac_start": core_fraction(np.abs(psi) ** 2, N),
            "core_frac_end": core_fraction(np.abs(cur) ** 2, N),
            "P0": P0, "P_traj": [round(p, 3) for p in ptraj], "mass_traj": [round(m, 4) for m in mtraj],
            "n_fin": len(td.detect_nodes(cur, dx)), "phase_diag": pdiag, "collapsed": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-npy", required=True); ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--dt", type=float, default=0.00025); ap.add_argument("--Tphys", type=float, default=6.0)
    ap.add_argument("--klocs", default="0,0.314,0.628"); ap.add_argument("--dtchunk", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    klocs = [float(x) for x in a.klocs.split(",")]
    T_steps = int(round(a.Tphys / a.dt))
    out = a.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C2_4_LOCAL_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    ops = _ops(False, a.N, a.dt)                              # pure NLS geometry-off
    Xax = _axis_grid(a.N)
    D = float(np.asarray(ops.D_diff))
    psi0 = np.load(a.object_npy)
    assert psi0.shape == (a.N, a.N, a.N), f"object shape {psi0.shape} != N={a.N}"
    mu_chk, res = stationarity(psi0, ops)
    print(f"=== C2.4 LOCAL BOOST | N={a.N} dt={a.dt} Tphys={a.Tphys} klocs={klocs} | D={D} "
          f"(Galilean v=2D*k_loc) | object={os.path.basename(a.object_npy)} "
          f"amp={float(np.abs(psi0).max()):.3f} residual={res:.2e} | out={out} ===", flush=True)
    results = []
    for k_loc in klocs:
        t0 = time.time()
        r = run_case(psi0, ops, Xax, k_loc, a.N, a.dt, T_steps, a.dtchunk)
        results.append(r)
        json.dump(results, open(os.path.join(out, "local_boost.json"), "w"), indent=2, default=float)
        if r.get("collapsed"):
            print(f"[local] k_loc={k_loc} -> COLLAPSED", flush=True)
            continue
        pd = r["phase_diag"]
        print(f"[local] k_loc={k_loc:.3f} (winding={pd['winding']:.2e}) v={_f(r['v'], '+.4f')} "
              f"(pred {r['v_pred_2Dk']:+.3f}, frac={_f(r['v_frac_of_galilean'])}) r2={_f(r['r2'], '.2f')} "
              f"disp={_f(r['disp_box'], '+.4f')}box mass={_f(r['mass_ret'], '.4f')} "
              f"core_frac={r['core_frac_start']:.3f}->{r['core_frac_end']:.3f} "
              f"P0={r['P0']:+.1f} P_end={r['P_traj'][-1]:+.1f} n={r['n_fin']} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
    # verdict: compare local-boost velocity fraction against the winding-boost drag (~0.007 of Galilean)
    moved = [r for r in results if r.get("k_loc") and not r.get("collapsed") and np.isfinite(r.get("v", np.nan))]
    fracs = [r["v_frac_of_galilean"] for r in moved]
    if moved and all(f > 0.5 for f in fracs):
        verdict = "C2_LOCAL_BOOST_GALILEAN_TRANSPORT_CONFIRMED"
    elif moved and all(f < 0.05 for f in fracs):
        verdict = "C2_LOCAL_BOOST_STILL_PINNED_PHYSICAL"
    elif moved:
        verdict = "C2_LOCAL_BOOST_PARTIAL_TRANSPORT"
    else:
        verdict = "C2_LOCAL_BOOST_INCONCLUSIVE"
    print(f"\n=== {verdict} | galilean fractions={[round(f, 4) for f in fracs]} ===", flush=True)
    json.dump({"verdict": verdict, "fracs": fracs, "results": results},
              open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"C2_4_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
