"""Phase D / C2.5 P2b — interrogate the 'true solitons found but PINNED' verdict (Tier-1b).
For a TRUE isolated soliton of pure NLS, the winding boost psi*e^{ikx} is an EXACT moving solution (v=2Dk) in the
periodic box — so its response separates three hypotheses for the P2 PINNED result:
  (a) ramp-shape artifact: local ramp (w_up=2.0) was narrower than the ell~2 soliton -> wide ramp (w=3.5) test;
  (b) numerical Galilean breaking: if even the winding boost creeps, N=48/dt numerics pin exact solutions;
  (c) soliton-on-pedestal: Petviashvili at fixed mu can converge to soliton + uniform background (rho_u solving
      g(rho_u)=mu) — a pedestal re-enables C2.4 flow-through pinning; measured via far-field density.
Reports: pedestal level, winding-boost v vs 2Dk, wide-ramp v, P(t), mass(t).

  wsl:  python jax_scout/phase_d_c2_5_p2b_winding.py [--fam 0.8:-0.2:0.0 --D 1.0 --mu 0.2]
"""
import os, sys, json, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css, physics
from jax_scout.phase_d_c1_transport import _evolve_chunk
from jax_scout.phase_d_c2_2_loss_source import _axis_grid, _circ_angle, _velocity
from jax_scout.phase_d_c2_3_exact_soliton import petviashvili, momentum_x
from jax_scout.phase_d_c2_4_local_boost import local_phase, core_fraction
from jax_scout.phase_d_c2_5_family_scout import _ops_family, N, DT

L = css.L_


def _center_peak(arr):
    """Roll a field so the density peak sits at the box centre (N//2,N//2,N//2)."""
    rho = np.abs(arr) ** 2 if np.iscomplexobj(arr) else arr
    ip = np.unravel_index(int(np.argmax(rho)), rho.shape)
    return np.roll(arr, tuple(N // 2 - i for i in ip), axis=(0, 1, 2))


def pedestal(rho):
    """Far-field density: median rho in a corner cube after centring the peak, vs peak."""
    rr = _center_peak(rho)
    far = rr[:N // 8, :N // 8, :N // 8]
    return float(np.median(far)), float(rho.max())


def isolate(psi, r_cut=4.0, w=0.5):
    """Kill the pedestal: centre the peak, apply a smooth radial tanh window -> isolated-soliton candidate."""
    ps = _center_peak(psi)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    R = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    W = 0.5 * (1.0 - np.tanh((R - r_cut) / w))
    return (ps * W).astype(np.complex128)


def evolve_track(psi, ops, Xax, Tphys, dt_chunk=250):
    M0 = float(np.sum(np.abs(psi) ** 2)); P0 = momentum_x(psi, ops.ikx)
    pk = physics.initial_psi_k(jnp.asarray(psi), ops)
    ang, tt, ptr, mtr = [], [], [], []
    cur = psi
    for c in range(int(round(Tphys / DT)) // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return None
        rho = np.abs(cur) ** 2
        ang.append(_circ_angle(rho, Xax)); tt.append((c + 1) * dt_chunk * DT)
        ptr.append(momentum_x(cur, ops.ikx)); mtr.append(float(rho.sum()) / M0)
    v, r2, pos = _velocity(ang, tt)
    return {"v": float(v), "r2": float(r2), "P0": P0, "P_end": ptr[-1], "mass_end": mtr[-1],
            "disp_box": float((pos[-1] - pos[0]) / L), "core_frac_end": core_fraction(np.abs(cur) ** 2, N)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", default="0.8:-0.2:0.0"); ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.2); ap.add_argument("--Tphys", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    a_ = ap.parse_args()
    av, sv, fv = (float(x) for x in a_.fam.split(":"))
    out = a_.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C2_5_P2B_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    ops = _ops_family(av, sv, fv, a_.D)
    Xax = _axis_grid(N)
    D = a_.D
    psi_c, prof = petviashvili(1.0, 0.08, ops, N, a_.mu)
    rho = np.abs(psi_c) ** 2
    ped, pk_rho = pedestal(rho)
    print(f"=== C2.5 P2b | fam a={av} s={sv} f={fv} D={D} mu={a_.mu} | residual={prof['residual']:.1e} "
          f"amp={prof['amp']:.3f} occ={prof['occ']:.4f} ===", flush=True)
    print(f"[pedestal] far-field median rho={ped:.5f} vs peak {pk_rho:.3f} -> ratio {ped/pk_rho:.4f} "
          f"({'SOLITON_ON_PEDESTAL' if ped/pk_rho > 0.01 else 'ISOLATED_SOLITON'})", flush=True)
    results = {"family": [av, sv, fv], "D": D, "mu": a_.mu, "pedestal_ratio": ped / pk_rho, "profile": prof}
    # Test A: winding boost n=1 — EXACT moving solution for a true isolated soliton
    k = 2 * np.pi / L
    r = evolve_track((psi_c * np.exp(1j * k * Xax)).astype(np.complex128), ops, Xax, a_.Tphys)
    r["v_pred"] = 2 * D * k; r["v_frac"] = r["v"] / r["v_pred"]
    results["winding_n1"] = r
    print(f"[winding n=1] v={r['v']:+.4f} (pred {r['v_pred']:+.4f}, frac={r['v_frac']:+.4f}) r2={r['r2']:.2f} "
          f"mass={r['mass_end']:.4f} P {r['P0']:+.1f}->{r['P_end']:+.1f}", flush=True)
    # Test B: wide local ramp (w_up=3.5, still zero winding)
    phi1d, pd = local_phase(k, N, w_up=3.5, w_down=2.0)
    r2b = evolve_track((psi_c * np.exp(1j * phi1d[:, None, None])).astype(np.complex128), ops, Xax, a_.Tphys)
    r2b["v_pred"] = 2 * D * k; r2b["v_frac"] = r2b["v"] / r2b["v_pred"]; r2b["winding"] = pd["winding"]
    results["wide_ramp"] = r2b
    print(f"[wide ramp w=3.5] v={r2b['v']:+.4f} (frac={r2b['v_frac']:+.4f}) r2={r2b['r2']:.2f} "
          f"mass={r2b['mass_end']:.4f} P {r2b['P0']:+.1f}->{r2b['P_end']:+.1f}", flush=True)
    # Test C: ISOLATE the soliton (mask the pedestal), settle briefly, then winding-boost the isolated object.
    psi_iso = isolate(psi_c)
    M_full = float(np.sum(np.abs(psi_c) ** 2)); M_iso0 = float(np.sum(np.abs(psi_iso) ** 2))
    print(f"[isolate] masked pedestal: mass {M_full:.0f} -> {M_iso0:.0f} "
          f"(pedestal fraction {(M_full - M_iso0) / M_full:.3f})", flush=True)
    rs = evolve_track(psi_iso, ops, Xax, 2.0)                 # settle check: does the isolated soliton hold?
    print(f"[isolated settle T=2] mass_ret={rs['mass_end']:.4f} core_frac={rs['core_frac_end']:.3f} "
          f"v_resid={rs['v']:+.4f}", flush=True)
    results["isolated_settle"] = rs
    ri = evolve_track((isolate(psi_c) * np.exp(1j * k * Xax)).astype(np.complex128), ops, Xax, a_.Tphys)
    ri["v_pred"] = 2 * D * k; ri["v_frac"] = ri["v"] / ri["v_pred"]
    results["isolated_winding_n1"] = ri
    print(f"[ISOLATED winding n=1] v={ri['v']:+.4f} (pred {ri['v_pred']:+.4f}, frac={ri['v_frac']:+.4f}) "
          f"r2={ri['r2']:.2f} mass={ri['mass_end']:.4f} P {ri['P0']:+.1f}->{ri['P_end']:+.1f}", flush=True)

    wf, lf, isf = results["winding_n1"]["v_frac"], results["wide_ramp"]["v_frac"], ri["v_frac"]
    verdict = ("P2B_ISOLATED_SOLITON_MOVES_PEDESTAL_PINS" if isf > 0.5 and wf < 0.05 else
               "P2B_GALILEAN_OK_RAMP_ARTIFACT" if wf > 0.5 and lf > 0.5 else
               "P2B_WINDING_MOVES_LOCAL_PINNED" if wf > 0.5 else
               "P2B_PINNED_EVEN_ISOLATED" if isf < 0.05 else
               "P2B_PARTIAL")
    results["verdict"] = verdict
    print(f"\n=== {verdict} | winding_frac={wf:+.4f} wide_ramp_frac={lf:+.4f} ISOLATED_frac={isf:+.4f} "
          f"pedestal={ped/pk_rho:.4f} ===", flush=True)
    json.dump(results, open(os.path.join(out, "p2b.json"), "w"), indent=2, default=float)
    print(f"P2B_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
