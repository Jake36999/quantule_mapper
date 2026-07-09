"""Phase D / C2.8b — collision ELASTICITY (the physically-meaningful resolution of C2.8's head-on ambiguity).
For SYMMETRIC identical solitons, pass-through vs rebound is undefinable (mirror symmetry -> identical density).
The measurable question is elasticity: do two clean solitons emerge at the incoming speed? This harness uses
ASYMMETRIC velocities (n_L != n_R) so the two cores have distinct speeds -> continuously trackable, unambiguous
identities, and net P != 0 as a conserved-momentum check. Elasticity e = |v_out_rel| / |v_in_rel|
(e~1 elastic, 0<e<1 inelastic, e~0 capture). Plus a static phase sweep (force-vs-Dphi curve).
Reuses the committed C2.8 soliton object; true-flat substrate; mirror-only; no production changes; no matter claims.

  wsl:  python jax_scout/phase_d_c2_8b_elasticity.py --phi-iso <run>/phi_iso.npy [--out DIR]
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
from jax_scout.phase_d_c2_3_exact_soliton import momentum_x
from jax_scout.phase_d_c2_8_two_node import build, axis, place, FAM


def _subcell_peak(rho_x, i, N, L):
    """Parabolic sub-cell interpolation of a peak at integer index i (periodic)."""
    dx = L / N
    a, b, c = rho_x[(i - 1) % N], rho_x[i], rho_x[(i + 1) % N]
    denom = (a - 2 * b + c)
    off = 0.5 * (a - c) / denom if abs(denom) > 1e-30 else 0.0
    return (-L / 2) + ((i + off) % N) * dx


def two_peaks_tracked(rho_x, N, L, prev):
    """Two well-separated maxima with parabolic refinement; associate to prev positions by nearest (periodic)."""
    idx = np.argsort(rho_x)[::-1]
    p1 = int(idx[0]); p2 = None
    for j in idx[1:]:
        d = abs(int(j) - p1); d = min(d, N - d)
        if d > 6:
            p2 = int(j); break
    if p2 is None:
        return None, None, 1
    xs = sorted([_subcell_peak(rho_x, p1, N, L), _subcell_peak(rho_x, p2, N, L)])
    xL, xR = xs
    if prev is not None:                                    # keep identities by nearest (minimal-image)
        def mi(a, b):
            d = a - b; return d - L * round(d / L)
        if abs(mi(xL, prev[0])) + abs(mi(xR, prev[1])) > abs(mi(xR, prev[0])) + abs(mi(xL, prev[1])):
            xL, xR = xR, xL
    return xL, xR, 2


def _fit_v(ts, xs, L):
    """Velocity from a minimal-image-unwrapped position series (robust to box wrap)."""
    xs = np.asarray(xs, float); ts = np.asarray(ts, float)
    un = xs.copy()
    for i in range(1, len(xs)):
        d = xs[i] - un[i - 1]; un[i] = un[i - 1] + (d - L * round(d / L))
    A = np.vstack([ts, np.ones_like(ts)]).T
    (m, b), *_ = np.linalg.lstsq(A, un, rcond=None)
    return float(m)


def collide_asym(phi_iso, ops, N, L, dt, nL, nR, out, tag, dt_chunk=1000):
    _, Xax = axis(N, L)
    D = FAM["D"]
    kL, kR = 2 * np.pi * nL / L, 2 * np.pi * nR / L
    vL, vR = 2 * D * kL, 2 * D * kR                          # left moves +, right moves -
    psi = (place(phi_iso, -5.0, N, L) * np.exp(1j * kL * (Xax + 5.0))
           + place(phi_iso, +5.0, N, L) * np.exp(-1j * kR * (Xax - 5.0))).astype(np.complex128)
    M0 = float(np.sum(np.abs(psi) ** 2)); P0 = momentum_x(psi, ops.ikx)
    closing = vL + vR; Tphys = min(10.0 / closing + 12.0, 45.0)
    pk = physics.initial_psi_k(jnp.asarray(psi), ops); cur = psi
    ts, xL_s, xR_s, mass_s, P_s, npk_s = [], [], [], [], [], []
    prev = (-5.0, 5.0)
    steps = int(round(Tphys / dt))
    for c in range(steps // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            break
        rho_x = (np.abs(cur) ** 2).sum(axis=(1, 2))
        xL, xR, npk = two_peaks_tracked(rho_x, N, L, prev)
        t = (c + 1) * dt_chunk * dt
        ts.append(t); npk_s.append(npk)
        mass_s.append(float((np.abs(cur) ** 2).sum()) / M0); P_s.append(momentum_x(cur, ops.ikx))
        if npk == 2:
            xL_s.append(xL); xR_s.append(xR); prev = (xL, xR)
        else:
            xL_s.append(np.nan); xR_s.append(np.nan)
    ts = np.array(ts)
    # incoming = first 25% of time (well separated); outgoing = last 25% IF 2 peaks & separated
    n = len(ts); q = max(3, n // 4)
    def seg_v(sl):
        m = [i for i in sl if npk_s[i] == 2 and np.isfinite(xL_s[i])]
        if len(m) < 3:
            return np.nan, np.nan
        return (_fit_v([ts[i] for i in m], [xL_s[i] for i in m], L),
                _fit_v([ts[i] for i in m], [xR_s[i] for i in m], L))
    vL_in, vR_in = seg_v(range(0, q))
    vL_out, vR_out = seg_v(range(n - q, n))
    sep_end = abs((xR_s[-1] - xL_s[-1]) - L * round((xR_s[-1] - xL_s[-1]) / L)) if npk_s[-1] == 2 else np.nan
    vrel_in = abs(vL_in - vR_in); vrel_out = abs(vL_out - vR_out) if np.isfinite(vL_out) else np.nan
    e = vrel_out / vrel_in if (np.isfinite(vrel_out) and vrel_in > 1e-9) else np.nan
    outcome = ("CAPTURE" if (npk_s[-1] == 1 or (np.isfinite(sep_end) and sep_end < 3.0 and (not np.isfinite(e) or e < 0.15)))
               else "ELASTIC" if (np.isfinite(e) and e > 0.85)
               else "INELASTIC_SURVIVE" if (np.isfinite(e) and e > 0.15)
               else "AMBIGUOUS")
    np.savez_compressed(os.path.join(out, f"elas_{tag}.npz"), t=ts, xL=np.array(xL_s), xR=np.array(xR_s),
                        mass=np.array(mass_s), P=np.array(P_s), npk=np.array(npk_s))
    rec = {"tag": tag, "nL": nL, "nR": nR, "vL_in": vL_in, "vR_in": vR_in, "vL_out": vL_out, "vR_out": vR_out,
           "v_in_pred": [vL, -vR], "vrel_in": vrel_in, "vrel_out": vrel_out, "elasticity": e,
           "outcome": outcome, "mass_end": mass_s[-1], "P0": P0, "P_end": P_s[-1], "sep_end": sep_end,
           "npk_end": npk_s[-1], "Tphys": Tphys}
    print(f"[collide {tag}] nL={nL} nR={nR} v_in=({vL_in:+.3f},{vR_in:+.3f}) v_out=({vL_out:+.3f},{vR_out:+.3f}) "
          f"e={e if np.isfinite(e) else 'na'} -> {outcome} | mass {mass_s[-1]:.4f} P {P0:.1f}->{P_s[-1]:.1f} "
          f"sep_end={sep_end}", flush=True)
    return rec


def static_phase(phi_iso, ops, N, L, dt, dphi, out, tag, dt_chunk=1000, Tphys=20.0):
    psi = (place(phi_iso, -3.0, N, L) + np.exp(1j * dphi) * place(phi_iso, +3.0, N, L)).astype(np.complex128)
    M0 = float(np.sum(np.abs(psi) ** 2))
    pk = physics.initial_psi_k(jnp.asarray(psi), ops); cur = psi
    seps, prev = [], (-3.0, 3.0)
    steps = int(round(Tphys / dt))
    for c in range(steps // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            break
        rho_x = (np.abs(cur) ** 2).sum(axis=(1, 2))
        xL, xR, npk = two_peaks_tracked(rho_x, N, L, prev)
        if npk == 2:
            s = abs((xR - xL) - L * round((xR - xL) / L)); seps.append(s); prev = (xL, xR)
        else:
            seps.append(0.0)
    s0, se = seps[0], seps[-1]
    trend = ("MERGE" if se < 1.6 else "ATTRACT" if se < s0 - 0.5 else "REPEL" if se > s0 + 0.5 else "HOLD")
    print(f"[static dphi={dphi:.3f}] {trend} sep {s0:.2f}->{se:.2f} mass {float((np.abs(cur)**2).sum())/M0:.4f}", flush=True)
    return {"tag": tag, "dphi": dphi, "trend": trend, "sep_start": s0, "sep_end": se}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", required=True); ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=20.0); ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--out", default=None)
    a_ = ap.parse_args()
    N, L, dt = a_.N, a_.L, a_.dt
    out = a_.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C2_8B_ELAS_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    ops = build(N, L, dt)
    phi_iso = np.load(a_.phi_iso)
    assert phi_iso.shape == (N, N, N)
    print(f"=== C2.8b ELASTICITY | family {FAM} | N={N} L={L} dt={dt} | phi_iso={a_.phi_iso} | out={out} ===", flush=True)
    res = {"collisions": [], "static_phase": []}
    for dphi in (np.pi / 4, np.pi / 2, 3 * np.pi / 4):      # CHEAP-FIRST: fill force-vs-phase curve (0 and pi from C2.8)
        res["static_phase"].append(static_phase(phi_iso, ops, N, L, dt, dphi, out, f"static_{dphi:.2f}"))
        json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    for (nL, nR) in [(3, 2), (4, 3)]:                       # asymmetric collisions: distinct speeds, net P != 0
        res["collisions"].append(collide_asym(phi_iso, ops, N, L, dt, nL, nR, out, f"asym_{nL}_{nR}"))
        json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    outcomes = [c["outcome"] for c in res["collisions"]]
    verdict = ("C2_8B_ELASTIC" if all(o == "ELASTIC" for o in outcomes) else
               "C2_8B_INELASTIC_SOLITON_INTERACTION" if any(o in ("INELASTIC_SURVIVE", "CAPTURE") for o in outcomes) else
               "C2_8B_AMBIGUOUS")
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} | collision outcomes={outcomes} ===", flush=True)
    print(f"C2_8B_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
