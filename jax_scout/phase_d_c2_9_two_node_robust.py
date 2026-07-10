"""Phase D / C2.9 — HARDENED two-node qualification. Replaces C2.8b's fragile projected-density peak-tracking with
momentum-based observables (the C2.8b elasticity numbers are NOT trusted; C2.8b = inconclusive, not negative).

Physics of the observable: for iψ_t = -D∇²ψ + N(ρ)ψ (N real), mass continuity is ∂_tρ + ∇·J = 0 with the mass
current J_x = 2D·Im(ψ*∂_xψ). Hence any region's COM velocity is v = 2D·(∫Im ψ*∂_xψ)/(∫ρ) — an INTEGRAL (no peak
ID, robust to breathing/proximity). Per-core quantities come from a window that follows each core's COM.

Diagnostics per frame: total mass & momentum (conservation telemetry), per-core windowed mass M_L/M_R (mass-exchange
detector), per-core windowed velocity v_L/v_R = 2D·P_win/M_win, per-core COM & separation (windowed, periodic-safe),
per-core peak amplitude (shape/breathing proxy), overlap flag. Elasticity e = <|v_L-v_R|>_out / <|v_L-v_R|>_in
measured only over cleanly-separated pre/post windows. Identity through asymmetric collisions is handled by comparing
the outgoing SPEED SET to the incoming one (the well-posed question; pass-through vs rebound is symmetry-undefinable).

  wsl:  python jax_scout/phase_d_c2_9_two_node_robust.py --phi-iso <run>/phi_iso.npy [--quick] [--out DIR]
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

W_WIN = 3.5                       # core window half-width (soliton ell ~ 2.07; window holds one core, edges in low rho)


def profiles(psi, ops):
    """1D x-profiles: rho_x(x) = Sum_yz |psi|^2 ; px_x(x) = Sum_yz Im(psi* d_x psi)  (mass-current / 2D)."""
    dxpsi = np.asarray(jnp.fft.ifftn(ops.ikx * jnp.fft.fftn(jnp.asarray(psi))))
    jx = np.imag(np.conj(psi) * dxpsi)                       # mass-current density / (2D)
    rho = np.abs(psi) ** 2
    return rho.sum(axis=(1, 2)), jx.sum(axis=(1, 2))


def _mi(a, b, L):
    d = a - b
    return d - L * np.round(d / L)


def core_window(rho_x, px_x, x, xc, L, D, w=W_WIN):
    """Windowed core near xc: returns (mass, velocity=2D*P/M, new circular-COM, peak_rho)."""
    m = np.abs(_mi(x, xc, L)) < w
    M = float(rho_x[m].sum())
    if M < 1e-12:
        return 0.0, np.nan, xc, 0.0
    P = float(px_x[m].sum())
    th = 2 * np.pi * x[m] / L
    C = float((rho_x[m] * np.cos(th)).sum()); S = float((rho_x[m] * np.sin(th)).sum())
    xc_new = L * np.arctan2(S, C) / (2 * np.pi)
    v = 2 * D * P / M
    return M, v, xc_new, float(rho_x[m].max())


def _anchor_two(rho_x, x, L):
    """Two dominant well-separated maxima of rho_x (for re-anchoring windows when separated)."""
    idx = np.argsort(rho_x)[::-1]
    p1 = int(idx[0]); p2 = None
    N = len(rho_x)
    for j in idx[1:]:
        d = abs(int(j) - p1); d = min(d, N - d)
        if d > 6:
            p2 = int(j); break
    if p2 is None:
        return float(x[p1]), None
    a, b = sorted([float(x[p1]), float(x[p2])])
    return a, b


def evolve_track(psi0, ops, N, L, dt, Tphys, x, xcL0, xcR0, tag, out, dt_chunk=500):
    D = FAM["D"]
    M0 = float(np.sum(np.abs(psi0) ** 2)); P0 = momentum_x(psi0, ops.ikx)
    pk = physics.initial_psi_k(jnp.asarray(psi0), ops); cur = psi0
    xcL, xcR = xcL0, xcR0
    rec = []
    steps = int(round(Tphys / dt))
    for c in range(steps // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return {"tag": tag, "collapsed": True, "rec": rec}
        rho_x, px_x = profiles(cur, ops)
        ML, vL, xcL_n, aL = core_window(rho_x, px_x, x, xcL, L, D)
        MR, vR, xcR_n, aR = core_window(rho_x, px_x, x, xcR, L, D)
        sep = abs(_mi(xcR_n, xcL_n, L))
        overlap = sep < 2 * W_WIN
        # re-anchor from peaks only when cleanly separated (keeps identity coasting through overlap otherwise)
        if not overlap:
            a, b = _anchor_two(rho_x, x, L)
            if b is not None:
                xcL, xcR = a, b
            else:
                xcL, xcR = xcL_n, xcR_n
        else:
            xcL, xcR = xcL_n, xcR_n
        Mtot = float((np.abs(cur) ** 2).sum())
        rec.append({"t": round((c + 1) * dt_chunk * dt, 3), "sep": round(sep, 3),
                    "xcL": round(xcL_n, 3), "xcR": round(xcR_n, 3),
                    "mL": round(ML / M0, 4), "mR": round(MR / M0, 4), "mtot": round(Mtot / M0, 5),
                    "vL": round(vL, 4) if np.isfinite(vL) else None, "vR": round(vR, 4) if np.isfinite(vR) else None,
                    "ampL": round(aL, 3), "ampR": round(aR, 3),
                    "P": round(momentum_x(cur, ops.ikx), 1), "overlap": bool(overlap)})
    np.savez_compressed(os.path.join(out, f"track_{tag}.npz"),
                        **{k: np.array([r[k] if r[k] is not None else np.nan for r in rec])
                           for k in ("t", "sep", "xcL", "xcR", "mL", "mR", "vL", "vR", "ampL", "ampR", "P")})
    return {"tag": tag, "collapsed": False, "P0": P0, "M0": M0, "rec": rec}


def elasticity(r):
    """e = <|vL-vR|>_out / <|vL-vR|>_in over cleanly-separated pre/post-collision windows."""
    rec = r["rec"]
    seps = np.array([q["sep"] for q in rec]); ts = np.array([q["t"] for q in rec])
    i_min = int(np.argmin(seps)); sep_min = float(seps[i_min])
    clean = 2.2 * W_WIN
    pre = [q for q in rec[:i_min] if q["sep"] > clean and q["vL"] is not None and q["vR"] is not None]
    post = [q for q in rec[i_min + 1:] if q["sep"] > clean and q["vL"] is not None and q["vR"] is not None]

    def relspeed(seg):
        return np.nanmean([abs(q["vL"] - q["vR"]) for q in seg]) if len(seg) >= 2 else np.nan

    vin, vout = relspeed(pre), relspeed(post)
    e = vout / vin if (np.isfinite(vin) and vin > 1e-6 and np.isfinite(vout)) else np.nan
    # per-core outgoing speed set vs incoming
    def speeds(seg):
        return sorted([abs(np.nanmean([q["vL"] for q in seg])), abs(np.nanmean([q["vR"] for q in seg]))]) if seg else [np.nan, np.nan]
    n_end = 1 if (rec[-1]["sep"] < 2 * W_WIN) else 2
    mass_end = rec[-1]["mtot"]
    if n_end == 1 or (np.isfinite(e) and e < 0.15):
        outcome = "CAPTURE"
    elif not np.isfinite(e):
        outcome = "SEPARATION_UNCLEAN"          # never cleanly re-separated within T
    elif e > 0.85:
        outcome = "ELASTIC"
    elif e > 0.4:
        outcome = "INELASTIC_SURVIVE"
    else:
        outcome = "STRONGLY_INELASTIC"
    return {"elasticity": e, "vrel_in": vin, "vrel_out": vout, "sep_min": sep_min,
            "speeds_in": speeds(pre), "speeds_out": speeds(post), "n_end": n_end,
            "mass_end": mass_end, "P0": r["P0"], "P_end": rec[-1]["P"], "outcome": outcome,
            "n_pre": len(pre), "n_post": len(post)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", required=True); ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=20.0); ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--quick", action="store_true", help="validation only: single-core v=2Dk + one static")
    ap.add_argument("--out", default=None)
    a_ = ap.parse_args()
    N, L, dt = a_.N, a_.L, a_.dt
    out = a_.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C2_9_ROBUST_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    ops = build(N, L, dt); x, Xax = axis(N, L); D = FAM["D"]
    phi_iso = np.load(a_.phi_iso); assert phi_iso.shape == (N, N, N)
    print(f"=== C2.9 ROBUST two-node | {FAM} | N={N} L={L} dt={dt} | out={out} ===", flush=True)
    res = {}

    # V0 — OBSERVABLE VALIDATION: single soliton boosted by n=2 should read v = 2Dk via windowed momentum.
    k = 2 * np.pi * 2 / L
    single = (place(phi_iso, 0.0, N, L) * np.exp(1j * k * Xax)).astype(np.complex128)
    r = evolve_track(single, ops, N, L, dt, 1.0, x, 0.0, 0.0, "v0_single", out, dt_chunk=250)
    vmeas = np.nanmean([q["vL"] for q in r["rec"]])
    print(f"[V0] single soliton boost n=2: measured v={vmeas:+.4f} vs 2Dk={2*D*k:+.4f} "
          f"(err {abs(vmeas-2*D*k)/(2*D*k)*100:.2f}%) mass {r['rec'][-1]['mtot']:.5f}", flush=True)
    res["V0_single"] = {"v_meas": float(vmeas), "v_pred": 2 * D * k, "mass_end": r["rec"][-1]["mtot"]}

    # V1 — static in-phase VALIDATION: must reproduce C2.8 attraction (5.83 -> ~1.46).
    # Quick mode uses a short window (attraction is clearly underway by T~5) to keep the smoke bounded (<~5 min).
    v1_T = 5.0 if a_.quick else 12.0
    psi = (place(phi_iso, -3.0, N, L) + place(phi_iso, +3.0, N, L)).astype(np.complex128)
    r = evolve_track(psi, ops, N, L, dt, v1_T, x, -3.0, 3.0, "v1_inphase", out)
    s0, se = r["rec"][0]["sep"], r["rec"][-1]["sep"]
    print(f"[V1] static in-phase (T={v1_T}): sep {s0:.2f} -> {se:.2f} (attracting; C2.8 ATTRACT) "
          f"mass {r['rec'][-1]['mtot']:.4f}", flush=True)
    res["V1_inphase"] = {"sep_start": s0, "sep_end": se}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    if a_.quick:
        print(f"\n=== C2_9_QUICK_VALIDATION_DONE (V0 err + V1 trend above) {out} ===", flush=True); return

    # Static phase sweep (0 and pi re-run as controls) — saved trajectories, robust separation.
    res["static"] = []
    for dphi in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi):
        psi = (place(phi_iso, -3.0, N, L) + np.exp(1j * dphi) * place(phi_iso, +3.0, N, L)).astype(np.complex128)
        r = evolve_track(psi, ops, N, L, dt, 16.0, x, -3.0, 3.0, f"static_{dphi:.2f}", out)
        s0, se = r["rec"][0]["sep"], r["rec"][-1]["sep"]
        merged = r["rec"][-1]["sep"] < 2 * W_WIN
        trend = "MERGE" if merged else "ATTRACT" if se < s0 - 0.5 else "REPEL" if se > s0 + 0.5 else "HOLD"
        print(f"[static dphi={dphi:.3f}] {trend} sep {s0:.2f}->{se:.2f} mass {r['rec'][-1]['mtot']:.4f}", flush=True)
        res["static"].append({"dphi": dphi, "trend": trend, "sep_start": s0, "sep_end": se})
        json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)

    # Asymmetric collisions — robust momentum-based elasticity.
    res["collisions"] = []
    for (nL, nR) in [(3, 2), (4, 3)]:
        kL, kR = 2 * np.pi * nL / L, 2 * np.pi * nR / L
        vL, vR = 2 * D * kL, 2 * D * kR
        psi = (place(phi_iso, -5.0, N, L) * np.exp(1j * kL * (Xax + 5.0))
               + place(phi_iso, +5.0, N, L) * np.exp(-1j * kR * (Xax - 5.0))).astype(np.complex128)
        Tphys = min(10.0 / (vL + vR) + 14.0, 48.0)
        r = evolve_track(psi, ops, N, L, dt, Tphys, x, -5.0, 5.0, f"asym_{nL}_{nR}", out)
        e = elasticity(r)
        e.update({"nL": nL, "nR": nR, "v_in_pred": [vL, -vR]})
        res["collisions"].append(e)
        print(f"[collide {nL}_{nR}] v_in_pred=({vL:.3f},{vR:.3f}) speeds_in={[round(s,3) for s in e['speeds_in']]} "
              f"speeds_out={[round(s,3) for s in e['speeds_out']]} e={e['elasticity'] if np.isfinite(e['elasticity']) else 'na'} "
              f"-> {e['outcome']} | mass {e['mass_end']:.4f} P {e['P0']:.1f}->{e['P_end']:.1f} sep_min={e['sep_min']:.2f}", flush=True)
        json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)

    print(f"\n=== C2_9_DONE {out} ===", flush=True)


if __name__ == "__main__":
    main()
