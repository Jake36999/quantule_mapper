"""Phase D / C3 — TWO-Q-BALL interaction (KG analog of the C2.9 NLS two-node study). Bounded.
The C3 Q-ball is existence- AND VK-stability-confirmed (docs/PHASE_D_C3_WAVE_KINETIC_RESULTS.md), so a two-body
study is well-motivated. Questions: (1) static-pair force vs relative phase Dphi (does it attract/repel with a
crossover, like the C2.9 NLS pi/2 law?); (2) head-on collision outcome (capture / pass-through / inelastic).
Second-order KG substrate: momentum lives in the field configuration, so this is genuine relational dynamics.

Observable: windowed centroid of |psi|^2 per core (periodic-safe), separation vs time; total E/Q/P conservation
telemetry (Strang splitting error meter). Reuses the validated C3 machinery. Standalone; no production/Phase C path.

  wsl:  python jax_scout/phase_d_c3_two_qball.py [--N 64 --L 16 --out DIR]
"""
import os, sys, json, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.phase_d_c3_wave import build_kg, kg_evolve, qball_petviashvili, contract_axis0, invariants

W_WIN = 3.0


def place(phi, x_shift, L):
    N = phi.shape[0]
    return np.roll(phi, int(round(x_shift / (L / N))), axis=0)


def _mi(a, b, L):
    d = a - b
    return d - L * np.round(d / L)


def core_com(rx, x, xc, L, w=W_WIN):
    """Windowed circular centroid of the 1D density profile near xc (periodic-safe); returns (xc_new, mass)."""
    m = np.abs(_mi(x, xc, L)) < w
    M = float(rx[m].sum())
    if M < 1e-12:
        return xc, 0.0
    th = 2 * np.pi * x[m] / L
    C = float((rx[m] * np.cos(th)).sum()); S = float((rx[m] * np.sin(th)).sum())
    return L * np.arctan2(S, C) / (2 * np.pi), M


def boosted_qball(phi, w, c, v, x_shift, L, op, dphi=0.0):
    """One Q-ball at x_shift moving at velocity v (signed), relative phase dphi. Returns (psi, pi) components."""
    N = phi.shape[0]
    gam = 1.0 / np.sqrt(1.0 - (v / c) ** 2) if v != 0 else 1.0
    k = gam * w * v / c ** 2
    phi_c = contract_axis0(phi, gam, L) if v != 0 else phi
    phi_s = place(phi_c, x_shift, L)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)[:, None, None]
    carrier = np.exp(1j * (k * _mi(x, x_shift, L) + dphi))          # carrier centered on the core + relative phase
    gx = np.asarray(jnp.fft.ifftn(op["ikx"] * jnp.fft.fftn(jnp.asarray(phi_s.astype(np.complex128)))))
    psi = (phi_s * carrier).astype(np.complex128)
    pi = ((-v * gx - 1j * gam * w * phi_s) * carrier).astype(np.complex128)
    return psi, pi


def evolve_pair(psi, pi, op, a, s, f, Tphys, xcL0, xcR0, tag, out, sample_every=2):
    N, L, dt = op["N"], op["L"], op["dt"]
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    inv0 = invariants(jnp.fft.fftn(jnp.asarray(psi)), jnp.fft.fftn(jnp.asarray(pi)), op, a, s, f)
    pk = jnp.fft.fftn(jnp.asarray(psi)); qk = jnp.fft.fftn(jnp.asarray(pi))
    xcL, xcR = xcL0, xcR0
    traj = []
    steps = int(round(Tphys / dt)); chunk = 250
    for cix in range(steps // chunk):
        pk, qk = kg_evolve(pk, qk, op, a, s, f, chunk)
        if not np.isfinite(np.asarray(pk)).all():
            traj.append({"t": (cix + 1) * chunk * dt, "collapsed": True}); break
        if cix % sample_every == 0 or cix == steps // chunk - 1:
            cur = np.asarray(jnp.fft.ifftn(pk)); rx = (np.abs(cur) ** 2).sum(axis=(1, 2))
            xcL, mL = core_com(rx, x, xcL, L); xcR, mR = core_com(rx, x, xcR, L)
            sep = abs(_mi(xcR, xcL, L))
            inv = invariants(pk, qk, op, a, s, f)
            traj.append({"t": round((cix + 1) * chunk * dt, 3), "sep": round(sep, 3),
                         "xcL": round(xcL, 3), "xcR": round(xcR, 3), "mL": round(mL, 2), "mR": round(mR, 2),
                         "dE_rel": abs(inv["E"] - inv0["E"]) / (abs(inv0["E"]) + 1e-30),
                         "dQ_rel": abs(inv["Q"] - inv0["Q"]) / (abs(inv0["Q"]) + 1e-30),
                         "P": inv["Px"], "amp": round(inv["amp"], 3)})
    np.savez_compressed(os.path.join(out, f"pair_{tag}.npz"),
                        t=np.array([q.get("t", np.nan) for q in traj]),
                        sep=np.array([q.get("sep", np.nan) for q in traj]))
    return traj, inv0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--c", type=float, default=0.5477); ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--a", type=float, default=0.8); ap.add_argument("--s", type=float, default=-0.5)
    ap.add_argument("--f", type=float, default=-0.1); ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--wfrac", type=float, default=0.964); ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="static phase 0/pi + one collision only")
    A = ap.parse_args()
    out = A.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C3_TWOQBALL_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    op = build_kg(A.N, A.L, A.c, A.m, A.dt)
    a, s, f = A.a, A.s, A.f
    w = A.wfrac * A.m; mu = A.m ** 2 - w ** 2
    print(f"=== C3 TWO-Q-BALL | N={A.N} L={A.L} c={A.c} m={A.m} w={w:.3f} mu={mu:.3f} | out={out} ===", flush=True)

    # T0: solve the VK-stable Q-ball in this box (seed scan), verify single-Q-ball hold.
    phi = None
    for sg in (1.5, 1.2, 1.8, 2.0):
        ph, pr = qball_petviashvili(op, a, s, f, mu, sig=sg)
        if ph is not None and pr["residual"] < 1e-6 and pr["occ"] < 0.5:
            phi = ph; prof = pr; break
    if phi is None:
        print(f"T0 FAIL: no Q-ball in L={A.L} box", flush=True); return
    print(f"[T0] Q-ball: residual={prof['residual']:.2e} amp={prof['amp']:.3f} occ={prof['occ']:.4f}", flush=True)
    res = {"config": vars(A), "w": w, "mu": mu, "qball": prof, "static": [], "collisions": []}

    # STATIC pair: relative-phase force sweep (rest Q-balls, sep d0).
    d0 = 6.0
    phases = [0.0, np.pi] if A.quick else [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    for dphi in phases:
        psi = place(phi, -d0 / 2, A.L) + np.exp(1j * dphi) * place(phi, +d0 / 2, A.L)
        pi = (-1j * w * psi).astype(np.complex128)                  # two rest Q-balls: pi=-i w psi
        traj, _ = evolve_pair(psi.astype(np.complex128), pi, op, a, s, f, 15.0, -d0 / 2, d0 / 2,
                              f"static_{dphi:.2f}", out)
        s0, se = traj[0]["sep"], traj[-1].get("sep", np.nan)
        dEr = max((q.get("dE_rel", 0) for q in traj if "dE_rel" in q), default=np.nan)
        trend = ("MERGE" if np.isfinite(se) and se < 2 * W_WIN else
                 "ATTRACT" if np.isfinite(se) and se < s0 - 0.4 else
                 "REPEL" if np.isfinite(se) and se > s0 + 0.4 else "HOLD")
        print(f"[static dphi={dphi:.3f}] {trend} sep {s0:.2f}->{se:.2f} (dE_rel<={dEr:.1e})", flush=True)
        res["static"].append({"dphi": dphi, "trend": trend, "sep_start": s0, "sep_end": se, "dE_rel_max": dEr})
        json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)

    # COLLISION: symmetric head-on (in-phase), speed ladder.
    vlist = [0.25 * A.c] if A.quick else [0.15 * A.c, 0.30 * A.c]
    for v in vlist:
        psiL, piL = boosted_qball(phi, w, A.c, +v, -5.0, A.L, op)   # left, moving right
        psiR, piR = boosted_qball(phi, w, A.c, -v, +5.0, A.L, op)   # right, moving left
        psi = psiL + psiR; pi = piL + piR
        Tp = min(10.0 / (2 * v) + 12.0, 40.0)
        traj, inv0 = evolve_pair(psi, pi, op, a, s, f, Tp, -5.0, 5.0, f"collide_v{v:.3f}", out)
        seps = [q["sep"] for q in traj if q.get("sep") is not None]
        i_min = int(np.argmin(seps)) if seps else 0; sep_min = seps[i_min] if seps else np.nan
        se = traj[-1].get("sep", np.nan); n_end = 2 if (np.isfinite(se) and se > 2 * W_WIN) else 1
        outcome = ("CAPTURE" if n_end == 1 or (np.isfinite(se) and se < 2 * W_WIN) else
                   "SURVIVE_SEPARATE")
        dEr = max((q.get("dE_rel", 0) for q in traj if "dE_rel" in q), default=np.nan)
        print(f"[collide v={v:.3f}] {outcome} sep_min={sep_min:.2f} sep_end={se:.2f} (dE_rel<={dEr:.1e})", flush=True)
        res["collisions"].append({"v": v, "outcome": outcome, "sep_min": sep_min, "sep_end": se, "dE_rel_max": dEr})
        json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)

    # crossover summary (static force vs phase)
    trends = {round(r["dphi"], 3): r["trend"] for r in res["static"]}
    print(f"\n=== C3_TWOQBALL_DONE | static force vs phase: {trends} | collisions "
          f"{[(round(c['v'],3), c['outcome']) for c in res['collisions']]} ===", flush=True)
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"C3_TWOQBALL_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
