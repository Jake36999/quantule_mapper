"""Phase D / C2.8 — TWO-NODE INTERACTION campaign (old Stage 4, unlocked by C2.7).
First relational-dynamics experiment on the fixed conservative substrate: two TRUE solitons (GALILEAN family A:
a=0.8, s=-0.5, f=-0.1, D=0.3, mu=0.070) in an enlarged box (L=20, N=96 -> same dx/CFL as the validated grid),
each built with its own local phase ramp phi(x-xi)*exp(+-ik(x-xi)) (standard NLS collision construction; symmetric
=> zero net winding, total P ~ 0 control).

Stages:
  T0  generate the soliton in the L=20 box (Petviashvili), isolate (mask pedestal), hold gate (mass_ret > 0.99).
  T1  HEAD-ON collision ladder: sep=10, kicks n in {2,4} per soliton (v=0.188*n each, closing speed 2v);
      outcome classification: PASS_THROUGH (2 nodes survive, velocities ~preserved) / MERGE (1 node) /
      DISRUPT (fragment/disperse). Elasticity = |v_out/v_in|.
  T2  STATIC PAIR at sep=6 with relative phase dphi in {0, pi}: NLS theory predicts in-phase ATTRACTION
      (approach/merge) vs anti-phase REPULSION -- the conservative analog of the dissipative D.5 merge-or-hold law.
Diagnostics: x-projected density rho_x(t) (saved for rendering), two-peak positions/separation (unwrapped),
mass, P_x, amplitudes. Mirror-only; true flat substrate (param_geom_off); no production changes; no matter claims.

  wsl:  python jax_scout/phase_d_c2_8_two_node.py [--out DIR]
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
from jax_scout.phase_d_c2_3_exact_soliton import _petviashvili_chunk, momentum_x

FAM = {"a": 0.8, "s": -0.5, "f": -0.1, "D": 0.3, "mu": 0.0704}


def build(N, L, dt):
    p = {**css.FEB, "param_a": FAM["a"], "param_s": FAM["s"], "param_f": FAM["f"], "param_D": FAM["D"],
         "kinetic_mode": "conservative", "param_a_coupling": 0.0, "param_geom_off": True}
    return physics.build_operators(N, L, dt, p)


def axis(N, L):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    return x, np.meshgrid(x, x, x, indexing="ij")[0]


def seed(A, sig_abs, N, L):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rng = np.random.default_rng(20260709)
    psi = A * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * sig_abs ** 2))
    return (psi + 0.002 * (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))).astype(np.complex128)


def petviashvili_L(A, sig_abs, ops, N, L, mu, iters=600, chunk=50, gamma=1.5, tol=1e-10):
    pk = jnp.fft.fftn(jnp.asarray(seed(A, sig_abs, N, L))) * ops.dealias_mask
    prev = np.asarray(jnp.fft.ifftn(pk))
    for it in range(0, iters, chunk):
        pk, S = _petviashvili_chunk(pk, ops, mu, gamma, chunk)
        cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all() or float(np.max(np.abs(cur))) < 1e-8:
            return None, {"fail": "COLLAPSED"}
        drel = float(np.linalg.norm(cur - prev) / (np.linalg.norm(cur) + 1e-300))
        prev = cur
        if drel < tol:
            break
    # residual of mu*psi = D*lap*psi + g(rho)*psi
    pk_np = np.fft.fftn(cur)
    lap = np.fft.ifftn(np.asarray(ops.minus_k_sq) * pk_np)
    rho = np.abs(cur) ** 2
    Hpsi = FAM["D"] * lap + (FAM["a"] * rho + FAM["s"] * rho ** 2 + FAM["f"] * rho ** 3) * cur
    mu_chk = float(np.real(np.vdot(cur, Hpsi)) / np.real(np.vdot(cur, cur)))
    res = float(np.linalg.norm(Hpsi - mu_chk * cur) / (abs(mu_chk) * np.linalg.norm(cur) + 1e-300))
    return cur, {"residual": res, "mu_check": mu_chk, "amp": float(np.abs(cur).max()),
                 "mass": float(rho.sum()), "S": float(S)}


def isolate_L(psi, N, L, r_cut=4.5, w=0.5):
    rho = np.abs(psi) ** 2
    ip = np.unravel_index(int(np.argmax(rho)), rho.shape)
    ps = np.roll(psi, tuple(N // 2 - i for i in ip), axis=(0, 1, 2))
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    R = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    W = 0.5 * (1.0 - np.tanh((R - r_cut) / w))
    return (ps * W).astype(np.complex128)


def place(phi, x_shift, N, L):
    """Shift a centred profile to x = x_shift along axis 0 (integer-cell roll)."""
    return np.roll(phi, int(round(x_shift / (L / N))), axis=0)


def two_peak(rho_x, N, L):
    """Positions of the two highest well-separated maxima of the x-projected density (units of x)."""
    dx = L / N
    idx = np.argsort(rho_x)[::-1]
    p1 = int(idx[0]); p2 = None
    for i in idx[1:]:
        d = abs(int(i) - p1); d = min(d, N - d)
        if d > 6:
            p2 = int(i); break
    xs = np.linspace(-L / 2, L / 2, N, endpoint=False)
    if p2 is None:
        return [float(xs[p1])], np.nan
    sep = abs(p1 - p2); sep = min(sep, N - sep) * dx
    return sorted([float(xs[p1]), float(xs[p2])]), float(sep)


def evolve_pair(psi0, ops, N, L, dt, Tphys, dt_chunk, tag, out):
    M0 = float(np.sum(np.abs(psi0) ** 2)); P0 = momentum_x(psi0, ops.ikx)
    pk = physics.initial_psi_k(jnp.asarray(psi0), ops)
    traj = []; rho_x_frames = []
    cur = psi0
    steps = int(round(Tphys / dt))
    for c in range(steps // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return {"tag": tag, "collapsed": True, "traj": traj}
        rho = np.abs(cur) ** 2
        rho_x = rho.sum(axis=(1, 2))
        pos, sep = two_peak(rho_x, N, L)
        t = (c + 1) * dt_chunk * dt
        traj.append({"t": round(t, 2), "sep": round(sep, 3) if np.isfinite(sep) else None,
                     "pos": [round(p, 2) for p in pos], "mass": round(float(rho.sum()) / M0, 4),
                     "P": round(momentum_x(cur, ops.ikx), 1), "amp": round(float(np.abs(cur).max()), 3),
                     "n_peaks": len(pos)})
        rho_x_frames.append(rho_x.astype(np.float32))
    np.savez_compressed(os.path.join(out, f"rho_x_{tag}.npz"),
                        rho_x=np.array(rho_x_frames), t=np.array([r["t"] for r in traj]))
    return {"tag": tag, "collapsed": False, "P0": P0, "traj": traj}


def classify_collision(r):
    if r.get("collapsed"):
        return "COLLAPSE"
    tr = r["traj"]
    seps = [q["sep"] for q in tr if q["sep"] is not None]
    n_end = tr[-1]["n_peaks"]; mass_end = tr[-1]["mass"]
    min_sep = min(seps) if seps else np.nan
    if n_end == 2 and mass_end > 0.9 and min_sep < 3.0:
        # they met and both survived
        return "PASS_THROUGH_OR_REBOUND"
    if n_end == 1 and mass_end > 0.8:
        return "MERGE"
    if mass_end < 0.7:
        return "DISRUPT"
    return "AMBIGUOUS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=96); ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.001); ap.add_argument("--dtchunk", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a_ = ap.parse_args()
    N, L, dt = a_.N, a_.L, a_.dt
    out = a_.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C2_8_TWONODE_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    ops = build(N, L, dt)
    x, Xax = axis(N, L)
    D = FAM["D"]
    print(f"=== C2.8 TWO-NODE | family {FAM} | N={N} L={L} dt={dt} | out={out} ===", flush=True)

    # T0: soliton in the L=20 box + isolation + hold gate
    phi, prof = petviashvili_L(1.0, 1.2, ops, N, L, FAM["mu"])
    if phi is None or prof.get("residual", 1) > 1e-6:
        print(f"T0 FAIL: {prof}", flush=True); return
    print(f"[T0] petviashvili L={L}: residual={prof['residual']:.2e} amp={prof['amp']:.3f} mass={prof['mass']:.0f}", flush=True)
    phi_iso = isolate_L(phi, N, L)
    M_iso = float(np.sum(np.abs(phi_iso) ** 2))
    print(f"[T0] isolated: mass {prof['mass']:.0f} -> {M_iso:.0f} (pedestal frac {(prof['mass']-M_iso)/prof['mass']:.3f})", flush=True)
    r0 = evolve_pair(phi_iso, ops, N, L, dt, 2.0, a_.dtchunk, "t0_hold", out)
    hold_mass = r0["traj"][-1]["mass"]
    print(f"[T0] hold T=2: mass_ret={hold_mass:.4f} amp={r0['traj'][-1]['amp']:.3f}", flush=True)
    if hold_mass < 0.98:
        print("T0 GATE FAIL (isolated soliton does not hold)", flush=True); return
    np.save(os.path.join(out, "phi_iso.npy"), phi_iso)

    results = {"T0": {"profile": prof, "pedestal_frac": (prof["mass"] - M_iso) / prof["mass"], "hold_mass": hold_mass}}

    # T1: head-on collisions, sep=10, speed ladder
    for n in (2, 4):
        k = 2 * np.pi * n / L
        v = 2 * D * k
        t_meet = 10.0 / (2 * v)
        Tphys = min(t_meet + 10.0, 40.0)
        psi = (place(phi_iso, -5.0, N, L) * np.exp(1j * k * (Xax + 5.0))
               + place(phi_iso, +5.0, N, L) * np.exp(-1j * k * (Xax - 5.0))).astype(np.complex128)
        print(f"[T1] head-on n={n}: v={v:.3f} each, closing {2*v:.3f}, t_meet~{t_meet:.1f}, T={Tphys:.1f}", flush=True)
        r = evolve_pair(psi, ops, N, L, dt, Tphys, a_.dtchunk, f"t1_headon_n{n}", out)
        verdict = classify_collision(r)
        tr = r["traj"]
        seps = [q["sep"] for q in tr if q["sep"] is not None]
        print(f"     -> {verdict} | min_sep={min(seps) if seps else 'na'} n_end={tr[-1]['n_peaks']} "
              f"mass_end={tr[-1]['mass']} P0={r.get('P0', 0):.1f} P_end={tr[-1]['P']}", flush=True)
        results[f"T1_n{n}"] = {"verdict": verdict, "v_each": v, "traj_tail": tr[-5:], "min_sep": min(seps) if seps else None}
        json.dump(results, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)

    # T2: static pair, relative phase 0 vs pi, sep=6
    for dphi, tag in ((0.0, "inphase"), (np.pi, "antiphase")):
        psi = (place(phi_iso, -3.0, N, L) + np.exp(1j * dphi) * place(phi_iso, +3.0, N, L)).astype(np.complex128)
        print(f"[T2] static pair sep=6 dphi={dphi:.2f} ({tag}), T=20", flush=True)
        r = evolve_pair(psi, ops, N, L, dt, 20.0, a_.dtchunk, f"t2_{tag}", out)
        tr = r["traj"]
        seps = [(q["t"], q["sep"]) for q in tr if q["sep"] is not None]
        s0 = seps[0][1] if seps else np.nan; s_end = seps[-1][1] if seps else np.nan
        n_end = tr[-1]["n_peaks"]
        trend = ("MERGED" if n_end == 1 else
                 "ATTRACT" if np.isfinite(s_end) and s_end < s0 - 0.5 else
                 "REPEL" if np.isfinite(s_end) and s_end > s0 + 0.5 else "HOLD")
        print(f"     -> {trend} | sep {s0} -> {s_end} n_end={n_end} mass_end={tr[-1]['mass']}", flush=True)
        results[f"T2_{tag}"] = {"trend": trend, "sep_start": s0, "sep_end": s_end, "traj_tail": tr[-5:]}
        json.dump(results, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)

    print(f"\n=== C2_8_DONE {out} ===", flush=True)


if __name__ == "__main__":
    main()
