"""Matter-likeness Phase 1 — kick / inertia test on the confirmed a* bound state (a x1.15, param_a ~ 0.552).

Endpoint-sequence step 5. The gain-ladder + FEB_ASTAR_CONFIRM runs established that a x1.15 (param_a ~ 0.552,
seed 619, 4-node) is a genuine long-time-stationary bound state (late er slope ~ 0 to T=144000). This asks
whether that state behaves like a PARTICLE: apply a Galilean phase-ramp "kick" exp(i k x) (a clean momentum
boost, k quantised to 2*pi*n/L so it stays periodic on the box), evolve, and measure

  - TRANSPORT: does the center-of-density translate ballistically (constant velocity v)?
  - INERTIA:   is v proportional to the kick k (v = mu*k)? => effective mass m_eff = M / mu (M = total mass).
                A clean, k-independent m_eff is the signature of matter-like inertia.
  - COHERENCE: does the soliton stay intact while moving (mass / peak / node-count / transverse COM held),
                or does the boost shatter/disperse it (=> no robust particle)?

n=0 is the no-kick CONTROL (intrinsic drift/breathing baseline over the same window). COM tracked with a
circular (periodic-box-safe) centroid, unwrapped, then linear-fit for v. Read-only physics; boosts a SAVED
settled field (default = the gain-ladder a1.15 psi_fin), so no re-settling needed. Geometry frozen e8d6a78ea.
WSL2 jax venv:  python jax_scout/feb_kick_inertia.py [--state NPZ] [--kicks 0,1,2,3] [--tkick 10000] [--out DIR]
"""
import os, sys, csv, json, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css, physics
from jax_scout import transfer_diag as td

N = 96
L, DT = css.L_, css.DT                 # 10.0, 0.005
DX = L / N
A_FACTOR = 1.15                        # confirmed a* (param_a ~ 0.552)
DEFAULT_STATE = os.path.join(ROOT, "sweep_runs",
                             "FEB_GAIN_LADDER_LONGT_T72000_20260701_175708", "a1.15_ladder_T72000_probe.npz")
COLS = ["kick_n", "k", "v_x", "v_y", "v_z", "mobility_v_over_k", "disp_x_boxes",
        "mass_ratio_fin", "peak_ratio_fin", "n_fin_start", "n_fin_end", "com_fit_r2", "wallclock_min"]

# coordinate grids (match family_ics: linspace(-L/2, L/2, N, endpoint=False), indexing="ij")
_x = np.linspace(-L / 2, L / 2, N, endpoint=False)
_X = [g for g in np.meshgrid(_x, _x, _x, indexing="ij")]      # [X, Y, Z]


def circ_angle(rho, axis):
    """Circular centroid ANGLE (rad, in (-pi,pi]) along one axis — periodic-box safe."""
    th = 2 * np.pi * _X[axis] / L
    C = float(np.sum(rho * np.cos(th))); S = float(np.sum(rho * np.sin(th)))
    return np.arctan2(S, C)


def velocity(angles, t_phys):
    """Unwrap the angle sequence -> position (box units of L), linear-fit -> velocity + R^2."""
    pos = L * np.unwrap(np.asarray(angles)) / (2 * np.pi)
    t = np.asarray(t_phys)
    A = np.vstack([t, np.ones_like(t)]).T
    (m, b), res, *_ = np.linalg.lstsq(A, pos, rcond=None)
    ss_tot = np.sum((pos - pos.mean()) ** 2)
    r2 = 1.0 - (float(res[0]) / ss_tot if res.size and ss_tot > 0 else 0.0)
    return float(m), float(r2), pos


def evolve_kick(psi0, pvec, n_kick, t_kick, dt_chunk):
    """Boost psi0 by exp(i k x), k = 2*pi*n/L; chunk-evolve; return COM/mass/peak trajectories."""
    k = 2 * np.pi * n_kick / L
    psi = (np.asarray(psi0) * np.exp(1j * k * _X[0])).astype(np.complex128)
    M0 = float(np.sum(np.abs(psi) ** 2)); p0 = float(np.max(np.abs(psi) ** 2))
    n_start = len(td.detect_nodes(psi, DX))
    n_chunks = t_kick // dt_chunk
    ang = {0: [], 1: [], 2: []}; t_phys = []; mass_r = []; peak_r = []
    cur = psi
    for c in range(n_chunks):
        _, pf, _, _, finite = physics.probe_one(jnp.asarray(pvec), jnp.asarray(cur),
                                                N, L, DT, dt_chunk, jnp.float64, jnp.complex128)
        cur = np.asarray(pf); rho = np.abs(cur) ** 2
        for ax in (0, 1, 2):
            ang[ax].append(circ_angle(rho, ax))
        t_phys.append((c + 1) * dt_chunk * DT)
        mass_r.append(float(np.sum(rho)) / M0); peak_r.append(float(np.max(rho)) / p0)
        if not bool(finite):
            break
    n_end = len(td.detect_nodes(cur, DX))
    vx, r2x, posx = velocity(ang[0], t_phys)
    vy, _, _ = velocity(ang[1], t_phys); vz, _, _ = velocity(ang[2], t_phys)
    return {"k": k, "v_x": vx, "v_y": vy, "v_z": vz, "com_fit_r2": r2x,
            "disp_x_boxes": float(posx[-1] - posx[0]) / L, "mass_ratio_fin": mass_r[-1],
            "peak_ratio_fin": peak_r[-1], "n_fin_start": n_start, "n_fin_end": n_end,
            "traj": {"t": t_phys, "pos_x": posx.tolist(), "mass_r": mass_r, "peak_r": peak_r,
                     "ang_y": ang[1], "ang_z": ang[2]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=DEFAULT_STATE, help="npz with psi_fin of the settled a* state")
    ap.add_argument("--kicks", default="0,1,2,3", help="comma list of integer kick numbers n (k=2*pi*n/L)")
    ap.add_argument("--tkick", type=int, default=10000, help="steps to evolve each kicked state")
    ap.add_argument("--dtchunk", type=int, default=100, help="steps per COM sample")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    kicks = [int(x) for x in args.kicks.split(",")]
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_KICK_INERTIA_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "feb_kick_inertia_results.csv")

    psi_settle = np.load(args.state)["psi_fin"].astype(np.complex128)
    pp = dict(css.FEB); pp["param_a"] = float(css.FEB["param_a"]) * A_FACTOR
    pvec = np.asarray([pp[k] for k in css.order], dtype=np.float64)
    print(f"=== FEB KICK / INERTIA (a x{A_FACTOR}, param_a={pp['param_a']:.4f}) | state={os.path.basename(args.state)} "
          f"| kicks={kicks} tkick={args.tkick} | out={out} ===", flush=True)
    print(f"    L={L} dt={DT} N={N}; k_n = 2*pi*n/L; settled mass={float(np.sum(np.abs(psi_settle)**2)):.1f} "
          f"n_nodes={len(td.detect_nodes(psi_settle, DX))}", flush=True)

    rows, trajs = [], {}
    for n in kicks:
        t0 = time.time()
        r = evolve_kick(psi_settle, pvec, n, args.tkick, args.dtchunk)
        r["wallclock_min"] = round((time.time() - t0) / 60.0, 1)
        r["kick_n"] = n; r["mobility_v_over_k"] = (r["v_x"] / r["k"]) if r["k"] != 0 else 0.0
        trajs[f"kick{n}"] = r.pop("traj")
        rows.append({c: r.get(c) for c in COLS})
        print(f"  n={n} k={r['k']:.3f} -> v_x={r['v_x']:+.4f} (v/k={r['mobility_v_over_k']:+.4f}) "
              f"disp={r['disp_x_boxes']:+.2f}box r2={r['com_fit_r2']:.3f} | v_perp=({r['v_y']:+.3f},{r['v_z']:+.3f}) "
              f"mass={r['mass_ratio_fin']:.3f} peak={r['peak_ratio_fin']:.3f} nodes {r['n_fin_start']}->{r['n_fin_end']} "
              f"({r['wallclock_min']}m)", flush=True)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    # inertia fit: v_x vs k through the kicked points (mobility mu; effective mass = M/mu)
    kk = np.array([r["k"] for r in rows]); vv = np.array([r["v_x"] for r in rows])
    mu = float(np.polyfit(kk, vv, 1)[0]) if len(kk) >= 2 else float("nan")
    M = float(np.sum(np.abs(psi_settle) ** 2))
    m_eff = (M / mu) if mu not in (0.0, float("nan")) and abs(mu) > 1e-9 else float("nan")
    summary = {"a_factor": A_FACTOR, "param_a": pp["param_a"], "state": args.state, "N": N, "L": L, "dt": DT,
               "tkick": args.tkick, "dtchunk": args.dtchunk, "kicks": kicks, "mass_M": M,
               "mobility_mu_v_per_k": mu, "effective_mass_M_over_mu": m_eff,
               "feb_params": css.FEB, "rows": rows}
    json.dump(summary, open(os.path.join(out, "feb_kick_inertia_summary.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(out, "kick_trajectories.npz"),
                        **{k: np.array(v["pos_x"]) for k, v in trajs.items()},
                        **{f"{k}_t": np.array(v["t"]) for k, v in trajs.items()})
    print(f"=== INERTIA: mobility mu=v/k={mu:+.4f}, mass M={M:.1f} -> effective mass M/mu={m_eff:.1f} ===", flush=True)
    print(f"=== DONE {out} ===", flush=True)


if __name__ == "__main__":
    main()
