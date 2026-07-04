"""Phase D / C2.1 Stage 2+3 — confirm native conservative candidates at N=96 (long window) + BOOST them (transport).
For each (A, sigma) candidate: evolve conservatively to confirm a stable localized soliton (mass/occ/amp/node
trajectory), then Galilean-boost the settled state (n=0,1,2) and measure centroid velocity / mobility / mass
retention during motion. Success = a native soliton that MOVES coherently. Mirror only; no solver default/gate change.

  wsl:  python jax_scout/phase_d_c2_confirm_boost.py [--Tconfirm 12000 --Tboost 6000 --out DIR]
"""
import os, sys, json, csv, time, argparse
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
from jax_scout.feb_kick_inertia import circ_angle, velocity, _X

N, L, DX, DT = 96, css.L_, css.L_ / 96, 0.001
CANDS = [(1.0, 0.083), (1.0, 0.11), (1.0, 0.15), (0.5, 0.11)]     # from the N=48 scout LOCALIZED region


def _ops():
    return physics.build_operators(N, L, DT, {**css.FEB, "param_a": float(css.FEB["param_a"]) * 1.15,
                                              "kinetic_mode": "conservative"})


def confirm(A, sig, ops, T, dt_chunk):
    psi = gaussian_ic(A, sig, N)
    M0 = float(np.sum(np.abs(psi) ** 2)); amp0 = float(np.max(np.abs(psi))); occ0 = occ(np.abs(psi) ** 2)
    pk = physics.initial_psi_k(jnp.asarray(psi), ops); cur = psi; traj = []
    for c in range(T // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return None, "COLLAPSE", traj
        rho = np.abs(cur) ** 2
        traj.append({"t": (c + 1) * dt_chunk * DT, "mass_ret": float(rho.sum()) / M0,
                     "amp_ret": float(np.max(np.abs(cur))) / (amp0 + 1e-30), "occ_ratio": occ(rho) / (occ0 + 1e-30),
                     "n": len(td.detect_nodes(cur, DX))})
    last = traj[-1]; half = traj[len(traj) // 2]
    stable = (last["mass_ret"] > 0.55 and last["occ_ratio"] < 2.5 and 1 <= last["n"] <= 2
              and abs(last["occ_ratio"] - half["occ_ratio"]) < 0.4)     # not still spreading
    return cur, ("STABLE_SOLITON" if stable else "DISPERSING"), traj


def boost(settled, ops, n_kick, T, dt_chunk):
    k = 2 * np.pi * n_kick / L
    psi = (settled * np.exp(1j * k * _X[0])).astype(np.complex128)
    M0 = float(np.sum(np.abs(psi) ** 2))
    pk = physics.initial_psi_k(jnp.asarray(psi), ops); ang = []; tt = []; cur = psi
    for c in range(T // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            break
        ang.append(circ_angle(np.abs(cur) ** 2, 0)); tt.append((c + 1) * dt_chunk * DT)
    if len(tt) < 3:
        return {"n": n_kick, "k": k, "v": np.nan, "disp": np.nan, "r2": np.nan, "mass_ret": np.nan, "n_fin": 0}
    v, r2, pos = velocity(ang, tt)
    return {"n": n_kick, "k": k, "v": float(v), "disp": float((pos[-1] - pos[0]) / L), "r2": float(r2),
            "mass_ret": float(np.sum(np.abs(cur) ** 2)) / M0, "n_fin": len(td.detect_nodes(cur, DX))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Tconfirm", type=int, default=12000); ap.add_argument("--Tboost", type=int, default=6000)
    ap.add_argument("--dtchunk", type=int, default=1000); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C2_CONFIRM_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    ops = _ops()
    print(f"=== C2.1 STAGE 2+3 confirm+boost | N={N} dt={DT} Tconfirm={a.Tconfirm} Tboost={a.Tboost} | cands={CANDS} ===", flush=True)
    results = []
    for (A, sig) in CANDS:
        t0 = time.time()
        settled, verdict, traj = confirm(A, sig, ops, a.Tconfirm, a.dtchunk)
        tr = traj[-1] if traj else {}
        print(f"[confirm] A={A} sig={sig} -> {verdict} mass_ret={tr.get('mass_ret')} occ={tr.get('occ_ratio')} "
              f"n={tr.get('n')} ({(time.time()-t0)/60:.1f}m)", flush=True)
        rec = {"A": A, "sigma": sig, "confirm": verdict, "traj_fin": tr, "boosts": []}
        if verdict == "STABLE_SOLITON":
            for nk in (0, 1, 2):
                b = boost(settled, ops, nk, a.Tboost, a.dtchunk)
                rec["boosts"].append(b)
                print(f"   [boost] n={nk} k={b['k']:.3f} -> v={b['v']:+.4f} (v/k={b['v']/b['k'] if b['k'] else 0:+.4f}) "
                      f"disp={b['disp']:+.4f}box r2={b['r2']:.2f} mass={b['mass_ret']:.2f} n={b['n_fin']}", flush=True)
            kk = np.array([b["k"] for b in rec["boosts"]]); vv = np.array([b["v"] for b in rec["boosts"]])
            mu = float(np.polyfit(kk, vv, 1)[0]) if kk.std() > 0 and np.isfinite(vv).all() else float("nan")
            rec["mobility_mu"] = mu
            coh = all(0.5 <= b["mass_ret"] <= 1.5 and b["n_fin"] >= 1 for b in rec["boosts"] if np.isfinite(b["mass_ret"]))
            rec["moves_coherently"] = bool(abs(mu) > 0.05 and coh)
            print(f"   => mobility mu={mu:+.4f} moves_coherently={rec['moves_coherently']}", flush=True)
        results.append(rec)
        json.dump(results, open(os.path.join(out, "confirm_boost.json"), "w"), indent=2, default=float)
    movers = [r for r in results if r.get("moves_coherently")]
    solitons = [r for r in results if r.get("confirm") == "STABLE_SOLITON"]
    verdict = ("C2_NATIVE_SOLITON_FOUND_TRANSPORT_SUPPORTED" if movers else
               "C2_NATIVE_SOLITON_FOUND_NO_TRANSPORT" if solitons else "C2_NATIVE_SOLITON_NOT_CONFIRMED")
    print(f"\n=== {verdict} | {len(solitons)} stable solitons, {len(movers)} move coherently ===", flush=True)
    json.dump({"verdict": verdict, "n_solitons": len(solitons), "n_movers": len(movers), "results": results},
              open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"C2_CONFIRM_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
