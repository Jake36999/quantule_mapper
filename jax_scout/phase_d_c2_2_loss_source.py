"""Phase D / C2.2 — LOSS-SOURCE ISOLATION for the native conservative soliton's lossy ballistic transport.
C2.1 found native C2 solitons that translate ballistically (v proportional to k) but shed a kick-associated,
approximately-k^2 radiative loss on top of a resting quasi-conservative bleed. Question: what CAUSES the
kick-associated loss? Candidates: (1) the density-sourced conformal geometry Omega^2(rho) breaking Galilean
invariance; (2) the instantaneous boost protocol; (3) resolution/dt; (4) intrinsic non-integrability.

Decisive test = GEOMETRY-OFF contrast. Setting param_a_coupling=0 makes omega_sq == 1 and d_omega_d_rho == 0
EXACTLY, so the covariant Laplacian collapses to the flat Laplacian and D_diff*(lap_cov - lap_flat) vanishes ->
a pure cubic-quintic-septic NLS, which is Galilean-invariant. If the boosted soliton then stops radiating, geometry
is the culprit; if it still radiates, geometry is not.

Mirror only; Phase C default (dissipative, a_coupling>0) UNTOUCHED; conservative branch reached only here via
build_operators; no clipping/caps; no matter claims. N-aware helpers (own grid) so the resolution spot check works.

  wsl:  python jax_scout/phase_d_c2_2_loss_source.py --geom both --cands 1.0:0.15,0.5:0.15 --kicks 0,1,2,3
        python jax_scout/phase_d_c2_2_loss_source.py --geom on --cands 1.0:0.15 --kicks 0,2 --N 128   # res spot
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

L = css.L_


def _f(x, p="+.4f"):
    """Format a float robustly for logging (None / non-finite -> 'nan')."""
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, p)


# ---- N-aware periodic-box helpers (do NOT reuse the N=96-locked feb_kick_inertia grids) ----
def _axis_grid(N):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, _, _ = np.meshgrid(x, x, x, indexing="ij")
    return X                                             # axial coordinate (axis 0)


def _circ_angle(rho, Xax):
    th = 2 * np.pi * Xax / L
    C = float(np.sum(rho * np.cos(th))); S = float(np.sum(rho * np.sin(th)))
    return np.arctan2(S, C)


def _velocity(angles, t_phys):
    pos = L * np.unwrap(np.asarray(angles)) / (2 * np.pi)
    t = np.asarray(t_phys)
    A = np.vstack([t, np.ones_like(t)]).T
    (m, b), res, *_ = np.linalg.lstsq(A, pos, rcond=None)
    ss_tot = np.sum((pos - pos.mean()) ** 2)
    r2 = 1.0 - (float(res[0]) / ss_tot if res.size and ss_tot > 0 else 0.0)
    return float(m), float(r2), pos


def _ops(geom_on, N, dt):
    """Conservative substrate. geom_on=False sets param_a_coupling=0 => exact flat (pure NLS)."""
    p = {**css.FEB, "param_a": float(css.FEB["param_a"]) * 1.15, "kinetic_mode": "conservative"}
    if not geom_on:
        p["param_a_coupling"] = 0.0
    return physics.build_operators(N, L, dt, p)


def confirm(A, sig, ops, N, dt, T, dt_chunk):
    dx = L / N
    psi = gaussian_ic(A, sig, N)
    M0 = float(np.sum(np.abs(psi) ** 2)); occ0 = occ(np.abs(psi) ** 2); amp0 = float(np.max(np.abs(psi)))
    pk = physics.initial_psi_k(jnp.asarray(psi), ops); cur = psi; traj = []
    for c in range(T // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return None, "COLLAPSE", traj
        rho = np.abs(cur) ** 2
        traj.append({"t": (c + 1) * dt_chunk * dt, "mass_ret": float(rho.sum()) / M0,
                     "amp_ret": float(np.max(np.abs(cur))) / (amp0 + 1e-30),
                     "occ_ratio": occ(rho) / (occ0 + 1e-30), "n": len(td.detect_nodes(cur, dx))})
    last = traj[-1]; half = traj[len(traj) // 2]
    stable = (last["mass_ret"] > 0.55 and last["occ_ratio"] < 2.5 and 1 <= last["n"] <= 2
              and abs(last["occ_ratio"] - half["occ_ratio"]) < 0.4)
    return cur, ("STABLE_SOLITON" if stable else "DISPERSING"), traj


def boost(settled, ops, Xax, n_kick, N, dt, T, dt_chunk):
    dx = L / N
    k = 2 * np.pi * n_kick / L
    psi = (settled * np.exp(1j * k * Xax)).astype(np.complex128)
    M0 = float(np.sum(np.abs(psi) ** 2))
    pk = physics.initial_psi_k(jnp.asarray(psi), ops); ang = []; tt = []; cur = psi
    for c in range(T // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            break
        ang.append(_circ_angle(np.abs(cur) ** 2, Xax)); tt.append((c + 1) * dt_chunk * dt)
    if len(tt) < 3:
        return {"n": n_kick, "k": k, "v": np.nan, "disp": np.nan, "r2": np.nan, "mass_ret": np.nan, "n_fin": 0}
    v, r2, pos = _velocity(ang, tt)
    return {"n": n_kick, "k": k, "v": float(v), "disp": float((pos[-1] - pos[0]) / L), "r2": float(r2),
            "mass_ret": float(np.sum(np.abs(cur) ** 2)) / M0, "n_fin": len(td.detect_nodes(cur, dx))}


def analyse(rec):
    """Derive kick-associated loss vs the n=0 control, k^2 scaling, mobility, cleanliness."""
    bs = {b["n"]: b for b in rec["boosts"] if np.isfinite(b.get("mass_ret", np.nan))}
    if 0 not in bs:
        return rec
    ctrl = bs[0]["mass_ret"]; rec["control_mass"] = ctrl
    rec["kick_loss"] = {n: ctrl - bs[n]["mass_ret"] for n in bs if n != 0}
    kk = np.array([b["k"] for b in rec["boosts"]]); vv = np.array([b["v"] for b in rec["boosts"]])
    good = np.isfinite(vv)
    rec["mobility_mu"] = float(np.polyfit(kk[good], vv[good], 1)[0]) if good.sum() >= 2 and kk[good].std() > 0 else float("nan")
    # k^2 fit of kick-associated loss (loss ~ c*k^2 through origin): c = sum(loss*k^2)/sum(k^4)
    ks = np.array([bs[n]["k"] for n in bs if n != 0]); ls = np.array([rec["kick_loss"][n] for n in bs if n != 0])
    rec["loss_per_k2"] = float(np.sum(ls * ks ** 2) / np.sum(ks ** 4)) if ks.size and np.sum(ks ** 4) > 0 else float("nan")
    nmax = max(rec["kick_loss"]) if rec["kick_loss"] else None
    rec["kick_loss_max_n"] = float(rec["kick_loss"][nmax]) if nmax is not None else float("nan")
    coh = all(bs[n]["n_fin"] >= 1 for n in bs)
    r2ok = all(b["r2"] > 0.9 for b in rec["boosts"] if n_nonzero(b) and np.isfinite(b["r2"]))
    rec["clean_transport"] = bool(rec["kick_loss_max_n"] < 0.06 and r2ok and coh and abs(rec.get("mobility_mu", 0)) > 1e-4)
    return rec


def n_nonzero(b):
    return b["n"] != 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", choices=["on", "off", "both"], default="both")
    ap.add_argument("--cands", default="1.0:0.15,0.5:0.15")
    ap.add_argument("--kicks", default="0,1,2,3")
    ap.add_argument("--N", type=int, default=96); ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--Tconfirm", type=int, default=12000); ap.add_argument("--Tboost", type=int, default=6000)
    ap.add_argument("--dtchunk", type=int, default=1000); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cands = [tuple(float(y) for y in c.split(":")) for c in a.cands.split(",")]
    kicks = [int(x) for x in a.kicks.split(",")]
    geoms = [True, False] if a.geom == "both" else [a.geom == "on"]
    out = a.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C2_2_LOSS_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    Xax = _axis_grid(a.N)
    print(f"=== C2.2 LOSS-SOURCE | N={a.N} dt={a.dt} geom={a.geom} cands={cands} kicks={kicks} "
          f"Tc={a.Tconfirm} Tb={a.Tboost} | out={out} ===", flush=True)
    results = []
    for geom_on in geoms:
        ops = _ops(geom_on, a.N, a.dt)
        tag = "GEOM_ON" if geom_on else "GEOM_OFF(pure-NLS,a_coupling=0)"
        for (A, sig) in cands:
            t0 = time.time()
            settled, verdict, traj = confirm(A, sig, ops, a.N, a.dt, a.Tconfirm, a.dtchunk)
            tr = traj[-1] if traj else {}
            print(f"[{tag}] confirm A={A} sig={sig} -> {verdict} mass_ret={tr.get('mass_ret')} "
                  f"occ={tr.get('occ_ratio')} n={tr.get('n')} ({(time.time()-t0)/60:.1f}m)", flush=True)
            rec = {"geom_on": geom_on, "A": A, "sigma": sig, "N": a.N, "dt": a.dt,
                   "confirm": verdict, "traj_fin": tr, "boosts": []}
            if verdict == "STABLE_SOLITON":
                for nk in kicks:
                    b = boost(settled, ops, Xax, nk, a.N, a.dt, a.Tboost, a.dtchunk)
                    rec["boosts"].append(b)
                    vk = b["v"] / b["k"] if b["k"] else 0.0
                    print(f"   [boost] n={nk} k={b['k']:.3f} v={b['v']:+.5f} (v/k={vk:+.4f}) "
                          f"disp={b['disp']:+.4f}box r2={b['r2']:.2f} mass={b['mass_ret']:.3f} n={b['n_fin']}", flush=True)
                analyse(rec)
                kl = {k: round(v, 3) for k, v in rec.get("kick_loss", {}).items()}
                print(f"   => mu={_f(rec.get('mobility_mu'))} kick_loss={kl} "
                      f"loss/k2={_f(rec.get('loss_per_k2'))} clean_transport={rec.get('clean_transport')}", flush=True)
            results.append(rec)
            json.dump(results, open(os.path.join(out, "loss_source.json"), "w"), indent=2, default=float)

    # ---- classification ----
    on = {(r["A"], r["sigma"]): r for r in results if r["geom_on"] and r["confirm"] == "STABLE_SOLITON"}
    off = {(r["A"], r["sigma"]): r for r in results if not r["geom_on"]}
    verdict, notes = classify(on, off, results)
    print(f"\n=== {verdict} ===", flush=True)
    for n in notes:
        print(f"    - {n}", flush=True)
    json.dump({"verdict": verdict, "notes": notes, "results": results},
              open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"C2_2_DONE {out}", flush=True)


def classify(on, off, results):
    notes = []
    off_stable = {k: r for k, r in off.items() if r["confirm"] == "STABLE_SOLITON"}
    off_disp = {k: r for k, r in off.items() if r["confirm"] != "STABLE_SOLITON"}
    if not on and not off_stable:
        return "C2_NUMERICAL_REGIME_UNCLEAR", ["no stable soliton confirmed in either substrate"]
    if off_disp and not off_stable:
        notes.append(f"geometry-off: soliton did NOT localize for {list(off_disp.keys())} "
                     f"({[r['confirm'] for r in off_disp.values()]}) -> geometry STABILIZES the soliton; "
                     f"loss-source entangled with existence")
        return "C2_NUMERICAL_REGIME_UNCLEAR", notes
    # compare kick-associated loss on vs off for shared candidates
    improved = []
    for key in off_stable:
        r_off = off_stable[key]; r_on = on.get(key)
        lo = r_off.get("kick_loss_max_n", np.nan)
        notes.append(f"{key} geom-OFF kick_loss_max={lo:.3f} mu={r_off.get('mobility_mu'):+.4f} clean={r_off.get('clean_transport')}")
        if r_on is not None:
            hi = r_on.get("kick_loss_max_n", np.nan)
            notes.append(f"{key} geom-ON  kick_loss_max={hi:.3f} mu={r_on.get('mobility_mu'):+.4f}")
            if np.isfinite(lo) and np.isfinite(hi) and lo < 0.5 * hi:
                improved.append(key)
    off_clean = [k for k, r in off_stable.items() if r.get("clean_transport")]
    if off_clean:
        return "C2_GEOMETRY_BREAKS_CLEAN_TRANSPORT", notes + [
            f"pure-NLS clean transport SUPPORTED for {off_clean}", "-> C2_PURE_NLS_CLEAN_TRANSPORT_SUPPORTED"]
    if improved:
        return "C2_GEOMETRY_BREAKS_CLEAN_TRANSPORT", notes + [
            f"geometry-off HALVES kick-loss for {improved} (not fully clean) -> geometry is a major loss source"]
    # geometry off didn't help -> intrinsic (or boost/numerical, see resolution runs)
    return "C2_INTRINSIC_LOSSY_TRANSPORT", notes + [
        "geometry-off did NOT reduce kick-associated loss -> loss not geometry-sourced (check boost/resolution runs)"]


if __name__ == "__main__":
    main()
