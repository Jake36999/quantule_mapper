"""Phase D / C2.5 — coefficient-family scout: does ANY (a,s,f) family admit a TRUE moving soliton?
The C2 transport null (C2.3/C2.4: no stationary branch for feb/a*, quasi-soliton pinned by flow-through) is
COEFFICIENT-SPECIFIC. This scout searches (a,s,f) space for families with a true localized stationary branch, then
gates them for stability and genuine transport. Theory guide: feb has s=+0.013 (FOCUSING quintic, anomalous); the
classic stable-3D-soliton families are cubic(+)-quintic(-) (defocusing quintic saturation) -> prioritize s<0.

Staged, cheap-first (all N=48, pure NLS geometry-off, checkpointed):
  P0  Petviashvili existence scan: for each family compute g_max = max g(rho) (binding ceiling), scan
      mu in fracs*g_max with narrow seeds -> TRUE_BRANCH / UNIFORM_ONLY / NO_FIXED_POINT. Seconds per case.
  P1  real-time stability of TRUE_BRANCH profiles (T=3, dt=1e-3): mass/amp/occ hold -> STABLE.
  P2  local-boost transport gate (C2.4 protocol, zero winding, k_loc=0.628): v/(2*D*k_loc) -> GALILEAN (>0.5) /
      PARTIAL (>10x the C2.4 drag baseline 0.0054) / PINNED. Promotion to N=96+dt-ladder (P3) is manual.
feb/a* itself runs as a CONTROL (expected NO_FIXED_POINT, validating the classifier against C2.3).

Mirror only; conservative branch via build_operators; Phase C untouched; no clipping; no matter claims.

  wsl:  python jax_scout/phase_d_c2_5_family_scout.py [--tier 1|2] [--out DIR]
"""
import os, sys, json, csv, time, argparse, itertools
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
from jax_scout.phase_d_c2_2_loss_source import _axis_grid, _f
from jax_scout.phase_d_c2_3_exact_soliton import petviashvili, stationarity
from jax_scout.phase_d_c2_4_local_boost import run_case

L = css.L_
N, DT = 48, 0.001
FEB_ASF = (round(float(css.FEB["param_a"]) * 1.15, 4), float(css.FEB["param_s"]), float(css.FEB["param_f"]))

# tier grids: (a, s, f). Tier 1 = 4x4x3 coarse; tier 2 adds finer s<0 and a values.
GRID_T1 = {"a": [0.30, FEB_ASF[0], 0.80, 1.20],
           "s": [-0.50, -0.20, -0.05, FEB_ASF[1]],
           "f": [FEB_ASF[2], -0.10, 0.0]}
GRID_T2 = {"a": [0.30, 0.45, FEB_ASF[0], 0.80, 1.00, 1.20],
           "s": [-1.00, -0.50, -0.30, -0.20, -0.10, -0.05, FEB_ASF[1]],
           "f": [FEB_ASF[2], -0.25, -0.10, 0.0]}
MU_FRACS = [0.25, 0.5, 0.75]
SEEDS = [(1.0, 0.08), (1.5, 0.06)]
COLS = ["a", "s", "f", "D", "g_max", "rho_star", "mu", "ell", "seedA", "seedsig", "p0", "residual", "amp", "occ",
        "mass", "p1", "p1_mass_ret", "p1_amp_ret", "p1_occ_ratio", "p2", "p2_vfrac", "p2_v", "p2_mass", "p2_corefrac"]


def g_of(a, s, f):
    rho = np.linspace(1e-4, 6.0, 6000)
    g = a * rho + s * rho ** 2 + f * rho ** 3
    i = int(np.argmax(g))
    unbounded = bool(i >= len(rho) - 2)
    return float(g[i]), float(rho[i]), unbounded


def _ops_family(a, s, f, D=None):
    """TRUE pure NLS (param_geom_off — a_coupling=0 alone is defeated by the soft-clip squash; see physics.Ops.geom_fac)."""
    p = {**css.FEB, "param_a": a, "param_s": s, "param_f": f,
         "kinetic_mode": "conservative", "param_a_coupling": 0.0, "param_geom_off": True}
    if D is not None:
        p["param_D"] = D
    return physics.build_operators(N, L, DT, p)


def classify_p0(prof):
    if prof is None:
        return "NO_FIXED_POINT"
    loc = prof["occ"] < 0.5 and 0.05 <= prof["amp"] <= 5.0
    if prof["residual"] < 1e-8 and loc and abs(prof["S_final"] - 1.0) < 1e-4:
        return "TRUE_BRANCH"
    if prof["residual"] < 1e-8 and prof["occ"] > 0.9:
        return "UNIFORM_ONLY"
    return "NO_FIXED_POINT"


def p1_stability(psi0, ops, Tphys=3.0, dt_chunk=500):
    M0 = float(np.sum(np.abs(psi0) ** 2)); amp0 = float(np.max(np.abs(psi0))); occ0 = occ(np.abs(psi0) ** 2)
    pk = physics.initial_psi_k(jnp.asarray(psi0), ops); cur = psi0
    for c in range(int(round(Tphys / DT)) // dt_chunk):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return "COLLAPSE", {}
    rho = np.abs(cur) ** 2
    m = {"mass_ret": float(rho.sum()) / M0, "amp_ret": float(np.max(np.abs(cur))) / (amp0 + 1e-30),
         "occ_ratio": occ(rho) / (occ0 + 1e-30)}
    ok = m["mass_ret"] > 0.90 and 0.6 <= m["amp_ret"] <= 1.6 and m["occ_ratio"] < 1.8
    return ("STABLE" if ok else "UNSTABLE"), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1); ap.add_argument("--out", default=None)
    ap.add_argument("--kloc", type=float, default=0.628); ap.add_argument("--Tboost", type=float, default=3.0)
    ap.add_argument("--families", default=None, help="override grid: 'a:s:f,a:s:f' (smoke/targeted)")
    ap.add_argument("--Ds", default=None, help="comma list of param_D values (default: feb 2.7329). "
                    "Soliton width ell=sqrt(D/mu) must fit the box (ell << L/4) — lower D opens the window.")
    a_ = ap.parse_args()
    Ds = [float(x) for x in a_.Ds.split(",")] if a_.Ds else [float(css.FEB["param_D"])]
    grid = GRID_T1 if a_.tier == 1 else GRID_T2
    out = a_.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C2_5_SCOUT_T{a_.tier}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True); csv_path = os.path.join(out, "family_scout.csv")
    if a_.families:
        fams = [tuple(float(y) for y in c.split(":")) for c in a_.families.split(",")]
    else:
        fams = [t for t in itertools.product(grid["a"], grid["s"], grid["f"])
                if not (t[1] >= 0 and t[2] >= 0)]                    # need saturation: s<0 or f<0
    Xax = _axis_grid(N)
    print(f"=== C2.5 FAMILY SCOUT tier{a_.tier} | {len(fams)} families | N={N} dt={DT} | feb/a*={FEB_ASF} (control) "
          f"| out={out} ===", flush=True)
    rows, hits = [], []
    t_start = time.time()
    for (Dv, (av, sv, fv)) in itertools.product(Ds, fams):
        g_max, rho_star, unb = g_of(av, sv, fv)
        tag = "FEB_CONTROL" if (round(av, 4), sv, fv) == FEB_ASF and abs(Dv - 2.7329) < 1e-6 else ""
        if g_max <= 0.05:
            rows.append({"a": av, "s": sv, "f": fv, "D": Dv, "g_max": g_max, "rho_star": rho_star, "p0": "NO_BINDING"})
            _dump(rows, csv_path); continue
        ops = _ops_family(av, sv, fv, Dv)
        best = None
        for frac in MU_FRACS:
            mu = frac * g_max
            for (A, sig) in SEEDS:
                psi_c, prof = petviashvili(A, sig, ops, N, mu)
                k = classify_p0(prof if psi_c is not None else None)
                rec = {"a": av, "s": sv, "f": fv, "D": Dv, "g_max": round(g_max, 4), "rho_star": round(rho_star, 3),
                       "mu": round(mu, 4), "ell": round(float(np.sqrt(Dv / mu)), 2), "seedA": A, "seedsig": sig, "p0": k}
                if psi_c is not None:
                    rec.update({"residual": prof["residual"], "amp": round(prof["amp"], 3),
                                "occ": round(prof["occ"], 4), "mass": round(prof["mass"], 1)})
                rows.append(rec)
                if k == "TRUE_BRANCH" and (best is None or prof["residual"] < best[1]["residual"]):
                    best = (psi_c, prof, rec)
                if k == "TRUE_BRANCH":
                    break                                             # one seed hit is enough per mu
        state = "TRUE_BRANCH" if best else ("checked")
        if best:
            psi_c, prof, rec = best
            p1, m1 = p1_stability(psi_c, ops)
            rec["p1"] = p1
            rec.update({f"p1_{k}": round(v, 4) for k, v in m1.items()})
            print(f"[fam a={av} s={sv} f={fv} D={Dv}]{tag} g_max={g_max:.3f} TRUE_BRANCH mu={rec['mu']} "
                  f"ell={rec['ell']} amp={rec['amp']} res={rec['residual']:.1e} -> P1 {p1} {m1}", flush=True)
            if p1 == "STABLE":
                T_steps = int(round(a_.Tboost / DT))
                r = run_case(psi_c, ops, Xax, a_.kloc, N, DT, T_steps, 500)
                if not r.get("collapsed"):
                    vf = r["v_frac_of_galilean"]
                    rec["p2"] = ("GALILEAN" if vf > 0.5 else "PARTIAL" if vf > 0.054 else "PINNED")
                    rec.update({"p2_vfrac": round(vf, 4), "p2_v": round(r["v"], 4),
                                "p2_mass": round(r["mass_ret"], 4), "p2_corefrac": round(r["core_frac_end"], 3)})
                else:
                    rec["p2"] = "COLLAPSED"
                print(f"    -> P2 {rec['p2']} vfrac={rec.get('p2_vfrac')} v={rec.get('p2_v')} "
                      f"mass={rec.get('p2_mass')}", flush=True)
                hits.append(rec)
        else:
            p0s = {r.get("p0") for r in rows
                   if r.get("a") == av and r.get("s") == sv and r.get("f") == fv and r.get("D") == Dv}
            print(f"[fam a={av} s={sv} f={fv} D={Dv}]{tag} g_max={g_max:.3f} -> {sorted(p0s)}", flush=True)
        _dump(rows, csv_path)
        json.dump({"hits": hits, "elapsed_min": round((time.time() - t_start) / 60, 1)},
                  open(os.path.join(out, "hits.json"), "w"), indent=2, default=float)
    galilean = [h for h in hits if h.get("p2") == "GALILEAN"]
    partial = [h for h in hits if h.get("p2") == "PARTIAL"]
    verdict = ("C2_5_MOVING_SOLITON_FAMILY_FOUND" if galilean else
               "C2_5_PARTIAL_TRANSPORT_FAMILIES" if partial else
               "C2_5_TRUE_BRANCHES_NO_TRANSPORT" if hits else
               "C2_5_NO_TRUE_BRANCH_IN_GRID")
    print(f"\n=== {verdict} | {len(hits)} stable true-branch families, {len(galilean)} GALILEAN, "
          f"{len(partial)} PARTIAL | {round((time.time()-t_start)/60,1)}m ===", flush=True)
    json.dump({"verdict": verdict, "n_families": len(fams), "hits": hits, "galilean": galilean, "partial": partial},
              open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"C2_5_DONE {out}", flush=True)


def _dump(rows, csv_path):
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
