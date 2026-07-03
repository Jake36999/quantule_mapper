"""Adiabatic drag — RELATIONAL mobility probe for the a* dissipative attractor (default-off physics variant).

Follow-up to the kick/inertia null (docs/PHASE_C_KICK_INERTIA_AND_OPERATOR_FINDING.md): the substrate is a real
dissipative Ginzburg-Landau system with NO inertial channel, so Galilean coasting is structurally absent. The
right mobility question for a dissipative attractor is "does it track a slowly-moving energetic preference?".

Coupling (default-off): a weak localized REAL well added to the real-space RHS, N += V0 * G(x - x_c(t)) * psi,
G Gaussian width w. REAL -> keeps the substrate dissipative (relational mobility, NOT inertia).
SIGN CONVENTION: +V0 adds a POSITIVE real term to the RHS on psi => LOCAL LINEAR GAIN = local loss reduction
                 = a "gain/comfort preference" the attractor should favour. (V0<0 would be a loss/repeller.)
V0=0 / drag_field=None -> exact baseline (numerically asserted: BASELINE_REPRODUCED). See
docs/PHASE_C_ADIABATIC_DRAG_DESIGN.md. NOT inertial-matter; label relational-mobility / adiabatic-tracking.

Static battery guards (per review): (1) baseline None==zeros numerically; (2) DRAG vs NUCLEATION — a bias only
counts if the EXISTING structure migrates (node count preserved, origin region depletes), not a new blob grown
at the well (node count rises / origin retained); (3) track global density COM AND node centroids/count;
(4) require morphology coherent; (5) two V0 (0.025, 0.05) before declaring no coupling.

Modes: static (default, gates moving) | moving (gated). Segregated output. Geometry frozen e8d6a78ea.
WSL2 jax venv:  python jax_scout/feb_adiabatic_drag.py [--mode static|moving] [--w ...] [--out DIR]
"""
import os, sys, csv, json, time, argparse
from functools import partial
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
L, DT = css.L_, css.DT
DX = L / N
A_FACTOR = 1.15
DEFAULT_STATE = os.path.join(ROOT, "sweep_runs",
                             "FEB_GAIN_LADDER_LONGT_T72000_20260701_175708", "a1.15_ladder_T72000_probe.npz")
BIAS_MIN = 0.15            # min COM shift/offset to count as coupling
ORIGIN_DEPLETE_MIN = 0.08  # min relative drop of origin-region mass for a genuine drag (structure left)
_x = np.linspace(-L / 2, L / 2, N, endpoint=False)
_X = np.meshgrid(_x, _x, _x, indexing="ij")


def circ_com(rho):
    out = []
    for ax in range(3):
        th = 2 * np.pi * _X[ax] / L
        out.append(L * np.arctan2(float(np.sum(rho * np.sin(th))), float(np.sum(rho * np.cos(th)))) / (2 * np.pi))
    return np.array(out)


def node_info(psi):
    """(#3) individual node centroids (physical) + count. detect_nodes centroid is grid-index -> physical."""
    nodes = td.detect_nodes(np.asarray(psi), DX)
    cents = np.array([np.asarray(nd["centroid"], float) * DX - L / 2 for nd in nodes]) if nodes else np.zeros((0, 3))
    ncom = cents.mean(axis=0) if len(cents) else np.array([np.nan] * 3)
    return len(nodes), cents, ncom


def mass_frac_near(rho, center, radius):
    def pd(coord, c): return (coord - c + L / 2) % L - L / 2
    r2 = pd(_X[0], center[0]) ** 2 + pd(_X[1], center[1]) ** 2 + pd(_X[2], center[2]) ** 2
    return float(np.sum(rho[r2 < radius ** 2])) / float(np.sum(rho))


def gaussian_well(xc, w):
    def pd(coord, c): return (coord - c + L / 2) % L - L / 2
    r2 = pd(_X[0], xc[0]) ** 2 + pd(_X[1], xc[1]) ** 2 + pd(_X[2], xc[2]) ** 2
    return np.exp(-r2 / (2.0 * w ** 2))


@partial(jax.jit, static_argnames=("n",))
def _evolve_baseline(psi_k, ops, n):
    def body(pk, _): return physics.step(pk, ops), None
    pk, _ = jax.lax.scan(body, psi_k, None, length=n); return pk


@partial(jax.jit, static_argnames=("n",))
def _evolve_drag(psi_k, ops, drag, n):
    def body(pk, _): return physics.step(pk, ops, drag_field=drag), None
    pk, _ = jax.lax.scan(body, psi_k, None, length=n); return pk


def track(psi0, ops, well_center_fn, V0, w, T, dt_chunk, well_final, com0):
    psi_k = physics.initial_psi_k(jnp.asarray(psi0), ops)
    rho0 = np.abs(np.asarray(psi0)) ** 2
    M0 = float(np.sum(rho0)); p0 = float(np.max(rho0))
    n0, cents0, ncom0 = node_info(psi0)
    origin0 = mass_frac_near(rho0, com0, 1.5 * w); well0 = mass_frac_near(rho0, well_final, 1.5 * w)
    com_tr, node_tr, wellx, tphys, massr, peakr = [], [], [], [], [], []
    n_chunks = T // dt_chunk; node_every = max(1, n_chunks // 12)
    psi = np.asarray(psi0)
    for c in range(n_chunks):
        xc = well_center_fn((c * dt_chunk) * DT)
        if V0 == 0.0:
            psi_k = _evolve_baseline(psi_k, ops, dt_chunk)
        else:
            psi_k = _evolve_drag(psi_k, ops, jnp.asarray((V0 * gaussian_well(xc, w)).astype(np.float64)), dt_chunk)
        psi = np.asarray(jnp.fft.ifftn(psi_k)); rho = np.abs(psi) ** 2
        com_tr.append(circ_com(rho)); wellx.append(float(xc[0])); tphys.append((c + 1) * dt_chunk * DT)
        massr.append(float(np.sum(rho)) / M0); peakr.append(float(np.max(rho)) / p0)
        if c % node_every == 0:
            node_tr.append(node_info(psi)[2])
    n1, cents1, ncom1 = node_info(psi); rho = np.abs(psi) ** 2
    return {"com0": com0, "com_fin": np.array(com_tr[-1]), "com_traj": np.array(com_tr),
            "node_com0": ncom0, "node_com_fin": ncom1, "node_com_traj": np.array(node_tr),
            "n_start": n0, "n_end": n1, "cents_start": cents0, "cents_end": cents1,
            "origin_mass0": origin0, "origin_mass_fin": mass_frac_near(rho, com0, 1.5 * w),
            "well_mass0": well0, "well_mass_fin": mass_frac_near(rho, well_final, 1.5 * w),
            "well_x": np.array(wellx), "t": np.array(tphys),
            "mass_r": np.array(massr), "peak_r": np.array(peakr)}


def baseline_reproduced(ops, psi):
    pk = physics.initial_psi_k(jnp.asarray(psi), ops)
    a = _evolve_baseline(pk, ops, 20)
    b = _evolve_drag(pk, ops, jnp.zeros((N, N, N), jnp.float64), 20)
    d = float(jnp.max(jnp.abs(a - b))); return d, d < 1e-10


def classify_offset(r, offset):
    """Fine taxonomy for an offset cell: which of the mobility responses occurred.
    NULL / WEAK_LOCAL_ACCRETION / RELOCATION_BIAS / NEW_BLOB_NUCLEATION / MORPHOLOGY_BREAK.
    Only RELOCATION_BIAS (existing structure migrates, origin depletes, no new node) = relational mobility."""
    dcom = float(r["com_fin"][0] - r["com0"][0]); bias = dcom / offset
    ncom = float(r["node_com_fin"][0] - r["node_com0"][0]) if np.isfinite(r["node_com_fin"][0]) else np.nan
    origin_drop = (r["origin_mass0"] - r["origin_mass_fin"]) / max(r["origin_mass0"], 1e-9)
    well_gain = float(r["well_mass_fin"] - r["well_mass0"]); dn = r["n_end"] - r["n_start"]
    if dn > 0 or (well_gain > 0.15 and origin_drop < ORIGIN_DEPLETE_MIN):
        label = "NEW_BLOB_NUCLEATION"
    elif dn < 0:
        label = "MORPHOLOGY_BREAK"
    elif bias > BIAS_MIN and origin_drop > ORIGIN_DEPLETE_MIN:
        label = "RELOCATION_BIAS"
    elif well_gain > 0.02:
        label = "WEAK_LOCAL_ACCRETION"
    else:
        label = "NULL"
    return {"bias_toward_well": round(bias, 4), "node_com_shift_x": round(ncom, 4) if ncom == ncom else None,
            "origin_mass_drop": round(float(origin_drop), 4), "well_mass_gain": round(well_gain, 4),
            "coherent": (dn == 0), "cell_label": label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "moving"], default="static")
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--offset", type=float, default=1.8)
    ap.add_argument("--V0s", default="0.025,0.05", help="offset-well V0 values (>=2 before declaring null)")
    ap.add_argument("--vwell", type=float, default=0.02)
    ap.add_argument("--V0move", type=float, default=0.05)
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--dtchunk", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_ADIABATIC_DRAG_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)

    psi_settle = np.load(args.state)["psi_fin"].astype(np.complex128)
    pp = dict(css.FEB); pp["param_a"] = float(css.FEB["param_a"]) * A_FACTOR
    ops = physics.build_operators(N, L, DT, pp)
    com0 = circ_com(np.abs(psi_settle) ** 2); n0 = node_info(psi_settle)[0]
    d, ok = baseline_reproduced(ops, psi_settle)
    print(f"=== FEB ADIABATIC DRAG [{args.mode}] a×{A_FACTOR} | w={args.w} offset={args.offset} | out={out} ===", flush=True)
    print(f"    SIGN: +V0 = local loss reduction / gain preference.  BASELINE_REPRODUCED={ok} "
          f"(max|None-zeros|={d:.2e})  COM0={np.round(com0,3)} nodes={n0}", flush=True)
    if not ok:
        print("    ABORT: default-off contract violated (None != zeros).", flush=True); return

    rows = []
    if args.mode == "static":
        V0s = [float(v) for v in args.V0s.split(",")]
        plan = [("baseline_None", 0.0, com0.copy(), com0.copy())]
        plan.append(("well_on_centre", V0s[-1], com0.copy(), com0.copy()))
        wc = com0 + np.array([args.offset, 0.0, 0.0])
        for v in V0s:
            plan.append((f"well_offset_V{v}", v, wc.copy(), wc.copy()))
        for label, V0, xc, well_final in plan:
            t0 = time.time()
            r = track(psi_settle, ops, (lambda c: (lambda t: c))(xc), V0, args.w, args.T, args.dtchunk, well_final, com0)
            row = {"cell": label, "V0": V0, "w": args.w, "com_shift_x": round(float(r["com_fin"][0] - r["com0"][0]), 4),
                   "n_start": r["n_start"], "n_end": r["n_end"], "origin_mass0": round(r["origin_mass0"], 4),
                   "origin_mass_fin": round(r["origin_mass_fin"], 4), "well_mass0": round(r["well_mass0"], 4),
                   "well_mass_fin": round(r["well_mass_fin"], 4), "mass_r_fin": round(float(r["mass_r"][-1]), 4),
                   "peak_r_fin": round(float(r["peak_r"][-1]), 4), "wallclock_min": round((time.time() - t0) / 60, 1)}
            if "offset" in label:
                row.update(classify_offset(r, args.offset))
            rows.append(row)
            np.savez_compressed(os.path.join(out, f"{label}_traj.npz"), com=r["com_traj"],
                                node_com=r["node_com_traj"], well_x=r["well_x"], t=r["t"])
            extra = f" {row.get('cell_label','')} bias={row.get('bias_toward_well','-')}" if "offset" in label else ""
            print(f"  {label}: V0={V0} com_shift_x={row['com_shift_x']:+.3f} nodes {r['n_start']}->{r['n_end']} "
                  f"origin_mass {r['origin_mass0']:.3f}->{r['origin_mass_fin']:.3f} well_mass {r['well_mass0']:.3f}->"
                  f"{r['well_mass_fin']:.3f} mass={row['mass_r_fin']:.3f}{extra} ({row['wallclock_min']}m)", flush=True)
        offs = [r for r in rows if "offset" in r["cell"]]
        labs = [r.get("cell_label") for r in offs]
        if "RELOCATION_BIAS" in labs:
            verdict = "STATIC_WELL_BIAS_SUPPORTED"
        elif "NEW_BLOB_NUCLEATION" in labs or "MORPHOLOGY_BREAK" in labs:
            verdict = ("STATIC_WELL_ACCRETION_THEN_NUCLEATION" if "WEAK_LOCAL_ACCRETION" in labs
                       else "STATIC_WELL_NUCLEATION_NO_RELOCATION")
        elif "WEAK_LOCAL_ACCRETION" in labs:
            verdict = "STATIC_WELL_ACCRETION_ONLY_NO_RELOCATION"
        else:
            verdict = "STATIC_WELL_NO_COUPLING"
        print(f"=== STATIC VERDICT: {verdict} (offset cells: "
              f"{', '.join(r['cell']+'='+str(r.get('cell_label')) for r in offs)}) ===", flush=True)
        summ_verdict = verdict
    else:
        t0 = time.time(); x0 = com0.copy(); wf = x0 + np.array([args.vwell * (args.T * DT), 0.0, 0.0])
        r = track(psi_settle, ops, lambda t: x0 + np.array([args.vwell * t, 0.0, 0.0]),
                  args.V0move, args.w, args.T, args.dtchunk, wf, com0)
        lag = r["well_x"] - r["com_traj"][:, 0]
        rows.append({"cell": "moving", "V0": args.V0move, "vwell": args.vwell,
                     "com_travel_x": round(float(r["com_traj"][-1, 0] - r["com0"][0]), 4),
                     "well_travel_x": round(float(r["well_x"][-1] - r["well_x"][0]), 4),
                     "final_lag": round(float(lag[-1]), 4), "n_start": r["n_start"], "n_end": r["n_end"],
                     "mass_r_fin": round(float(r["mass_r"][-1]), 4), "wallclock_min": round((time.time() - t0) / 60, 1)})
        np.savez_compressed(os.path.join(out, "moving_traj.npz"), com=r["com_traj"], well_x=r["well_x"], t=r["t"])
        print(f"  moving: {rows[-1]}", flush=True); summ_verdict = "MOVING_DONE"

    with open(os.path.join(out, "feb_adiabatic_drag_results.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r})); wr.writeheader(); wr.writerows(rows)
    json.dump({"mode": args.mode, "verdict": summ_verdict, "sign": "+V0 = local loss reduction / gain preference",
               "a_factor": A_FACTOR, "param_a": pp["param_a"], "baseline_reproduced": ok, "baseline_maxdiff": d,
               "N": N, "L": L, "dt": DT, "w": args.w, "offset": args.offset, "rows": rows},
              open(os.path.join(out, "feb_adiabatic_drag_summary.json"), "w"), indent=2, default=float)
    print(f"=== DONE {out} ===", flush=True)


if __name__ == "__main__":
    main()
