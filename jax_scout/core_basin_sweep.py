"""
GL PARAMETER BASIN SWEEP — map where the bare S-NCGL field self-sustains a ROTATIONAL CORE vs
spins down (no A-field, no coupling, no solver mod). Follows the hi-fi finding
(HIFI_CONTINUATION_STRUCTURES_REAL_VORTEX_SUSTAIN_VS_DECAY): stability is an energy gain/dissipation
balance of a rotational core set by the GL params, NOT a missing topological-spin DOF.

Engine: physics.sweep_probe (vmap, batched, bare S-NCGL incl. scalar Omega^2 geometry; FAST, no
per-step A update). Shared multiseed IC. For each config: energy(t) trajectory + psi at mid (T/2)
and final (T). Cheap gate on energy; for viable configs, host-side rotational-core metrics
(dominant-node core density + tangential v_t + radial v_r via rt_decomp) at mid vs final.

Classes:
  SUSTAIN        : er_final in [0.5,2], core density retained (>=0.7x mid), v_t retained (>=0.5x mid),
                   1-8 nodes -> self-sustaining rotational core (the basin we want).
  SPIN_DOWN      : viable/finite but core or circulation fades (the 'unstable' dissipative decay).
  BLOWUP         : non-finite or er>3 (energy runaway).
  COLLAPSE       : er_final<0.3 (dissipative collapse to ~vacuum).
  FRAGMENT       : >8 nodes (space-filling).
Objective: NOT routing/bridge/Payan. Just rotational-core persistence + bounded energy.

Time-boxed; incremental CSV + periodic summary.json. Seeds feb56dc7/b31c0396 as references.
WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/core_basin_sweep.py --hours 6
"""
import os, sys, csv, glob, json, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td, geometry_diag as gd
from jax_scout.afield_current_coupled import multiseed_ic, L as L_, dt as DT
from jax_scout.payan_hifi_continuation import rt_decomp

order = physics.SWEEP_PARAM_ORDER     # [D, eta, rho_vac, omega0, a_coupling, s, f, a]
SEED = 20260619
REF_HASHES = {"feb56dc7": "ref_stable", "b31c0396": "ref_unstable"}


def F(r, k):
    try: return float(r[k])
    except: return float("nan")


def load_refs():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "SUBSTRATE_HUNT_2026*")))[-1]
    rows = {r.get("hash"): r for r in csv.DictReader(open(os.path.join(d, "all_evals.csv")))}
    out = []
    for h, tag in REF_HASHES.items():
        if h in rows:
            out.append(([F(rows[h], k) for k in order], f"{tag}_{h}"))
    return out


def core_metrics(psi, par, N, dx):
    nodes = sorted(td.detect_nodes(psi, dx), key=lambda n: -n["E"])
    if not nodes:
        return {"n_nodes": 0, "core_rho": 0.0, "v_r": 0.0, "v_t": 0.0}
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes]) ** (1 / 3))))
    c = np.round(nodes[0]["centroid"]).astype(int) % N
    dec = rt_decomp(psi, par, c, dx, N, max(1, node_r - 1), node_r + 2)
    return {"n_nodes": len(nodes), **dec}


def classify(finite, er, mid, fin):
    er_max = float(np.max(er)); er_fin = float(er[-1])
    base = {"er_fin": er_fin, "er_max": er_max}
    if not finite or not np.isfinite(er_max) or er_max > 3.0:
        return "BLOWUP", base
    if er_fin < 0.3:
        return "COLLAPSE", base
    if mid is None or fin is None:
        return "VIABLE_NO_NODES", base
    base.update({"n_mid": mid["n_nodes"], "n_fin": fin["n_nodes"],
                 "core_mid": mid["core_rho"], "core_fin": fin["core_rho"],
                 "vt_mid": mid["v_t"], "vt_fin": fin["v_t"], "vr_fin": fin["v_r"]})
    if fin["n_nodes"] > 8 or mid["n_nodes"] > 8:
        return "FRAGMENT", base
    if fin["n_nodes"] < 1:
        return "COLLAPSE", base
    cd_ratio = fin["core_rho"] / (mid["core_rho"] + 1e-30)
    vt_ratio = fin["v_t"] / (mid["v_t"] + 1e-30)
    base.update({"cd_ratio": cd_ratio, "vt_ratio": vt_ratio,
                 "swirl_fin": fin["v_t"] / (abs(fin["v_r"]) + fin["v_t"] + 1e-30)})
    if 0.5 <= er_fin <= 2.0 and cd_ratio >= 0.7 and vt_ratio >= 0.5 and 1 <= fin["n_nodes"] <= 8:
        return "SUSTAIN", base
    return "SPIN_DOWN", base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--T", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--bounds-file", default=os.path.join(ROOT, "jax_scout", "gain_bounds.json"))
    ap.add_argument("--calibrate", action="store_true", help="one small batch, no timebox")
    args = ap.parse_args()
    N, T, B = args.N, args.T, args.batch
    dx = L_ / N
    bounds = json.load(open(args.bounds_file))
    lo = np.array([bounds[k][0] for k in order]); hi = np.array([bounds[k][1] for k in order])
    rng = np.random.default_rng(SEED)
    psi0 = jnp.asarray(multiseed_ic(N, SEED)); ic_e = float(np.sum(np.abs(np.asarray(psi0)) ** 2)) + 1e-30

    tag = "CORE_BASIN_CALIB" if args.calibrate else "CORE_BASIN"
    outdir = os.path.join(ROOT, "sweep_runs", f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    cols = ["idx", "label", "klass", "er_fin", "er_max", "n_mid", "n_fin", "core_mid", "core_fin",
            "vt_mid", "vt_fin", "vr_fin", "cd_ratio", "vt_ratio", "swirl_fin", *order]
    log = open(os.path.join(outdir, "all_evals.csv"), "w", newline="")
    cw = csv.DictWriter(log, fieldnames=cols, extrasaction="ignore"); cw.writeheader()

    refs = load_refs()
    deadline = float("inf") if args.calibrate else time.time() + args.hours * 3600
    max_batches = 1 if args.calibrate else 10 ** 9
    t0 = time.time(); idx = 0; nb = 0; counts = {}
    print(f"=== CORE BASIN SWEEP N={N} T={T} batch={B} (bare S-NCGL, no A) ===", flush=True)
    while time.time() < deadline and nb < max_batches:
        nb += 1
        rows = [np.array(p) for p, _ in refs] if (nb == 1) else []
        labels = [lab for _, lab in refs] if (nb == 1) else []
        while len(rows) < B:
            rows.append(lo + rng.random(len(order)) * (hi - lo)); labels.append("rand")
        pm = jnp.asarray(np.stack(rows))
        tb = time.time()
        psi_mid, psi_fin, energy, max_amp, finite = physics.sweep_probe(pm, psi0, N, L_, DT, T, jnp.float64, jnp.complex128)
        energy = np.asarray(energy); finite = np.asarray(finite)
        psi_mid = np.asarray(psi_mid); psi_fin = np.asarray(psi_fin)
        for b in range(B):
            er = energy[b] / ic_e
            par = {k: float(rows[b][i]) for i, k in enumerate(order)}
            viable = bool(finite[b]) and np.isfinite(er).all() and float(np.max(er)) <= 3.0 and float(er[-1]) >= 0.3
            mid = core_metrics(psi_mid[b], par, N, dx) if viable else None
            fin = core_metrics(psi_fin[b], par, N, dx) if viable else None
            kl, m = classify(bool(finite[b]), er, mid, fin)
            counts[kl] = counts.get(kl, 0) + 1
            row = {"idx": idx, "label": labels[b], "klass": kl, **m,
                   **{k: round(float(rows[b][i]), 4) for i, k in enumerate(order)}}
            cw.writerow(row); idx += 1
        log.flush()
        el = (time.time() - t0) / 3600
        print(f"[batch {nb} t={el:.2f}h] evals={idx} SUSTAIN={counts.get('SUSTAIN',0)} "
              f"SPIN_DOWN={counts.get('SPIN_DOWN',0)} BLOWUP={counts.get('BLOWUP',0)} "
              f"COLLAPSE={counts.get('COLLAPSE',0)} FRAG={counts.get('FRAGMENT',0)} "
              f"({time.time()-tb:.0f}s/batch)", flush=True)
        json.dump({"N": N, "T": T, "batch": B, "n_eval": idx, "elapsed_h": el, "counts": counts,
                   "order": order}, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    log.close()
    print(f"\n=== DONE: {idx} evals in {(time.time()-t0)/3600:.2f}h | counts={counts} ===")
    print(f"SUSTAIN (self-sustaining rotational cores): {counts.get('SUSTAIN',0)}/{idx}")
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()
