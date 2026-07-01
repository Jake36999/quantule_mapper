"""
CHIRAL-PAIR VISUAL CAPTURE (WSL/JAX) — qualitative support for interpreting the 90% anti-alignment.
NO solver modification, NO coupling. Settles 1 stable + 1 unstable strong-bridge substrate and saves
psi snapshots at several time slices + node/bridge metadata for rendering on Windows (matplotlib/
pyvista). Also writes rho_history.h5 for the existing visual_analysis_suite (3D isosurface).

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/payan_chiral_capture.py
"""
import os, sys, csv, glob, json
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import transfer_diag as td
from jax_scout import afield_current_coupled as cc
from jax_scout.afield_current_coupled import multiseed_ic, L
from jax_scout.afield_payan_diagnostic import payan_observables, order

SEED = 20260619
N = 48
SETTLE = 800
NSNAP = 8                       # snapshots at t = 0,100,...,800
SAVE_IDX = [2, 4, 6, 8]        # t ~ 200,400,600,800


def F(r, k):
    try: return float(r[k])
    except: return float("nan")


def pick(d):
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    stable = sorted([r for r in rows if r["reject"] == "" and 2 <= F(r, "n_s") <= 8],
                    key=lambda r: -F(r, "bridge_s"))[0]
    unstable = sorted([r for r in rows if r["reject"] == "energy_drift" and 0.5 <= F(r, "er_s") <= 2.0
                       and F(r, "n_s") >= 2 and not (0.5 <= F(r, "er_e") <= 2.0)],
                      key=lambda r: -F(r, "bridge_s"))[0]
    def pack(r):
        return ({k: F(r, k) for k in order}, F(r, "gamma_A"), F(r, "kappa"), F(r, "c_A"),
                r.get("hash", "?"), F(r, "bridge_s"))
    return pack(stable), pack(unstable)


def capture(par, g, kap, cA):
    snaps, _, fin = cc.capture_cc(par, multiseed_ic(N, SEED), g, N, SETTLE, NSNAP, kappa=kap, c_A=cA)
    if not fin:
        return None
    return np.asarray(snaps)   # [NSNAP+1, N,N,N] complex (index 0 = IC)


def metadata(psi_final, par):
    """node centroids (voxel), bridge pair, axis (unit voxel), per-node axial spin + sign."""
    ob = payan_observables(psi_final, par, N)
    dx = L / N
    nodes = sorted(td.detect_nodes(psi_final, dx), key=lambda n: -n["E"])
    cents = [(np.round(n["centroid"]).astype(int) % N).tolist() for n in nodes]
    md = {"n_nodes": len(nodes), "centroids_vox": cents,
          "node_r": int(max(2, round(np.mean([n["size"] for n in nodes]) ** (1 / 3))))}
    if ob:
        # recompute bridge pair centroids/axis for the renderer
        geo = td.geometry_fields(psi_final, par, dx); best, bp = -1.0, (0, 1)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                c = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
                if c > best:
                    best, bp = c, (i, j)
        ci = np.array(cents[bp[0]], float); cj = np.array(cents[bp[1]], float)
        disp = cj - ci; disp -= N * np.round(disp / N); axis = disp / (np.linalg.norm(disp) + 1e-30)
        md.update({"bridge_pair": [int(bp[0]), int(bp[1])], "bridge_axis": axis.tolist(),
                   "bridge_cond": ob["bridge_cond"], "s_i": ob["s_i"], "s_j": ob["s_j"],
                   "A": ob["A"], "aligned": ob["aligned"], "spin_mag": ob["spin_mag"]})
    return md


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "SUBSTRATE_HUNT_2026*")))[-1]
    outdir = os.path.join(d, "chiral_viz"); os.makedirs(outdir, exist_ok=True)
    (sp, sg, sk, sc, sh, sbr), (up, ug, uk, uc, uh, ubr) = pick(d)
    print(f"stable {sh} (bridge_s={sbr:.2f}) | unstable {uh} (bridge_s={ubr:.2f})")
    out = {}; meta = {}
    for tag, (par, g, kap, cA, h, br) in (("stable", (sp, sg, sk, sc, sh, sbr)),
                                          ("unstable", (up, ug, uk, uc, uh, ubr))):
        snaps = capture(par, g, kap, cA)
        if snaps is None:
            print(f"  [{tag} {h}] settle nonfinite — skipped"); continue
        frames = snaps[SAVE_IDX]                              # [4,N,N,N] complex
        out[f"psi_{tag}"] = frames.astype(np.complex64)
        out[f"rho_hist_{tag}"] = (np.abs(snaps) ** 2).astype(np.float32)   # all snapshots, for 3D suite on Windows
        md = metadata(snaps[-1], par); md.update({"hash": h, "bridge_s": br, "times": [i * (SETTLE // NSNAP) for i in SAVE_IDX]})
        meta[tag] = md
        a = md.get("aligned"); A = md.get("A")
        print(f"  [{tag} {h}] saved {len(SAVE_IDX)} frames; nodes={md['n_nodes']} "
              f"bridge_pair={md.get('bridge_pair')} A={A} aligned={a} s_i={md.get('s_i'):.1f} s_j={md.get('s_j'):.1f}")
    np.savez_compressed(os.path.join(outdir, "chiral_fields.npz"), **out)
    json.dump(meta, open(os.path.join(outdir, "chiral_meta.json"), "w"), indent=2, default=float)
    print(f"wrote {outdir}/chiral_fields.npz + chiral_meta.json + rho_history_*.h5")
    print(f"render on Windows: python plugins/visualizers/payan_chiral_slices.py")


if __name__ == "__main__":
    main()
