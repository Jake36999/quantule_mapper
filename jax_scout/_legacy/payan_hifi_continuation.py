"""
HI-FI CONTINUATION / RESOLUTION VALIDATION (no solver coupling, no solver mod).

Question (per the chiral-slice reading): are the node-merger + vortex-core (radial inflow feeding
angular circulation) dynamics REAL persistent structures, or short-window / low-grid artifacts?
Re-runs the SAME two configs (stable feb56dc7, unstable b31c0396; same seed/params) at parametrized
grid N and horizon T, tracking the rich per-node dynamics over time -- crucially the RADIAL /
TANGENTIAL flow decomposition around each node shell (vortex-sink / spiral-attractor test).

Per finite frame: node tracks, node-node min distance, per-node core density + shell (v_r, v_theta,
|omega|) decomposition + classification (spiral_sink / radial_sink / vortex / source / mixed),
max bridge conductance, energy ratio, curvature, component count. Truncates at blow-up (keeps finite
frames). Saves time-series (npz/json) + subsampled psi frames (complex64) for Windows rendering.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/payan_hifi_continuation.py --N 96 --T 1600
"""
import os, sys, csv, glob, json, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import transfer_diag as td, geometry_diag as gd
from jax_scout import afield_current_coupled as cc
from jax_scout.afield_current_coupled import multiseed_ic, L as L_DEFAULT
from jax_scout._legacy.afield_payan_diagnostic import order

SEED = 20260619
HASHES = {"stable": "feb56dc7", "unstable": "b31c0396"}


def F(r, k):
    try: return float(r[k])
    except: return float("nan")


def load_by_hash(d, h):
    for r in csv.DictReader(open(os.path.join(d, "all_evals.csv"))):
        if r.get("hash") == h:
            return {k: F(r, k) for k in order}, F(r, "gamma_A"), F(r, "kappa"), F(r, "c_A"), F(r, "bridge_s")
    return None


def _rel(coord, c, N):
    return ((coord - c + N / 2) % N) - N / 2     # signed minimal-image displacement (voxels)


def rt_decomp(psi, par, c, dx, N, r_in, r_out):
    """Radial/tangential flow decomposition + core density around node center c (voxel)."""
    geo = td.geometry_fields(psi, par, dx)
    rho = geo["rho"]; Jx, Jy, Jz = geo["J"]
    rs = np.maximum(rho, 1e-6); vx, vy, vz = Jx / rs, Jy / rs, Jz / rs
    ax = np.arange(N)
    DX = _rel(ax[:, None, None], c[0], N) * np.ones((N, N, N))
    DY = _rel(ax[None, :, None], c[1], N) * np.ones((N, N, N))
    DZ = _rel(ax[None, None, :], c[2], N) * np.ones((N, N, N))
    rr = np.sqrt(DX ** 2 + DY ** 2 + DZ ** 2) + 1e-9
    shell = (rr >= r_in) & (rr <= r_out)
    rhx, rhy, rhz = DX / rr, DY / rr, DZ / rr
    vr = vx * rhx + vy * rhy + vz * rhz
    vpx, vpy, vpz = vx - vr * rhx, vy - vr * rhy, vz - vr * rhz
    vt = np.sqrt(vpx ** 2 + vpy ** 2 + vpz ** 2)
    if not np.any(shell):
        return {"v_r": 0.0, "v_t": 0.0, "core_rho": float(rho[tuple(c % N)])}
    return {"v_r": float(np.mean(vr[shell])), "v_t": float(np.mean(vt[shell])),
            "core_rho": float(rho[tuple(c % N)])}


def classify(v_r, v_t):
    swirl = v_t / (abs(v_r) + v_t + 1e-30)
    if v_r < 0 and swirl > 0.6:
        return "spiral_sink"
    if v_r < 0:
        return "radial_sink"
    if v_r > 0 and swirl < 0.4:
        return "source"
    if swirl > 0.6:
        return "vortex"
    return "mixed"


def track(prev_tracks, nodes, N, thresh):
    """Greedy nearest-centroid tracking (minimal image)."""
    cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    used = set()
    for tr in prev_tracks:
        last = np.array(tr["pos"][-1])
        best, bi = thresh, -1
        for i, c in enumerate(cents):
            if i in used:
                continue
            dd = _rel(c.astype(float), last.astype(float), N)
            dist = float(np.linalg.norm(dd))
            if dist < best:
                best, bi = dist, i
        if bi >= 0:
            tr["pos"].append(cents[bi].tolist()); tr["alive"] = True; used.add(bi); tr["_match"] = bi
        else:
            tr["alive"] = False; tr["_match"] = -1
    for i, c in enumerate(cents):
        if i not in used:
            prev_tracks.append({"pos": [c.tolist()], "alive": True, "_match": i, "born": True})
    return prev_tracks, cents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=L_DEFAULT)
    ap.add_argument("--T", type=int, default=1600)
    ap.add_argument("--save-frames", type=int, default=12)
    args = ap.parse_args()
    N, Lb, T = args.N, args.L, args.T
    dx = Lb / N
    if Lb != L_DEFAULT:
        cc.L = Lb   # multiseed_ic + capture_cc read cc.L
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "SUBSTRATE_HUNT_2026*")))[-1]
    outdir = os.path.join(d, f"hifi_N{N}_L{int(Lb)}_T{T}"); os.makedirs(outdir, exist_ok=True)
    n_snap = max(8, T // 100)
    save_idx = sorted(set(np.linspace(1, n_snap, args.save_frames).astype(int).tolist()))
    print(f"=== HI-FI CONTINUATION N={N} L={Lb} T={T} dx={dx:.4f} ({n_snap} snaps) ===")
    summary = {}
    saved_frames = {}
    for tag, h in HASHES.items():
        cfg = load_by_hash(d, h)
        if cfg is None:
            print(f"[{tag} {h}] not found in CSV"); continue
        par, g, kap, cA, br = cfg
        t0 = time.time()
        snaps, Asnaps, fin = cc.capture_cc(par, multiseed_ic(N, SEED), g, N, T, n_snap, kappa=kap, c_A=cA)
        snaps = np.asarray(snaps)
        # contiguous finite prefix
        finite = [t for t in range(snaps.shape[0]) if np.all(np.isfinite(np.abs(snaps[t])))]
        last = finite[-1] if finite else 0
        nfin = last + 1
        e0 = float(np.sum(np.abs(snaps[0]) ** 2)) + 1e-30
        node_r = None; tracks = []; series = []
        for t in range(nfin):
            psi = snaps[t]
            nodes = sorted(td.detect_nodes(psi, dx), key=lambda n: -n["E"])
            er = float(np.sum(np.abs(psi) ** 2) / e0)
            curv = float(gd.curvature_max_only(psi, par, dx))
            if node_r is None and nodes:
                node_r = max(2, int(round(np.mean([n["size"] for n in nodes]) ** (1 / 3))))
            nr = node_r or 3
            cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
            # min pairwise distance
            mind = float("inf")
            for i in range(len(cents)):
                for j in range(i + 1, len(cents)):
                    mind = min(mind, float(np.linalg.norm(_rel(cents[i].astype(float), cents[j].astype(float), N))))
            # bridge conductance + per top-3 node decomposition
            bridge = 0.0; ndec = []
            if len(nodes) >= 2:
                geo = td.geometry_fields(psi, par, dx)
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        bridge = max(bridge, td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"])
            for n in nodes[:3]:
                c = np.round(n["centroid"]).astype(int) % N
                dec = rt_decomp(psi, par, c, dx, N, max(1, nr - 1), nr + 2)
                dec["klass"] = classify(dec["v_r"], dec["v_t"]); dec["E"] = float(n["E"])
                ndec.append(dec)
            track(tracks, nodes, N, thresh=max(4, nr * 2))
            series.append({"frame": t, "t_step": t * (T // n_snap), "n_nodes": len(nodes), "er": er,
                           "curv": curv, "bridge_cond": float(bridge), "min_node_dist": (None if mind == float("inf") else mind),
                           "nodes": ndec})
        # save subsampled frames (complex64) for rendering
        keep = [t for t in save_idx if t < nfin]
        saved_frames[f"psi_{tag}"] = snaps[keep].astype(np.complex64)
        saved_frames[f"frames_{tag}"] = np.array(keep)
        last_er = series[-1]["er"] if series else float("nan")
        outcome = "stayed_finite" if nfin == snaps.shape[0] else f"blew_up_at_frame_{nfin}"
        summary[tag] = {"hash": h, "bridge_s": br, "n_frames_finite": nfin, "n_snap": n_snap,
                        "outcome": outcome, "final_er": last_er,
                        "final_n_nodes": series[-1]["n_nodes"] if series else 0,
                        "n_tracks": len(tracks), "series": series,
                        "tracks": [{"pos": tr["pos"]} for tr in tracks]}
        nn = [s["n_nodes"] for s in series]
        print(f"[{tag} {h}] {outcome}; frames={nfin}/{snaps.shape[0]} final_er={last_er:.2f} "
              f"nodes:{nn[0] if nn else 0}->{nn[-1] if nn else 0} (min {min(nn) if nn else 0}/max {max(nn) if nn else 0}) "
              f"tracks={len(tracks)}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(outdir, "hifi_frames.npz"), **saved_frames)
    json.dump({"N": N, "L": Lb, "T": T, "dx": dx, "summary": summary},
              open(os.path.join(outdir, "hifi_series.json"), "w"), indent=2, default=float)
    print(f"wrote {outdir}/hifi_series.json + hifi_frames.npz")


if __name__ == "__main__":
    main()
