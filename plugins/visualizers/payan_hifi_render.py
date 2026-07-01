"""
HI-FI CONTINUATION RENDERER (Windows / matplotlib) — qualitative + quantitative readout of the
resolution/time ladder from jax_scout/payan_hifi_continuation.py. For every hifi_N*_L*_T* dir:

  * timeseries_<dir>.png : er, n_nodes, min_node_dist, bridge_cond vs time (stable & unstable).
  * vortex_<tag>.png     : per-node core density + radial flow v_r + tangential flow v_t + swirl
                           ratio over time (the vortex-sink / venting-via-spin test).
  * tracks_<tag>.png     : node trajectories (x-y projection, colour = time) + merge/birth markers.
  * slices_<tag>.png     : rho contour + J quiver + axial vorticity at a few frames (dominant node).
And a convergence_overlay.png comparing er(t) and n_nodes(t) across N (resolution convergence).

Run (Windows python): python plugins/visualizers/payan_hifi_render.py
"""
import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from plugins.visualizers.payan_chiral_slices import fields   # J/vorticity recompute from psi


def _series(summary, tag, key):
    s = summary.get(tag, {}).get("series", [])
    return [r["t_step"] for r in s], [r.get(key) for r in s]


def timeseries(meta, outdir, name):
    summ = meta["summary"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("er", "energy ratio", 0, 0), ("n_nodes", "node count", 0, 1),
              ("min_node_dist", "min node-node dist (vox)", 1, 0), ("bridge_cond", "max bridge cond", 1, 1)]
    for key, lab, r, c in panels:
        ax = axes[r][c]
        for tag, col in (("stable", "tab:blue"), ("unstable", "tab:red")):
            if tag in summ:
                x, y = _series(summ, tag, key)
                y = [np.nan if v is None else v for v in y]
                ax.plot(x, y, "-o", ms=3, color=col, label=tag)
        ax.set_title(lab); ax.set_xlabel("t step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        if key == "er":
            ax.axhspan(0.5, 2.0, color="green", alpha=0.07)
    fig.suptitle(f"HI-FI continuation {name}  (N={meta['N']} L={meta['L']} T={meta['T']})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = os.path.join(outdir, f"timeseries_{name}.png"); fig.savefig(fp, dpi=110); plt.close(fig)
    print(f"  wrote {fp}")


def vortex(meta, outdir, tag):
    s = meta["summary"].get(tag, {}).get("series", [])
    if not s:
        return
    t = [r["t_step"] for r in s]
    # dominant (highest-E) node per frame = nodes[0]
    cr = [r["nodes"][0]["core_rho"] if r["nodes"] else np.nan for r in s]
    vr = [r["nodes"][0]["v_r"] if r["nodes"] else np.nan for r in s]
    vt = [r["nodes"][0]["v_t"] if r["nodes"] else np.nan for r in s]
    swirl = [(abs(b) / (abs(a) + abs(b) + 1e-30)) if (r["nodes"]) else np.nan
             for r, a, b in zip(s, vr, vt)]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(t, cr, "-o", ms=3, color="purple"); ax[0].set_title("core density (dominant node)")
    ax[1].plot(t, vr, "-o", ms=3, color="tab:red", label="v_r (radial; <0 = inflow)")
    ax[1].plot(t, vt, "-o", ms=3, color="tab:blue", label="v_t (tangential/circulation)")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].legend(fontsize=8); ax[1].set_title("radial vs tangential flow")
    ax[2].plot(t, swirl, "-o", ms=3, color="green"); ax[2].axhline(0.6, color="0.6", ls="--")
    ax[2].set_title("swirl ratio v_t/(|v_r|+v_t)  (>0.6 ~ vortex/spiral)")
    for a in ax: a.set_xlabel("t step"); a.grid(alpha=0.3)
    fig.suptitle(f"VORTEX-CORE dynamics — {tag} (N={meta['N']} T={meta['T']})  "
                 f"inflow+circulation => venting via spin", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fp = os.path.join(outdir, f"vortex_{tag}.png"); fig.savefig(fp, dpi=110); plt.close(fig)
    print(f"  wrote {fp}")


def tracks(meta, outdir, tag):
    trk = meta["summary"].get(tag, {}).get("tracks", [])
    if not trk:
        return
    N = meta["N"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for k, tr in enumerate(trk):
        pos = np.array(tr["pos"])
        if len(pos) < 1:
            continue
        ax.plot(pos[:, 0], pos[:, 1], "-", lw=1, alpha=0.5)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=np.arange(len(pos)), cmap="viridis", s=18, zorder=3)
        ax.plot(pos[0, 0], pos[0, 1], "ks", ms=6)       # birth
        ax.plot(pos[-1, 0], pos[-1, 1], "r*", ms=12)    # last seen
        ax.text(pos[-1, 0], pos[-1, 1], f" {k}", fontsize=8)
    ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_aspect("equal")
    ax.set_title(f"node tracks (x-y proj) — {tag} N={meta['N']}\n■ birth  ★ last  colour=frame")
    fig.colorbar(sc, ax=ax, label="frame")
    fp = os.path.join(outdir, f"tracks_{tag}.png"); fig.savefig(fp, dpi=110); plt.close(fig)
    print(f"  wrote {fp}")


def slices(meta, frames, outdir, tag, L=10.0):
    key = f"psi_{tag}"
    if key not in frames:
        return
    psis = frames[key]; idx = frames[f"frames_{tag}"]
    s = meta["summary"].get(tag, {}).get("series", [])
    N = psis.shape[1]; dx = meta["L"] / N
    nshow = min(4, psis.shape[0]); pick = np.linspace(0, psis.shape[0] - 1, nshow).astype(int)
    fig, axes = plt.subplots(1, nshow, figsize=(4 * nshow, 4), squeeze=False)
    for col, fi in enumerate(pick):
        psi = psis[fi].astype(np.complex128)
        rho, (Jx, Jy, Jz), (wx, wy, wz) = fields(psi, dx, N)
        # dominant node centroid at this frame (nearest series entry)
        fr = int(idx[fi])
        srow = min(s, key=lambda r: abs(r["frame"] - fr)) if s else None
        # slice at z of densest voxel
        cz = int(np.unravel_index(np.argmax(rho), rho.shape)[2])
        ax = axes[0][col]
        w = wz[:, :, cz]; wl = np.percentile(np.abs(w), 99) + 1e-30
        ax.imshow(w.T, origin="lower", cmap="bwr", vmin=-wl, vmax=wl, alpha=0.85)
        ax.contour(rho[:, :, cz].T, levels=6, colors="k", linewidths=0.4, alpha=0.6)
        st = max(1, N // 16)
        gx, gy = np.meshgrid(np.arange(0, N, st), np.arange(0, N, st), indexing="ij")
        ax.quiver(gx, gy, Jx[::st, ::st, cz], Jy[::st, ::st, cz], color="0.15",
                  angles="xy", scale_units="xy", width=0.004)
        nn = srow["n_nodes"] if srow else "?"
        ax.set_title(f"t={fr*(meta['T']//meta['summary'][tag]['n_snap'])} nodes={nn}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"slices (z@densest) — {tag} N={N}: bg=vorticity_z (red+/blue-), quiver=J, contour=ρ", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fp = os.path.join(outdir, f"slices_{tag}.png"); fig.savefig(fp, dpi=110); plt.close(fig)
    print(f"  wrote {fp}")


def main():
    dirs = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "SUBSTRATE_HUNT_2026*", "hifi_N*_T*")))
    if not dirs:
        print("no hifi dirs found"); return
    convergence = {}
    for dd in dirs:
        meta = json.load(open(os.path.join(dd, "hifi_series.json")))
        name = os.path.basename(dd)
        print(f"[{name}] rendering...")
        frames = np.load(os.path.join(dd, "hifi_frames.npz"))
        timeseries(meta, dd, name)
        for tag in ("stable", "unstable"):
            if tag in meta["summary"]:
                vortex(meta, dd, tag); tracks(meta, dd, tag); slices(meta, frames, dd, tag, L=meta["L"])
        convergence[name] = meta
    # cross-N convergence overlay (er and n_nodes vs t)
    if len(convergence) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for name, meta in sorted(convergence.items()):
            for ci, tag in enumerate(("stable", "unstable")):
                if tag not in meta["summary"]:
                    continue
                x, er = _series(meta["summary"], tag, "er")
                _, nn = _series(meta["summary"], tag, "n_nodes")
                axes[0][ci].plot(x, [np.nan if v is None else v for v in er], "-o", ms=2, label=f"N{meta['N']}")
                axes[1][ci].plot(x, [np.nan if v is None else v for v in nn], "-o", ms=2, label=f"N{meta['N']}")
                axes[0][ci].set_title(f"er(t) — {tag}"); axes[1][ci].set_title(f"n_nodes(t) — {tag}")
        for a in axes.ravel():
            a.set_xlabel("t step"); a.grid(alpha=0.3); a.legend(fontsize=8)
        axes[0][0].axhspan(0.5, 2.0, color="green", alpha=0.07); axes[0][1].axhspan(0.5, 2.0, color="green", alpha=0.07)
        fig.suptitle("RESOLUTION CONVERGENCE: er(t) & n_nodes(t) across N (overlap => structures are real, not low-N artifacts)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fp = os.path.join(os.path.dirname(dirs[0]), "hifi_convergence_overlay.png")
        fig.savefig(fp, dpi=120); plt.close(fig)
        print(f"wrote {fp}")
    print("done")


if __name__ == "__main__":
    main()
