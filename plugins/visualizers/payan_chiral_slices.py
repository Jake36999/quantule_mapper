"""
CHIRAL-PAIR SLICE RENDERER (Windows / matplotlib + optional pyvista) — qualitative support only.

Reads chiral_viz/chiral_fields.npz + chiral_meta.json (from jax_scout/payan_chiral_capture.py) and
renders, per config, slices PERPENDICULAR to the bridge axis at each bridge node across time:
  background = density rho ; quiver = in-plane informational current J ; colormap = axial vorticity
  omega.axis (handedness: red=+ / blue=-). Lets us see whether the 90% anti-alignment is local
  compensatory circulation vs true bridge-pair anti-handedness, and how radial density grading evolves.
Also (best-effort) a 3D density isosurface GIF per config via the existing visual_analysis_suite.

Run (Windows python, has matplotlib/pyvista):
  python plugins/visualizers/payan_chiral_slices.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
L = 10.0


def _grad_fft(psi, dx, N):
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    pk = np.fft.fftn(psi)
    gx = np.fft.ifftn(1j * kx * pk); gy = np.fft.ifftn(1j * ky * pk); gz = np.fft.ifftn(1j * kz * pk)
    return gx, gy, gz


def fields(psi, dx, N):
    gx, gy, gz = _grad_fft(psi, dx, N)
    rho = np.abs(psi) ** 2
    c = np.conj(psi)
    Jx = np.imag(c * gx); Jy = np.imag(c * gy); Jz = np.imag(c * gz)
    rs = np.maximum(rho, 1e-6)
    vx, vy, vz = Jx / rs, Jy / rs, Jz / rs
    # vorticity via finite diff
    dvx = np.gradient(vx, dx); dvy = np.gradient(vy, dx); dvz = np.gradient(vz, dx)
    wx = dvz[1] - dvy[2]; wy = dvx[2] - dvz[0]; wz = dvy[0] - dvx[1]
    return rho, (Jx, Jy, Jz), (wx, wy, wz)


def render_config(tag, psi_frames, md, outdir):
    N = psi_frames.shape[1]; dx = L / N
    times = md.get("times", list(range(psi_frames.shape[0])))
    bp = md.get("bridge_pair", [0, 1]); cents = md["centroids_vox"]
    axis = np.array(md.get("bridge_axis", [0, 0, 1]), float)
    p = int(np.argmax(np.abs(axis)))               # principal grid axis ~ bridge axis
    inplane = [a for a in range(3) if a != p]
    ci, cj = cents[bp[0]], cents[bp[1]]
    nodes = [("node_i", ci, md.get("s_i")), ("node_j", cj, md.get("s_j"))]
    nf = psi_frames.shape[0]
    fig, axes = plt.subplots(2, nf, figsize=(3.4 * nf, 7), squeeze=False)
    step = max(1, N // 16)
    for r, (nm, c, s) in enumerate(nodes):
        for t in range(nf):
            rho, (Jx, Jy, Jz), (wx, wy, wz) = fields(psi_frames[t].astype(np.complex128), dx, N)
            sl = [slice(None)] * 3; sl[p] = int(c[p]) % N; sl = tuple(sl)
            rho2 = rho[sl]; w_axis = (wx, wy, wz)[p][sl]
            J_in = [(Jx, Jy, Jz)[inplane[0]][sl], (Jx, Jy, Jz)[inplane[1]][sl]]
            ax = axes[r][t]
            wlim = np.percentile(np.abs(w_axis), 99) + 1e-30
            ax.imshow(w_axis.T, origin="lower", cmap="bwr", vmin=-wlim, vmax=wlim, alpha=0.85)
            ax.contour(rho2.T, levels=6, colors="k", linewidths=0.4, alpha=0.6)
            gx, gy = np.meshgrid(np.arange(0, N, step), np.arange(0, N, step), indexing="ij")
            ax.quiver(gx, gy, J_in[0][::step, ::step], J_in[1][::step, ::step],
                      color="0.15", scale_units="xy", angles="xy", width=0.004)
            ax.plot(c[inplane[0]], c[inplane[1]], "g+", ms=14, mew=2)
            chir = "+" if (s is not None and s > 0) else "-"
            ax.set_title(f"{nm} t={times[t]} | spin {chir}{'' if s is None else f' ({s:.0f})'}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    aligned = md.get("aligned"); A = md.get("A")
    fig.suptitle(f"CHIRAL PAIR — {tag} ({md.get('hash')}) bridge_s={md.get('bridge_s'):.2f}  "
                 f"A={A} aligned={aligned}  | slices ⟂ bridge axis (grid axis {p}); "
                 f"bg=axial vorticity (red+/blue-), quiver=J, contour=ρ", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = os.path.join(outdir, f"chiral_slices_{tag}.png")
    fig.savefig(fp, dpi=110); plt.close(fig)
    print(f"  wrote {fp}")


def render_3d(tag, rho_hist, outdir):
    try:
        import h5py
        from plugins.visualizers import visual_analysis_suite as vas
    except Exception as e:
        print(f"  [3D {tag}] skipped ({type(e).__name__})"); return
    h5p = os.path.join(outdir, f"rho_history_{tag}.h5")
    with h5py.File(h5p, "w") as f:
        f.create_dataset("rho_history", data=rho_hist.astype(np.float32))
    try:
        vas.render_single_run(h5p, None, f"{tag}", os.path.join(outdir, f"density3d_{tag}.gif"))
        print(f"  wrote {outdir}/density3d_{tag}.gif")
    except Exception as e:
        print(f"  [3D {tag}] render failed ({type(e).__name__}: {e})")


def main():
    d = sorted([p for p in __import__("glob").glob(os.path.join(ROOT, "sweep_runs", "SUBSTRATE_HUNT_2026*"))])[-1]
    outdir = os.path.join(d, "chiral_viz")
    npz = np.load(os.path.join(outdir, "chiral_fields.npz"))
    meta = json.load(open(os.path.join(outdir, "chiral_meta.json")))
    for tag in ("stable", "unstable"):
        if f"psi_{tag}" not in npz:
            print(f"[{tag}] no fields"); continue
        print(f"[{tag}] rendering slices...")
        render_config(tag, npz[f"psi_{tag}"], meta[tag], outdir)
        if f"rho_hist_{tag}" in npz:
            render_3d(tag, npz[f"rho_hist_{tag}"], outdir)
    print(f"done -> {outdir}")


if __name__ == "__main__":
    main()
