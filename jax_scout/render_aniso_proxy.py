"""
Before/after 3D Plotly renderings of the anisotropic-proxy bridge-selectivity result.
Inspect whether the routing-selective bridge corresponds to real spatial structure (not a metric
artifact). gen6 (selectivity 0.20->9.21) is priority; gen29 (1.70->4.34) also rendered.

Per config x lambda{0, 0.1}: re-run the conservative anisotropic capture, render
  * resonance density |psi|^2 isosurface,
  * persistent node markers,
  * bridge-kick / void-kick / node-kick site markers,
  * informational current |J| flow isosurface (directional-structure overlay).
Self-contained HTML (offline-viewable).

WSL2 jax venv:  python jax_scout/render_aniso_proxy.py
"""
import os, sys, json, glob, csv
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
import plotly.graph_objects as go
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td
from jax_scout.afield_anisotropic import capture_aniso
from jax_scout.afield_current_coupled import multiseed_ic, L, order

BASE_SEED, SETTLE, N = 20260619, 800, 48
OUTDIR = os.path.join(ROOT, "sweep_runs", sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "sweep_runs", "AF_BRIDGE_HUNT_2026*")))[-1:][0])


def _bump_d2(c):
    G = np.meshgrid(*([np.arange(N)]*3), indexing="ij")
    return sum(np.minimum((G[a]-c[a]) % N, (c[a]-G[a]) % N).astype(float)**2 for a in range(3))


def _grad(f, dx):
    return ((np.roll(f, -1, 0)-np.roll(f, 1, 0))/(2*dx), (np.roll(f, -1, 1)-np.roll(f, 1, 1))/(2*dx),
            (np.roll(f, -1, 2)-np.roll(f, 1, 2))/(2*dx))


def render(par, g, kap, cA, lam, label, tag):
    dx = L/N
    snaps, fin = capture_aniso(par, multiseed_ic(N, BASE_SEED), g, lam, N, SETTLE, 20, kappa=kap, c_A=cA, q_source="stress")
    if not fin:
        print(f"  {tag}: nonfinite, skip"); return None
    psi = snaps[-1]; rho = np.abs(psi)**2
    nodes = td.detect_nodes(psi, dx)
    cents = [np.round(n["centroid"]).astype(int) % N for n in sorted(nodes, key=lambda n: -n["E"])] if nodes else []
    # current magnitude (directional-structure overlay)
    gx, gy, gz = _grad(psi, dx); Jm = np.sqrt(np.imag(np.conj(psi)*gx)**2+np.imag(np.conj(psi)*gy)**2+np.imag(np.conj(psi)*gz)**2)
    # bridge/void/node-kick markers
    bpt = void = None; bridge_cond = 0.0
    if len(nodes) >= 2:
        geo = td.geometry_fields(psi, par, dx); best, bp = -1, (0, 1)
        for i in range(len(cents)):
            for j in range(i+1, len(cents)):
                c = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
                if c > best:
                    best, bp = c, (i, j)
        bridge_cond = float(best)
        disp = (cents[bp[1]]-cents[bp[0]]).astype(float); disp = disp - N*np.round(disp/N)
        bpt = np.round(cents[bp[0]]+0.5*disp).astype(int) % N
        far = np.ones((N, N, N), bool)
        for c in cents:
            far &= _bump_d2(c) > 8**2
        void = np.array(np.unravel_index(np.argmin(np.where(far, rho, rho.max()+1)), rho.shape))

    X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    data = [go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=rho.flatten(),
                          isomin=float(rho.mean()+1.5*rho.std()), isomax=float(rho.max()),
                          surface_count=3, opacity=0.45, colorscale="Viridis", showscale=True,
                          colorbar=dict(title="rho", x=0.95), name="density")]
    data.append(go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=Jm.flatten(),
                              isomin=float(Jm.mean()+2*Jm.std()), isomax=float(Jm.max()),
                              surface_count=2, opacity=0.18, colorscale="Hot", showscale=False, name="|J| current"))
    if cents:
        data.append(go.Scatter3d(x=[c[0] for c in cents], y=[c[1] for c in cents], z=[c[2] for c in cents],
                                 mode="markers", marker=dict(size=7, color="lime", symbol="circle"), name="nodes"))
    for pt, col, nm in [(bpt, "red", "bridge-kick"), (void, "blue", "void-kick"),
                        (cents[0] if cents else None, "orange", "node-kick")]:
        if pt is not None:
            data.append(go.Scatter3d(x=[pt[0]], y=[pt[1]], z=[pt[2]], mode="markers",
                                     marker=dict(size=9, color=col, symbol="diamond"), name=nm))
    fig = go.Figure(data=data)
    fig.update_layout(title=f"{label}  (lambda={lam}, q=stress)  bridge_cond={bridge_cond:.3f}  nodes={len(nodes)}  er={float(rho.sum()/(np.abs(snaps[0])**2).sum()):.2f}",
                      scene=dict(aspectmode="cube"), width=900, height=800, margin=dict(l=0, r=0, t=40, b=0))
    out = os.path.join(OUTDIR, f"{tag}.html")
    fig.write_html(out, include_plotlyjs=True)
    print(f"  wrote {out}  (bridge_cond={bridge_cond:.3f}, nodes={len(nodes)})")
    return out


def main():
    rows = list(csv.DictReader(open(os.path.join(OUTDIR, "all_evals.csv"))))
    def F(r, k):
        try: return float(r[k])
        except: return float("nan")
    sb = json.load(open(os.path.join(OUTDIR, "afield_aniso_strongbridge.json")))
    repro = [p for p in sb["panel"] if p.get("reproduced")]
    written = []
    for p in sorted(repro, key=lambda p: -p["bridge0"]):  # gen29 0.68, gen6 0.41 -> gen6 priority noted below
        gen = p["label"].split("_")[0].replace("gen", "")
        r = next((r for r in rows if r["gen"] == gen and abs(F(r, "bridge")-p["bridge0"]) < 0.01), None)
        if not r:
            continue
        par = {k: F(r, k) for k in order}; g, kap, cA = F(r, "gamma_A"), F(r, "kappa"), F(r, "c_A")
        name = p["label"]
        for lam, suf in [(0.0, "lambda0_before"), (0.1, "lambda01_after")]:
            o = render(par, g, kap, cA, lam, name, f"{name.split('_')[0]}_{suf}")
            if o:
                written.append(o)
    print(f"\nrendered {len(written)} HTMLs in {OUTDIR}")


if __name__ == "__main__":
    main()
