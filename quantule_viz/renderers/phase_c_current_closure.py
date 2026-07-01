"""Phase C N96 current-closure / signed-vorticity static analysis (read-only, jax-free).

Computes signed-current and signed-vorticity *closure* diagnostics on the saved N96 final
fields (``probe_data.npz`` -> ``psi_fin``) of the Stage 1 Option B validation, to test the
corrected working hypothesis that stable branches need coherent signed current/vorticity
closure + source support around the node, rather than the retired distributed/compact rule.

Terminology is deliberate: *signed vorticity*, *current closure*, *signed circulation*,
*polarity balance* (geometric handedness of the current field). This is NOT electric-charge
polarity and carries no charge / topological / proof / ground-state interpretation.

All quantities are computed in voxel-index space (np.gradient unit spacing); ratios, signs,
and normalized scores are scale-free. Offsets are also reported in box units (dx = L/N).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from ..io import load_json, write_csv
from ..plots import CLASS_COLORS, as_density, find_component_centroids, phase_current, save_figure

L_BOX = 10.0

# Canonical case order + roles (Stage 1 keys)
CASE_ORDER = [
    "k6_high_mass_true",
    "k6_mid_mass_true",
    "k4_intermediate_true",
    "feb56dc7_control",
    "k2_intermediate_true",
    "k1_low_mass_true",
    "k1_high_mass_failure",
    "k6_near_threshold_near",
]
CASE_LABEL = {
    "k6_high_mass_true": "K6 high-mass (spin-down)",
    "k6_mid_mass_true": "K6 mid-mass (TRUE)",
    "k4_intermediate_true": "K4 intermediate (TRUE)",
    "feb56dc7_control": "feb56dc7 (TRUE control)",
    "k2_intermediate_true": "K2 compact (TRUE)",
    "k1_low_mass_true": "K1 low-mass (blowup)",
    "k1_high_mass_failure": "K1 high-mass (blowup)",
    "k6_near_threshold_near": "K6 near-threshold (rising TRUE)",
}

OUTPUT_FILES = (
    "current_closure_score_table.csv",
    "ring_vorticity_profiles.png",
    "current_closure_comparison.png",
    "density_current_vorticity_slices.png",
    "node_center_offset_summary.png",
    "axis_polarity_profiles.png",
)


# --------------------------------------------------------------------------- field loading
def _finite_fraction(arr: np.ndarray) -> float:
    return float(np.isfinite(arr).mean())


def _load_analysis_field(npz_path: Path) -> tuple[np.ndarray | None, str, float]:
    """Return (field, field_used, finite_fraction). psi_fin preferred; psi_mid fallback.
    None field => no finite field available (blowup)."""
    bundle = np.load(npz_path, allow_pickle=True)
    for name in ("psi_fin", "psi_mid"):
        if name in bundle.files:
            arr = np.asarray(bundle[name])
            frac = _finite_fraction(arr)
            if frac > 0.9999:
                return arr.astype(np.complex128), name, frac
    # nothing finite: report psi_fin's fraction for the record
    frac = _finite_fraction(np.asarray(bundle["psi_fin"])) if "psi_fin" in bundle.files else 0.0
    return None, "psi_fin", frac


# --------------------------------------------------------------------------- geometry
def _primary_node(rho: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(rho)), rho.shape))  # type: ignore[return-value]


def _core_radius(rho: np.ndarray, c: tuple[int, int, int]) -> int:
    """Spherically-averaged radius (voxels) where density falls below half-peak about c."""
    N = rho.shape[0]
    rmax = max(4, N // 4)
    a0, a1, a2 = np.indices(rho.shape)
    r = np.sqrt((a0 - c[0]) ** 2 + (a1 - c[1]) ** 2 + (a2 - c[2]) ** 2)
    rbin = np.minimum(r.astype(int), rmax)
    peak = float(rho[c])
    prof = np.array([rho[rbin == k].mean() if np.any(rbin == k) else 0.0 for k in range(rmax + 1)])
    below = np.where(prof < 0.5 * peak)[0]
    r0 = int(below[0]) if below.size else rmax
    return int(np.clip(r0, 3, rmax))


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Unit vectors ~evenly on a sphere (index-space directions)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _sample(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Trilinear sample real field at index-space points [M,3] (periodic wrap)."""
    return map_coordinates(field, [pts[:, 0], pts[:, 1], pts[:, 2]], order=1, mode="grid-wrap")


# --------------------------------------------------------------------------- shell metrics
def _shell_metrics(jx, jy, jz, c, r0) -> dict:
    """Net signed radial flux (source/sink) and radial/tangential split on a sphere r0."""
    dirs = _fibonacci_sphere(300)
    pts = np.asarray(c, dtype=float)[None, :] + r0 * dirs
    Jx = _sample(jx, pts)
    Jy = _sample(jy, pts)
    Jz = _sample(jz, pts)
    J = np.stack([Jx, Jy, Jz], axis=1)
    Jr = np.sum(J * dirs, axis=1)                       # radial component
    Jt = J - Jr[:, None] * dirs                         # tangential vector
    Jt_mag = np.linalg.norm(Jt, axis=1)
    denom = float(np.sum(np.abs(Jr))) + 1e-30
    net_flux = float(np.sum(Jr)) / denom                # >0 outflow/leak, <0 inflow/feed, ~0 closed
    mean_abs_Jr = float(np.mean(np.abs(Jr)))
    mean_abs_Jt = float(np.mean(Jt_mag))
    rad_tan_ratio = mean_abs_Jr / (mean_abs_Jt + 1e-30)
    tang_fraction = mean_abs_Jt / (mean_abs_Jt + mean_abs_Jr + 1e-30)
    return {"net_flux_score": net_flux, "rad_tan_ratio": rad_tan_ratio, "tang_fraction": tang_fraction}


# --------------------------------------------------------------------------- plane / ring metrics
def _plane(jx, jy, jz, axis: int, center: int):
    """In-plane current (u,v) + signed pseudo-vorticity on the plane perpendicular to axis."""
    if axis == 0:
        u, v = jy[center, :, :], jz[center, :, :]
    elif axis == 1:
        u, v = jx[:, center, :], jz[:, center, :]
    else:
        u, v = jx[:, :, center], jy[:, :, center]
    vort = np.gradient(v, axis=0) - np.gradient(u, axis=1)
    return np.asarray(u), np.asarray(v), np.asarray(vort)


def _plane_axes(axis: int, c) -> tuple[int, int]:
    """In-plane center coords (the two indices kept when slicing perpendicular to axis)."""
    keep = [d for d in range(3) if d != axis]
    return c[keep[0]], c[keep[1]]


def _ring_profile(vort: np.ndarray, cx: float, cy: float, r0: float, n: int = 180) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xs = cx + r0 * np.cos(theta)
    ys = cy + r0 * np.sin(theta)
    w = map_coordinates(vort, [xs, ys], order=1, mode="grid-wrap")
    return theta, np.asarray(w)


def _ring_scores(theta: np.ndarray, w: np.ndarray) -> dict:
    pos = float(np.sum(np.clip(w, 0, None)))
    neg = float(np.sum(np.clip(-w, 0, None)))
    lobe_balance = (pos - neg) / (pos + neg + 1e-30)     # ~0 balanced +/- ; +-1 one-handed
    rms = float(np.sqrt(np.mean(w ** 2))) + 1e-30
    a1 = 2.0 * np.mean(w * np.cos(theta)); b1 = 2.0 * np.mean(w * np.sin(theta))
    a2 = 2.0 * np.mean(w * np.cos(2 * theta)); b2 = 2.0 * np.mean(w * np.sin(2 * theta))
    dipole = float(np.hypot(a1, b1)) / rms
    quadrupole = float(np.hypot(a2, b2)) / rms
    return {"lobe_balance": lobe_balance, "dipole": dipole, "quadrupole": quadrupole}


def _best_plane(jx, jy, jz, c):
    """Pick the coordinate plane through c with the strongest in-plane vorticity structure."""
    best = None
    for axis in range(3):
        u, v, vort = _plane(jx, jy, jz, axis, c[axis])
        score = float(np.sum(np.abs(vort)))
        if best is None or score > best[0]:
            cx, cy = _plane_axes(axis, c)
            best = (score, axis, u, v, vort, cx, cy)
    return best[1], best[2], best[3], best[4], best[5], best[6]


def _current_center_offset(vort: np.ndarray, cx: float, cy: float, r0: float) -> float:
    """|w|-weighted circulation centroid vs density node center, within a 2*r0 window (voxels)."""
    n = vort.shape
    x0, x1 = int(max(0, cx - 2 * r0)), int(min(n[0], cx + 2 * r0 + 1))
    y0, y1 = int(max(0, cy - 2 * r0)), int(min(n[1], cy + 2 * r0 + 1))
    sub = np.abs(vort[x0:x1, y0:y1])
    if sub.sum() <= 0:
        return float("nan")
    gx, gy = np.indices(sub.shape)
    wcx = float((gx * sub).sum() / sub.sum()) + x0
    wcy = float((gy * sub).sum() / sub.sum()) + y0
    return float(np.hypot(wcx - cx, wcy - cy))


def _axis_polarity(jx, jy, jz, c, r0):
    """Signed radial current J_r sampled at +/- r0 along each index axis from c."""
    base = np.asarray(c, dtype=float)
    out = {}
    for ax, name in zip(range(3), ("a0", "a1", "a2")):
        for sgn, tag in ((+1, "+"), (-1, "-")):
            p = base.copy(); p[ax] += sgn * r0
            J = np.array([_sample(jx, p[None])[0], _sample(jy, p[None])[0], _sample(jz, p[None])[0]])
            rhat = np.zeros(3); rhat[ax] = sgn
            out[f"{tag}{name}"] = float(J @ rhat)
    return out


def _closure_score(tang_fraction: float, lobe_balance: float, offset: float, r0: float) -> float:
    if not np.isfinite(offset):
        offset = 2.0 * r0
    s = tang_fraction * (1.0 - abs(lobe_balance)) * float(np.exp(-offset / max(r0, 1e-9)))
    return float(np.clip(s, 0.0, 1.0))


# --------------------------------------------------------------------------- per-case driver
def _analyze_case(key: str, npz_path: Path) -> dict:
    field, used, frac = _load_analysis_field(npz_path)
    rec = {"case": key, "label": CASE_LABEL.get(key, key), "field_used": used, "finite_fraction": round(frac, 4)}
    if field is None:
        rec.update({"status": "non_finite_blowup", "n_nodes": 0})
        return rec
    rho = as_density(field)
    jx, jy, jz = phase_current(field)
    c = _primary_node(rho)
    r0 = _core_radius(rho, c)
    nodes = find_component_centroids(field)
    shell = _shell_metrics(jx, jy, jz, c, r0)
    axis, u, v, vort, cx, cy = _best_plane(jx, jy, jz, c)
    theta, w = _ring_profile(vort, cx, cy, r0)
    ring = _ring_scores(theta, w)
    offset = _current_center_offset(vort, cx, cy, r0)
    closure = _closure_score(shell["tang_fraction"], ring["lobe_balance"], offset, r0)
    rec.update({
        "status": "ok",
        "n_nodes": len(nodes),
        "rho_peak": round(float(rho.max()), 5),
        "core_radius_vox": r0,
        "net_flux_score": round(shell["net_flux_score"], 4),
        "rad_tan_ratio": round(shell["rad_tan_ratio"], 4),
        "tang_fraction": round(shell["tang_fraction"], 4),
        "lobe_balance": round(ring["lobe_balance"], 4),
        "dipole_alt": round(ring["dipole"], 4),
        "quadrupole_alt": round(ring["quadrupole"], 4),
        "node_current_offset_vox": round(offset, 3) if np.isfinite(offset) else "nan",
        "node_current_offset_L": round(offset * L_BOX / rho.shape[0], 4) if np.isfinite(offset) else "nan",
        "current_closure_score": round(closure, 4),
        "_plane_axis": axis, "_c": c, "_r0": r0, "_cx": cx, "_cy": cy,
        "_theta": theta, "_w": w, "_u": u, "_v": v, "_vort": vort,
        "_rho_slice": rho.take(c[axis], axis=axis),
        "_axis_pol": _axis_polarity(jx, jy, jz, c, r0),
    })
    return rec


# --------------------------------------------------------------------------- figures
def _class_color(klass: str) -> str:
    return CLASS_COLORS.get(klass, "#666666")


def _fig_ring_profiles(recs, classes, out):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    any_line = False
    for r in recs:
        if r.get("status") != "ok":
            continue
        ax.plot(r["_theta"], r["_w"] / (np.max(np.abs(r["_w"])) + 1e-30),
                label=f"{r['label']}  (lobe={r['lobe_balance']:+.2f})",
                color=_class_color(classes.get(r["case"], "")), linewidth=1.8)
        any_line = True
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("ring azimuth theta (rad)"); ax.set_ylabel("normalized signed vorticity")
    ax.set_title("Ring signed-vorticity profile at core radius (dominant node, best plane)\nN96 final fields")
    if any_line:
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    return save_figure(fig, out)


def _fig_comparison(recs, classes, out):
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for r in recs:
        if r.get("status") != "ok":
            continue
        x = r["net_flux_score"]; y = r["current_closure_score"]
        ax.scatter(x, y, s=140, color=_class_color(classes.get(r["case"], "")), edgecolor="k", zorder=3)
        ax.annotate(r["label"], (x, y), fontsize=7.5, xytext=(6, 4), textcoords="offset points")
    ax.axvline(0, color="k", lw=0.7, alpha=0.5)
    ax.set_xlabel("net signed radial flux  (<0 inflow/feed   0 closed   >0 outflow/leak)")
    ax.set_ylabel("current-closure score (higher = more closed/standing)")
    ax.set_title("Current closure vs source/sink balance — N96 (color = N96 class)")
    ax.grid(alpha=0.25)
    return save_figure(fig, out)


def _fig_offset(recs, classes, out):
    ok = [r for r in recs if r.get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    xs = range(len(ok))
    ax.bar(xs, [r["node_current_offset_vox"] if isinstance(r["node_current_offset_vox"], (int, float)) else 0 for r in ok],
           color=[_class_color(classes.get(r["case"], "")) for r in ok], edgecolor="k")
    ax.set_xticks(list(xs)); ax.set_xticklabels([r["label"] for r in ok], rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("node->current-center offset (voxels)")
    ax.set_title("Density-node vs current-circulation-center offset — N96 (smaller = more node-centered closure)")
    return save_figure(fig, out)


def _fig_axis_polarity(recs, classes, out):
    ok = [r for r in recs if r.get("status") == "ok"]
    n = len(ok)
    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
    order = ["+a0", "-a0", "+a1", "-a1", "+a2", "-a2"]
    for i, r in enumerate(ok):
        ax = axes[i // cols][i % cols]
        vals = [r["_axis_pol"][k] for k in order]
        ax.bar(order, vals, color=_class_color(classes.get(r["case"], "")), edgecolor="k")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{r['label']}\nnet_flux={r['net_flux_score']:+.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_axis_off()
    fig.suptitle("Axis polarity: signed radial current J_r at +/- core radius along each axis (N96)", fontsize=11)
    return save_figure(fig, out)


def _fig_slices(recs, classes, out):
    n = len(recs)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.1 * n), squeeze=False)
    for i, r in enumerate(recs):
        a_rho, a_cur, a_vort = axes[i]
        if r.get("status") != "ok":
            for a in (a_rho, a_cur, a_vort):
                a.text(0.5, 0.5, "non-finite (blowup)\nclosure undefined", ha="center", va="center", fontsize=9)
                a.set_xticks([]); a.set_yticks([])
            a_rho.set_ylabel(r["label"], fontsize=8)
            continue
        rho_s = r["_rho_slice"]; u = r["_u"]; v = r["_v"]; vort = r["_vort"]; cx, cy, r0 = r["_cx"], r["_cy"], r["_r0"]
        a_rho.imshow(rho_s.T, origin="lower", cmap="magma"); a_rho.plot(cx, cy, "c+", ms=10)
        a_rho.add_patch(plt.Circle((cx, cy), r0, fill=False, color="cyan", lw=0.8))
        a_rho.set_ylabel(r["label"], fontsize=8); a_rho.set_xticks([]); a_rho.set_yticks([])
        if i == 0:
            a_rho.set_title("density (best plane)")
        a_cur.imshow(rho_s.T, origin="lower", cmap="gray_r")
        step = max(1, rho_s.shape[0] // 20)
        gx, gy = np.meshgrid(np.arange(0, u.shape[0], step), np.arange(0, u.shape[1], step), indexing="ij")
        a_cur.quiver(gx, gy, np.asarray(u[::step, ::step]), np.asarray(v[::step, ::step]), color="#1f77b4", width=0.005)
        a_cur.set_xticks([]); a_cur.set_yticks([])
        if i == 0:
            a_cur.set_title("in-plane current J")
        vmax = float(np.percentile(np.abs(vort[np.isfinite(vort)]), 99)) if np.isfinite(vort).any() else 1.0
        a_vort.imshow(np.nan_to_num(vort).T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        a_vort.plot(cx, cy, "k+", ms=10); a_vort.set_xticks([]); a_vort.set_yticks([])
        if i == 0:
            a_vort.set_title("signed pseudo-vorticity")
    fig.suptitle("Density / current / signed-vorticity through dominant node — N96 final fields", fontsize=12)
    return save_figure(fig, out)


def _fig_case_panel(r, klass, out):
    fig = plt.figure(figsize=(12, 4.6))
    if r.get("status") != "ok":
        fig.text(0.5, 0.5, f"{r['label']}\nnon-finite final field (blowup) — closure undefined\n"
                           f"field_used={r['field_used']} finite={r['finite_fraction']}", ha="center", va="center")
        return save_figure(fig, out)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1])
    rho_s = r["_rho_slice"]; vort = r["_vort"]; cx, cy, r0 = r["_cx"], r["_cy"], r["_r0"]
    ax0 = fig.add_subplot(gs[0]); ax0.imshow(rho_s.T, origin="lower", cmap="magma")
    ax0.add_patch(plt.Circle((cx, cy), r0, fill=False, color="cyan", lw=1.0)); ax0.plot(cx, cy, "c+")
    ax0.set_title("density"); ax0.set_xticks([]); ax0.set_yticks([])
    vmax = float(np.percentile(np.abs(vort[np.isfinite(vort)]), 99)) if np.isfinite(vort).any() else 1.0
    ax1 = fig.add_subplot(gs[1]); ax1.imshow(np.nan_to_num(vort).T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax1.add_patch(plt.Circle((cx, cy), r0, fill=False, color="k", lw=0.8)); ax1.set_title("signed vorticity"); ax1.set_xticks([]); ax1.set_yticks([])
    ax2 = fig.add_subplot(gs[2]); ax2.plot(r["_theta"], r["_w"], color=_class_color(klass)); ax2.axhline(0, color="k", lw=0.6)
    ax2.set_title("ring vorticity(theta)"); ax2.set_xlabel("theta")
    txt = (f"class={klass}\nclosure={r['current_closure_score']:.3f}  net_flux={r['net_flux_score']:+.3f}\n"
           f"tang_frac={r['tang_fraction']:.3f}  rad/tan={r['rad_tan_ratio']:.3f}\n"
           f"lobe_bal={r['lobe_balance']:+.3f}  dipole={r['dipole_alt']:.2f}  quad={r['quadrupole_alt']:.2f}\n"
           f"offset={r['node_current_offset_vox']}vox  n_nodes={r['n_nodes']}  r0={r['_r0']}")
    fig.text(0.5, -0.02, txt, ha="center", fontsize=8)
    fig.suptitle(r["label"], fontsize=12)
    return save_figure(fig, out)


# --------------------------------------------------------------------------- entry point
def render(stage1_root: str, *, outdir: str, overwrite: bool = False, manifest: str | None = None) -> list[str]:
    root = Path(stage1_root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    out = Path(outdir)
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    man_path = Path(manifest) if manifest else root / "stage1_manifest.json"
    classes: dict[str, str] = {}
    if man_path.exists():
        for r in load_json(man_path).get("runs", []):
            classes[r["key"]] = r.get("n96_class", "")

    keys = [k for k in CASE_ORDER if (root / k / "probe_data.npz").exists()]
    keys += [p.parent.name for p in root.glob("*/probe_data.npz") if p.parent.name not in keys]
    if not keys:
        raise FileNotFoundError(f"No <case>/probe_data.npz under {root}")

    recs = [_analyze_case(k, root / k / "probe_data.npz") for k in keys]

    # table
    cols = ["case", "label", "n96_class", "field_used", "finite_fraction", "status", "n_nodes", "rho_peak",
            "core_radius_vox", "net_flux_score", "rad_tan_ratio", "tang_fraction", "lobe_balance",
            "dipole_alt", "quadrupole_alt", "node_current_offset_vox", "node_current_offset_L",
            "current_closure_score"]
    table_rows = []
    for r in recs:
        row = {k: r.get(k, "") for k in cols}
        row["n96_class"] = classes.get(r["case"], "")
        table_rows.append(row)
    csv_path = out / "current_closure_score_table.csv"
    write_csv(csv_path, table_rows, cols)

    outputs = [str(csv_path)]
    outputs.append(str(_fig_ring_profiles(recs, classes, out / "ring_vorticity_profiles.png")))
    outputs.append(str(_fig_comparison(recs, classes, out / "current_closure_comparison.png")))
    outputs.append(str(_fig_slices(recs, classes, out / "density_current_vorticity_slices.png")))
    outputs.append(str(_fig_offset(recs, classes, out / "node_center_offset_summary.png")))
    outputs.append(str(_fig_axis_polarity(recs, classes, out / "axis_polarity_profiles.png")))

    panel_dir = out / "case_metric_panels"
    panel_dir.mkdir(exist_ok=True)
    for r in recs:
        outputs.append(str(_fig_case_panel(r, classes.get(r["case"], ""), panel_dir / f"{r['case']}.png")))

    return outputs
