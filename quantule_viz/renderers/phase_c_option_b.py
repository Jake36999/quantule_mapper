"""Phase C Option B v2 visual-analysis renderer.

Adds clearer, comparison-oriented figures on top of the v1 structured pack:
- population/morphology_node_family_map.png  (TRUE count + dominant node family + role + resolution risk)
- population/true_rate_heatmap_clean.png     (reuses the clean v1 heatmap, captioned N=48/T=4000 discovery)
- branch_comparison/branch_representative_comparison.png (8 representatives, matched rows)
- n96_shortlist/n96_shortlist_inspection_sheet.{csv,png}
- cases/<case>/{density_slices_timeline,current_vorticity_panel,node_track_overlay,scalar_trace_panel}.png

Read-only and JAX-free. Reuses quantule_viz.plots + quantule_viz.renderers.phase_c_structured.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ..io import guard_outputs, load_json, load_npz, write_csv
from ..plots import (
    CLASS_COLORS,
    TRUE_CLASS,
    as_density,
    blank_figure,
    density_slice,
    find_component_centroids,
    save_figure,
    vector_slice,
)
from . import phase_c_structured as v1

# raw target mass scales by the grid-volume ratio across resolutions: (dx48/dx96)^3 = (96/48)^3 = 8.
DX48 = 10.0 / 48.0
DX96 = 10.0 / 96.0
N96_MASS_SCALE = (DX48 ** 3) / (DX96 ** 3)  # == 8.0
HIGH_K_RISK_THRESHOLD = 0.015               # shortlist: feb/K6/K4 ~0.008-0.013 (low); K2 0.022, K1lo 0.039 (high)

COMPARISON_ORDER = [
    "k1_low_mass_true",
    "k1_high_mass_failure",
    "k2_intermediate_true",
    "k4_intermediate_true",
    "k6_mid_mass_true",
    "k6_high_mass_true",
    "k6_near_threshold_near",
    "feb56dc7_control",
]

EXPECTED_VALIDATION_ROLE = {
    "k6_high_mass_true": "K6 high-mass distributed branch",
    "k6_mid_mass_true": "K6 mid-mass distributed branch",
    "k4_intermediate_true": "K4 intermediate distributed branch",
    "k2_intermediate_true": "K2 compact intermediate branch / resolution-risk contrast",
    "k1_low_mass_true": "K1 fragile low-mass control / resolution-risk contrast",
    "k1_high_mass_failure": "K1 high-mass failure control",
    "k6_near_threshold_near": "K6 near-threshold inconclusive branch",
    "feb56dc7_control": "feb56dc7 anchor control",
}
INSPECTION_PRIORITY = [
    "k6_high_mass_true", "k6_mid_mass_true", "k4_intermediate_true", "feb56dc7_control",
    "k2_intermediate_true", "k1_low_mass_true", "k1_high_mass_failure", "k6_near_threshold_near",
]
CASE_DYNAMICS = [
    "density_slices_timeline.png", "current_vorticity_panel.png",
    "node_track_overlay.png", "scalar_trace_panel.png",
]


def _role_for_cell(kval: int, mass: float, has_true: bool, dominant: str, masses: list[float]) -> str:
    if has_true:
        return {1: "K1 fragile pocket", 2: "K2 compact", 3: "K3 weak",
                4: "K4 distributed", 6: "K6 distributed"}.get(kval, f"K{kval}")
    if kval == 1 and mass >= 1200:
        return "K1 failure wall"
    return ""


def _resolution_risk(high_k: float, kval: int) -> str:
    if np.isfinite(high_k):
        return "HIGH" if high_k > HIGH_K_RISK_THRESHOLD else "low"
    return "HIGH" if kval in (1, 2, 3) else "low"  # K-based heuristic when high_k absent (population cells)


def _morphology_node_family_map(rows: list[dict[str, Any]], outpath: Path) -> Path:
    ks, masses, _ = v1._grid_axes(rows)
    rgb = np.ones((len(ks), len(masses), 3))
    text: list[list[str]] = [["" for _ in masses] for _ in ks]
    for i, kval in enumerate(ks):
        distributed = kval in (4, 6)
        for j, mass in enumerate(masses):
            bucket = [r for r in rows if r["K_int"] == kval and abs(r["mass_float"] - mass) < 1e-9]
            total = len(bucket) or 1
            true_rows = [r for r in bucket if r.get("klass") == TRUE_CLASS and r["n_fin_int"] >= 0]
            n_true = len(true_rows)
            frac = n_true / total
            if n_true:
                # distributed -> green family; compact -> orange family; intensity by TRUE fraction
                base = np.array([0.10, 0.60, 0.30]) if distributed else np.array([0.95, 0.55, 0.10])
                rgb[i, j] = 1.0 - frac * (1.0 - base)
                node_txt = v1._true_node_text(bucket)
                role = _role_for_cell(kval, mass, True, "", masses)
                risk = "" if distributed else "  [risk]"
                text[i][j] = f"T{n_true}/{total}\nn_fin {node_txt}\n{role}{risk}"
            else:
                dominant = v1._dominant_class(bucket) if bucket else ""
                short = v1.CLASS_SHORT.get(dominant, "")
                role = _role_for_cell(kval, mass, False, dominant, masses)
                if role == "K1 failure wall":
                    rgb[i, j] = np.array([0.96, 0.80, 0.80])
                    text[i][j] = f"{short}\nK1 wall"
                else:
                    rgb[i, j] = np.array([0.92, 0.92, 0.92])
                    text[i][j] = short
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.imshow(rgb, aspect="auto", origin="upper")
    ax.set_xticks(range(len(masses))); ax.set_xticklabels([v1._mass_label(m) for m in masses], rotation=30, ha="right")
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel("raw target mass (N=48)"); ax.set_ylabel("initial blob count K")
    ax.set_title("Morphology / node-family map  (green=distributed low-risk, orange=compact higher-risk, red=K1 failure wall)\nN=48 / T=4000 discovery — NOT N96 validation")
    for i in range(len(ks)):
        for j in range(len(masses)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=7.5)
    return save_figure(fig, outpath)


def _last_finite_frame(frames: np.ndarray) -> np.ndarray:
    for t in range(frames.shape[0] - 1, -1, -1):
        if np.all(np.isfinite(np.abs(frames[t]))):
            return frames[t]
    return frames[-1]


def _branch_comparison(case_root: Path, outpath: Path) -> Path:
    cols = []
    for key in COMPARISON_ORDER:
        cdir = case_root / key
        if not (cdir / "diagnostic_summary.json").exists() or not (cdir / "frames.npz").exists():
            continue
        payload = load_json(cdir / "diagnostic_summary.json")
        bundle = load_npz(cdir / "frames.npz")
        fkey = "psi" if "psi" in bundle.files else ("frames" if "frames" in bundle.files else bundle.files[0])
        frames = np.asarray(bundle[fkey])
        if frames.ndim == 3:
            frames = frames[None, ...]
        cols.append((key, payload, frames))
    if not cols:
        return blank_figure("No case bundles found for comparison.", outpath, title="Branch comparison")
    ncol = len(cols)
    fig, axes = plt.subplots(4, ncol, figsize=(2.7 * ncol, 11.2), squeeze=False)
    for c, (key, payload, frames) in enumerate(cols):
        spec = v1.CASE_SPECS.get(key, {"label": key, "role": ""})
        summ = payload.get("summary", {}); cand = payload.get("candidate", {})
        last = _last_finite_frame(frames)
        # row 0: late density slice
        rho2, axis, center = density_slice(last)
        ax = axes[0][c]; rho2 = np.asarray(rho2, float)
        vmax = float(np.percentile(rho2, 99.5)) or 1.0
        ax.imshow(rho2.T, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{spec['label']}\nK={cand.get('K')} {cand.get('klass','')[:4]}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0: ax.set_ylabel("late ρ slice", fontsize=9)
        # row 1: vector / vorticity
        ax = axes[1][c]
        try:
            rs, u, v, vort = vector_slice(last.astype(np.complex128))
            vort = np.nan_to_num(np.asarray(vort, float)); vl = float(np.percentile(np.abs(vort), 99)) or 1.0
            ax.imshow(vort.T, origin="lower", cmap="coolwarm", vmin=-vl, vmax=vl)
            st = max(1, rs.shape[0] // 14)
            xx, yy = np.meshgrid(np.arange(0, rs.shape[0], st), np.arange(0, rs.shape[1], st), indexing="ij")
            uu = np.nan_to_num(np.asarray(u[::st, ::st], float)); vv = np.nan_to_num(np.asarray(v[::st, ::st], float))
            ax.quiver(xx, yy, uu, vv, color="0.1", width=0.005, scale_units="xy")
        except Exception:
            ax.text(0.5, 0.5, "no coherent\nvector field", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0: ax.set_ylabel("J + vorticity", fontsize=9)
        # row 2: scalar timeline (er + node count)
        ax = axes[2][c]
        try:
            arr = v1._trace_arrays(payload); t = arr["times"]
            ax.plot(t, arr["energy_ratio"], color="tab:blue", lw=1.4)
            ax.set_ylim(0, max(2.2, float(np.nanmax(arr["energy_ratio"])) * 1.1))
            ax2 = ax.twinx(); ax2.plot(t, arr["node_count"], color="tab:red", lw=1.2, alpha=0.8)
            ax2.set_ylim(0, max(7, float(np.nanmax(arr["node_count"])) + 1)); ax2.tick_params(labelsize=6)
        except Exception:
            ax.text(0.5, 0.5, "no trace", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.tick_params(labelsize=6); ax.grid(alpha=0.25)
        if c == 0: ax.set_ylabel("er(blue)/nodes(red)", fontsize=8)
        # row 3: summary text
        ax = axes[3][c]; ax.set_axis_off()
        hk = v1._safe_float(summ.get("high_k_fraction_max"))
        risk = _resolution_risk(hk, v1._safe_int(cand.get("K")))
        txt = (f"class {cand.get('klass','')[:18]}\ndiag {summ.get('diagnostic_label','')[:20]}\n"
               f"n_fin {summ.get('node_count_last')}\nhigh_k {hk:.3g}\nrisk {risk}\n{spec.get('role','')}")
        ax.text(0.02, 0.95, txt, va="top", family="monospace", fontsize=7.2)
    fig.suptitle("Branch representative comparison — distributed vs compact morphology (N=48/T=4000 discovery)", fontsize=12)
    return save_figure(fig, outpath, dpi=140)


def _n96_inspection_sheet(shortlist_table: list[dict[str, Any]], out_csv: Path, out_png: Path) -> list[Path]:
    order = {k: i for i, k in enumerate(INSPECTION_PRIORITY)}
    rows = sorted(shortlist_table, key=lambda r: order.get(r.get("case_key"), 99))
    out_rows = []
    for r in rows:
        raw = v1._safe_float(r.get("raw_target_mass"))
        scaled = raw * N96_MASS_SCALE if np.isfinite(raw) else float("nan")
        hk = v1._safe_float(r.get("high_k_fraction"))
        out_rows.append({
            "candidate_label": r["candidate_label"], "source_run": r["source_run"], "idx": r["idx"],
            "K": r["K"], "raw_N48_target": ("" if not np.isfinite(raw) else round(raw, 6)),
            "dx_weighted_mass": r.get("dx_weighted_target_mass"),
            "scaled_N96_raw_target": ("" if not np.isfinite(scaled) else round(scaled, 6)),
            "ic_seed": r["ic_seed"], "class": r["class"], "diagnostic_label": r["diagnostic_label"],
            "final_node_count": r["final_node_count"], "late_slope": r["late_slope"],
            "high_k_fraction": r["high_k_fraction"], "compactness": r["compactness"], "core_radius": r["core_radius"],
            "resolution_risk": _resolution_risk(hk, v1._safe_int(r.get("K"))),
            "expected_validation_role": EXPECTED_VALIDATION_ROLE.get(r.get("case_key"), ""),
        })
    fields = list(out_rows[0].keys())
    write_csv(out_csv, out_rows, fields)
    # table image
    show_cols = ["candidate_label", "K", "raw_N48_target", "scaled_N96_raw_target", "ic_seed",
                 "class", "final_node_count", "high_k_fraction", "resolution_risk", "expected_validation_role"]
    cell = [[str(r.get(c, "")) for c in show_cols] for r in out_rows]
    fig, ax = plt.subplots(figsize=(17, 0.6 * len(out_rows) + 1.6)); ax.set_axis_off()
    tbl = ax.table(cellText=cell, colLabels=[c.replace("_", "\n") for c in show_cols], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.5)
    for ci, r in enumerate(out_rows):
        color = CLASS_COLORS.get(str(r["class"]), "#ffffff")
        for cj in range(len(show_cols)):
            tbl[ci + 1, cj].set_facecolor((*_hex_to_rgb(color), 0.18))
        if str(r["resolution_risk"]) == "HIGH":
            tbl[ci + 1, show_cols.index("resolution_risk")].set_facecolor((1.0, 0.85, 0.5))
    ax.set_title("N96/T6000 shortlist inspection sheet — scaled raw target = raw_N48 × 8  (validate distributed/low-risk first)", fontsize=11)
    return [out_csv, save_figure(fig, out_png, dpi=150)]


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _case_dynamics(read_dir: Path, write_dir: Path, label: str, overwrite: bool) -> list[Path]:
    write_dir.mkdir(parents=True, exist_ok=True)
    guard_outputs(write_dir, CASE_DYNAMICS, overwrite)
    payload, frames = v1._load_case_result(read_dir)
    case_dir = write_dir
    out = []
    # density slices timeline
    picks = sorted(set(np.linspace(0, frames.shape[0] - 1, min(5, frames.shape[0])).astype(int)))
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.4), squeeze=False)
    for ax, fi in zip(axes[0], picks):
        rho2, axis, center = density_slice(frames[fi]); rho2 = np.asarray(rho2, float)
        vmax = float(np.percentile(rho2, 99.5)) or 1.0
        ax.imshow(rho2.T, origin="lower", cmap="magma", vmin=0, vmax=vmax); ax.set_title(f"frame {fi}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{label} — density slices over time", fontsize=11)
    out.append(save_figure(fig, case_dir / "density_slices_timeline.png"))
    # current + vorticity panel (last finite frame)
    last = _last_finite_frame(frames)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    try:
        rs, u, v, vort = vector_slice(last.astype(np.complex128)); rs = np.asarray(rs, float)
        ax[0].imshow(rs.T, origin="lower", cmap="magma"); ax[0].set_title("density");
        ax[1].imshow(rs.T, origin="lower", cmap="gray_r")
        st = max(1, rs.shape[0] // 16); xx, yy = np.meshgrid(np.arange(0, rs.shape[0], st), np.arange(0, rs.shape[1], st), indexing="ij")
        ax[1].quiver(xx, yy, np.nan_to_num(u[::st, ::st]), np.nan_to_num(v[::st, ::st]), color="tab:blue", width=0.004)
        ax[1].set_title("current J (quiver)")
        vort = np.nan_to_num(np.asarray(vort, float)); vl = float(np.percentile(np.abs(vort), 99)) or 1.0
        ax[2].imshow(vort.T, origin="lower", cmap="coolwarm", vmin=-vl, vmax=vl); ax[2].set_title("pseudo-vorticity")
    except Exception:
        for a in ax: a.text(0.5, 0.5, "no coherent\nvector field", ha="center", va="center", transform=a.transAxes)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{label} — current / vorticity (late frame)", fontsize=11)
    out.append(save_figure(fig, case_dir / "current_vorticity_panel.png"))
    # node track overlay
    fig, ax = plt.subplots(figsize=(5.6, 5.4)); N = frames.shape[1]
    cmap = plt.cm.viridis(np.linspace(0, 1, frames.shape[0]))
    for ti in range(frames.shape[0]):
        for cpt in find_component_centroids(frames[ti]):
            ax.scatter(cpt[0], cpt[1], color=cmap[ti], s=18)
    ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_aspect("equal")
    ax.set_title(f"{label} — node positions x-y (colour=frame)")
    out.append(save_figure(fig, case_dir / "node_track_overlay.png"))
    # scalar trace panel
    try:
        arr = v1._trace_arrays(payload); t = arr["times"]
        fig, ax = plt.subplots(2, 3, figsize=(13, 6.4))
        for a, key, ttl in zip(ax.flat,
                               ["energy_ratio", "node_count", "compactness", "high_k_fraction", "rho_peak", "core_radius"],
                               ["mass / initial", "node count", "compactness", "high-k fraction", "peak density", "core radius"]):
            a.plot(t, arr[key], lw=1.5); a.set_title(ttl); a.set_xlabel("time"); a.grid(alpha=0.25)
        fig.suptitle(f"{label} — scalar traces", fontsize=11)
        out.append(save_figure(fig, case_dir / "scalar_trace_panel.png"))
    except Exception:
        out.append(blank_figure("no trace data", case_dir / "scalar_trace_panel.png", title=label))
    return out


def render(summary_csv: str | None, *, outdir: str, overwrite: bool,
           shortlist: str | None = None, cases_root: str | None = None) -> list[Path]:
    summary_path = v1._resolve_path(summary_csv, v1.DEFAULT_SUMMARY_CSV)
    shortlist_path = v1._resolve_path(shortlist, v1.DEFAULT_SHORTLIST_JSON)
    out = v1._resolve_path(outdir, Path(outdir))
    pop = out / "population"; bc = out / "branch_comparison"; ins = out / "n96_shortlist"; cases = v1._resolve_path(cases_root, out / "cases")
    for d in (pop, bc, ins):
        d.mkdir(parents=True, exist_ok=True)
    guard_outputs(pop, ["morphology_node_family_map.png", "true_rate_heatmap_clean.png"], overwrite)
    guard_outputs(bc, ["branch_representative_comparison.png"], overwrite)
    guard_outputs(ins, ["n96_shortlist_inspection_sheet.csv", "n96_shortlist_inspection_sheet.png"], overwrite)

    rows = v1._structured_rows(summary_path)
    shortlist_rows = v1._shortlist_rows(shortlist_path)
    shortlist_table = v1._shortlist_table_rows(shortlist_rows, v1._summary_lookup(rows))

    outputs = [
        _morphology_node_family_map(rows, pop / "morphology_node_family_map.png"),
        v1._plot_true_rate_heatmap(rows, pop / "true_rate_heatmap_clean.png"),
        _branch_comparison(cases, bc / "branch_representative_comparison.png"),
    ]
    outputs.extend(_n96_inspection_sheet(shortlist_table, ins / "n96_shortlist_inspection_sheet.csv",
                                         ins / "n96_shortlist_inspection_sheet.png"))
    for key in v1.CASE_ORDER:
        cdir = cases / key
        if (cdir / "diagnostic_summary.json").exists() and (cdir / "frames.npz").exists():
            outputs.extend(_case_dynamics(cdir, out / "cases" / key,
                                          v1.CASE_SPECS.get(key, {}).get("label", key), overwrite))
    return outputs
