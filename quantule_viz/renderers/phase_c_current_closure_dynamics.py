"""Time-resolved current-closure dynamics from N96 trace bundles (read-only, jax-free).

The static single-snapshot analysis was inconclusive because the instantaneous current of an
oscillating soliton is not a persistent-transport signal. This renderer reads the focused-4
trace bundles (frames.npz: psi [T,N,N,N] + times; diagnostic_summary.json: scalar morphology
trace) and computes:

  * per-snapshot closure metrics over time (net radial flux, tangential fraction, |J|);
  * the TIME-AVERAGED current field <J>(x) and its net radial flux  -> persistent leak vs closed
    (the discriminator the snapshot could not provide);
  * morphology-over-time from the diagnostic trace (mass, rho_peak, node_count, compactness);
  * K1 pre-blowup density timeline + blowup onset time.

No GPU, no PDE/solver/classifier/geometry/search change. Terminology stays geometric
(signed current, radial flux, closure) — not electric charge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..io import load_json, write_csv
from ..plots import CLASS_COLORS, as_density, density_slice, phase_current, save_figure
from .phase_c_current_closure import CASE_LABEL, _core_radius, _primary_node, _shell_metrics

TRACE_ORDER = ["k6_high_mass_true", "k6_mid_mass_true", "k6_near_threshold_near", "k1_low_mass_true"]


def _load_frames(case_dir: Path):
    fz = case_dir / "frames.npz"
    if not fz.exists():
        return None, None
    bundle = np.load(fz, allow_pickle=True)
    key = "psi" if "psi" in bundle.files else bundle.files[0]
    psi = np.asarray(bundle[key])
    times = np.asarray(bundle["times"]) if "times" in bundle.files else np.arange(len(psi))
    return psi, times


def _frame_metrics(psi: np.ndarray) -> dict | None:
    if not np.all(np.isfinite(psi)):
        return None
    field = psi.astype(np.complex128)
    rho = as_density(field)
    if rho.max() <= 0:
        return None
    c = _primary_node(rho)
    r0 = _core_radius(rho, c)
    jx, jy, jz = phase_current(field)
    shell = _shell_metrics(jx, jy, jz, c, r0)
    absj = float(np.mean(np.sqrt(jx ** 2 + jy ** 2 + jz ** 2))) / (float(rho.max()) + 1e-30)
    return {"net_flux": shell["net_flux_score"], "tang_fraction": shell["tang_fraction"],
            "absJ_norm": absj, "rho_peak": float(rho.max())}


def _time_averaged_flux(frames_finite: list[np.ndarray]) -> dict:
    """Average the current field over finite frames (lab frame), net radial flux on <J>."""
    jx_acc = jy_acc = jz_acc = None
    rho_acc = None
    for psi in frames_finite:
        f = psi.astype(np.complex128)
        gx, gy, gz = phase_current(f)
        rho = as_density(f)
        if jx_acc is None:
            jx_acc, jy_acc, jz_acc, rho_acc = gx * 0, gy * 0, gz * 0, rho * 0
        jx_acc = jx_acc + gx; jy_acc = jy_acc + gy; jz_acc = jz_acc + gz; rho_acc = rho_acc + rho
    n = len(frames_finite)
    jx, jy, jz, rho_mean = jx_acc / n, jy_acc / n, jz_acc / n, rho_acc / n
    c = _primary_node(rho_mean)
    r0 = _core_radius(rho_mean, c)
    shell = _shell_metrics(jx, jy, jz, c, r0)
    persistent_absJ = float(np.mean(np.sqrt(jx ** 2 + jy ** 2 + jz ** 2))) / (float(rho_mean.max()) + 1e-30)
    return {"persistent_net_flux": shell["net_flux_score"], "persistent_tang_fraction": shell["tang_fraction"],
            "persistent_absJ_norm": persistent_absJ, "n_avg_frames": n}


def _class_color(klass: str) -> str:
    return CLASS_COLORS.get(klass, "#666666")


def _analyze(case_dir: Path) -> dict:
    key = case_dir.name
    rec = {"case": key, "label": CASE_LABEL.get(key, key)}
    diag_path = case_dir / "diagnostic_summary.json"
    diag = load_json(diag_path) if diag_path.exists() else {}
    rec["n96_class"] = (diag.get("summary") or {}).get("diagnostic_label") or ((diag.get("candidate") or {}).get("klass"))
    psi, times = _load_frames(case_dir)
    if psi is None:
        rec["status"] = "no_frames"
        rec["_diag"] = diag
        return rec
    series_t, series = [], []
    finite_frames = []
    blowup_onset = None
    for i in range(len(psi)):
        m = _frame_metrics(psi[i])
        if m is None:
            if blowup_onset is None:
                blowup_onset = float(times[i])
            continue
        series_t.append(float(times[i])); series.append(m); finite_frames.append(psi[i])
    rec["n_finite_frames"] = len(finite_frames)
    rec["n_total_frames"] = int(len(psi))
    rec["blowup_onset_time"] = blowup_onset
    rec["_series_t"] = np.asarray(series_t)
    rec["_series"] = series
    rec["_diag"] = diag
    if len(finite_frames) >= 2:
        # use the late half (settled window) for persistent average
        half = len(finite_frames) // 2
        ta = _time_averaged_flux(finite_frames[half:])
        rec.update(ta)
        nf = np.array([s["net_flux"] for s in series])
        rec["mean_net_flux"] = float(np.mean(nf)); rec["std_net_flux"] = float(np.std(nf))
        rec["mean_tang_fraction"] = float(np.mean([s["tang_fraction"] for s in series]))
        rec["status"] = "ok"
    else:
        rec["status"] = "insufficient_finite_frames"
    return rec


def _fig_timeseries(recs, out):
    ok = [r for r in recs if r.get("status") == "ok"]
    if not ok:
        return None
    fig, axes = plt.subplots(len(ok), 3, figsize=(13, 3.0 * len(ok)), squeeze=False)
    for i, r in enumerate(ok):
        t = r["_series_t"]; series = r["_series"]; col = _class_color(r.get("n96_class", ""))
        a0, a1, a2 = axes[i]
        a0.plot(t, [s["net_flux"] for s in series], color=col, marker=".", ms=3)
        a0.axhline(0, color="k", lw=0.6); a0.set_ylim(-1.05, 1.05)
        a0.set_ylabel(r["label"], fontsize=8)
        if i == 0:
            a0.set_title("per-frame net radial flux (t)")
        a1.plot(t, [s["tang_fraction"] for s in series], color=col, marker=".", ms=3)
        a1.set_ylim(0, 1.05)
        if i == 0:
            a1.set_title("tangential fraction (t)")
        a2.semilogy(t, [s["absJ_norm"] for s in series], color=col, marker=".", ms=3)
        if i == 0:
            a2.set_title("|J|/rho_peak (t, log)")
        for a in (a0, a1, a2):
            a.tick_params(labelsize=7); a.set_xlabel("t", fontsize=7)
    fig.suptitle("Time-resolved current-closure metrics — N96 trace bundles", fontsize=12)
    return save_figure(fig, out)


def _fig_morphology(recs, out):
    ok = [r for r in recs if r.get("_diag", {}).get("trace")]
    if not ok:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for r in ok:
        diag = r["_diag"]; tr = diag["trace"]; tt = diag.get("times", list(range(len(tr))))
        col = _class_color(r.get("n96_class", ""))
        def g(name):
            return [d.get(name, np.nan) for d in tr]
        axes[0].plot(tt, g("mass"), color=col, label=r["label"])
        axes[1].plot(tt, g("rho_peak"), color=col)
        axes[2].plot(tt, g("node_count"), color=col)
    axes[0].set_title("mass(t)"); axes[1].set_title("rho_peak(t)"); axes[2].set_title("node_count(t)")
    for a in axes:
        a.set_xlabel("t"); a.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle("Morphology-over-time from diagnostic trace — N96", fontsize=12)
    return save_figure(fig, out)


def _fig_spindown_vs_survivor(recs, out):
    by = {r["case"]: r for r in recs}
    hi, mid = by.get("k6_high_mass_true"), by.get("k6_mid_mass_true")
    if not (hi and mid and hi.get("status") == "ok" and mid.get("status") == "ok"):
        return None
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for r, name in ((hi, "K6 high-mass (spin-down)"), (mid, "K6 mid-mass (survivor)")):
        col = _class_color(r.get("n96_class", ""))
        ax[0].plot(r["_series_t"], [s["net_flux"] for s in r["_series"]], color=col, marker=".", label=name)
        ax[0].axhline(r["persistent_net_flux"], color=col, ls="--", lw=1,
                      label=f"{name.split()[1]} persistent <J> flux={r['persistent_net_flux']:+.3f}")
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_ylim(-1.05, 1.05)
    ax[0].set_title("net radial flux(t) + persistent time-averaged <J> flux"); ax[0].set_xlabel("t")
    ax[0].legend(fontsize=7)
    for r, name in ((hi, "K6 high-mass"), (mid, "K6 mid-mass")):
        diag = r["_diag"]
        if diag.get("trace"):
            tt = diag.get("times", range(len(diag["trace"])))
            ax[1].plot(tt, [d.get("mass", np.nan) for d in diag["trace"]],
                       color=_class_color(r.get("n96_class", "")), label=f"{name} mass(t)")
    ax[1].set_title("mass(t): does the spin-down case shed mass?"); ax[1].set_xlabel("t"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.25)
    fig.suptitle("Spin-down vs survivor: persistent current + mass evolution (N96)", fontsize=12)
    return save_figure(fig, out)


def _fig_density_timelines(recs, root: Path, out):
    ok = [r for r in recs if (root / r["case"] / "frames.npz").exists()]
    if not ok:
        return None
    ncols = 5
    fig, axes = plt.subplots(len(ok), ncols, figsize=(2.5 * ncols, 2.6 * len(ok)), squeeze=False)
    for i, r in enumerate(ok):
        psi, times = _load_frames(root / r["case"])
        finite_idx = [j for j in range(len(psi)) if np.all(np.isfinite(psi[j])) and as_density(psi[j].astype(np.complex128)).max() > 0]
        picks = finite_idx if len(finite_idx) <= ncols else [finite_idx[k] for k in np.linspace(0, len(finite_idx) - 1, ncols).astype(int)]
        for cidx in range(ncols):
            ax = axes[i][cidx]
            ax.set_xticks([]); ax.set_yticks([])
            if cidx < len(picks):
                j = picks[cidx]
                slc, _, _ = density_slice(psi[j].astype(np.complex128))
                ax.imshow(slc.T, origin="lower", cmap="magma")
                ax.set_title(f"t={float(times[j]):.0f}", fontsize=7)
            if cidx == 0:
                ax.set_ylabel(r["label"], fontsize=7)
        # annotate blowup
        if r.get("blowup_onset_time") is not None:
            axes[i][ncols - 1].text(0.5, -0.18, f"blowup onset t≈{r['blowup_onset_time']:.0f}",
                                    transform=axes[i][ncols - 1].transAxes, ha="center", color="red", fontsize=7)
    fig.suptitle("Density-slice timelines (finite frames) — N96 trace", fontsize=12)
    return save_figure(fig, out)


def render(trace_root: str, *, outdir: str, overwrite: bool = False) -> list[str]:
    root = Path(trace_root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    out = Path(outdir)
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    keys = [k for k in TRACE_ORDER if (root / k / "frames.npz").exists() or (root / k / "diagnostic_summary.json").exists()]
    keys += [p.parent.name for p in root.glob("*/diagnostic_summary.json") if p.parent.name not in keys]
    if not keys:
        raise FileNotFoundError(f"No <case>/frames.npz or diagnostic_summary.json under {root}")

    recs = []
    for k in keys:
        r = _analyze(root / k)
        r["out_dir"] = str(root / k)
        recs.append(r)

    cols = ["case", "label", "n96_class", "status", "n_total_frames", "n_finite_frames", "blowup_onset_time",
            "mean_net_flux", "std_net_flux", "mean_tang_fraction",
            "persistent_net_flux", "persistent_tang_fraction", "persistent_absJ_norm", "n_avg_frames"]
    rows = [{c: r.get(c, "") for c in cols} for r in recs]
    csv_path = out / "time_averaged_closure_table.csv"
    write_csv(csv_path, rows, cols)

    outputs = [str(csv_path)]
    for fn, fig in (("closure_dynamics_timeseries.png", _fig_timeseries(recs, out / "closure_dynamics_timeseries.png")),
                    ("morphology_over_time.png", _fig_morphology(recs, out / "morphology_over_time.png")),
                    ("spindown_vs_survivor_comparison.png", _fig_spindown_vs_survivor(recs, out / "spindown_vs_survivor_comparison.png")),
                    ("density_slice_timelines.png", _fig_density_timelines(recs, root, out / "density_slice_timelines.png"))):
        if fig is not None:
            outputs.append(str(fig))
    return outputs
