#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Universal field renderer — one view library, no per-campaign code.

WHY THIS SHAPE (docs/VISUAL_HUD_SCOPE_RFC.md). The existing renderers under quantule_viz/renderers/
are keyed on the CAMPAIGN rather than the VIEW - five phase_c* variants, 3,622 lines - so every new
experiment needed a new renderer, and TG/C2/C3 got none at all. Meanwhile 1,410 field arrays sit in
sweep_runs/ that nothing has ever displayed.

The audit that made this cheap: across all 670 packs, **every field satisfies one structural rule -
an array whose last three dimensions form a cube** - and a dtype/prefix classifier resolves every
key name without a lookup table. So substrate knowledge lives in ~20 lines, not in a renderer per
sector.

TWO INPUTS, ONE PIPELINE
  * legacy field packs  - arbitrary .npz, fields found by the cube rule
  * HUD snapshots       - snap_*.npz written by jax_scout/snapshots.py, using the
                          '<field>__<plane>' convention, rendered as a time montage

OUTPUT goes to <run_dir>/rendered/ so tools/build_run_catalogue.py imports it like any other
run-produced figure - one path into the vault, not two.

MEMORY. The largest single array in the corpus is 856 MB (a (121,96,96,96) time series). npz loads
lazily per key, so keys are read one at a time, reduced immediately, and freed. Anything above
--max-array-mb is skipped with a note rather than swapping the machine.

Usage:
    .venv/Scripts/python.exe tools/render_fields.py --run TG_B2_MIDPLANE_FLUX_N64
    .venv/Scripts/python.exe tools/render_fields.py --snapshots sweep_runs/<run>/snapshots/well
    .venv/Scripts/python.exe tools/render_fields.py --all --limit 20
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(REPO, "sweep_runs")

PLANE_RE = re.compile(r"^(?P<name>.+)__(?P<plane>xy|xz|yz|vol)$")

# ------------------------------------------------------------------ classification


def classify(name, arr):
    """Semantic class from dtype and name. Probed against every odd key in the corpus
    (psi_1194, fields, rho_hist_*, Pi, current) - no lookup table required."""
    if np.iscomplexobj(arr):
        return "wavefunction"
    if name.startswith("R_"):
        return "resolution"
    return "scalar"


def is_cube(a):
    return a.ndim in (3, 4) and a.shape[-1] == a.shape[-2] == a.shape[-3]


def centre_plane(a):
    """Reduce a cube (or a time series of cubes) to a single 2-D centre plane."""
    if a.ndim == 4:
        a = a[a.shape[0] // 2]
    return a[:, :, a.shape[-1] // 2]


# ------------------------------------------------------------------ views


def draw(ax, arr, name, cls, *, title=None):
    """Render one 2-D array according to its class. Returns the mappable for a colourbar."""
    if cls == "wavefunction":
        rho = np.abs(arr) ** 2
        im = ax.imshow(rho.T, origin="lower", cmap="magma")
        ax.set_title(title or f"{name}  |ψ|²", fontsize=8)
    elif cls == "resolution":
        im = ax.imshow(np.real(arr).T, origin="lower", cmap="viridis")
        ax.set_title(title or name, fontsize=8)
    else:
        v = np.real(arr)
        lim = float(np.max(np.abs(v))) or 1.0
        im = ax.imshow(v.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(title or name, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def draw_phase(ax, arr, name):
    """Phase, masked by density so vacuum phase noise does not dominate the eye.

    The mask matters: unmasked phase in near-zero-density regions is uniform noise and hides the
    structure that carries the physics.
    """
    rho = np.abs(arr) ** 2
    m = rho > 0.02 * (rho.max() or 1.0)
    ph = np.angle(arr)
    ph = np.where(m, ph, np.nan)
    im = ax.imshow(ph.T, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax.set_title(f"{name}  arg ψ (masked)", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def find_centroids(arr, *, max_nodes=6, rel_thresh=0.55, max_blobs=8):
    """HUD item 6.3 — node centroids via a STANDARD detector, not a bespoke peak-tracker.

    Why this specific choice. C2.8b was caused by a hand-rolled peak-tracker that failed when two
    cores overlapped, producing an elasticity of 3.21 and violating energy conservation. The RFC
    named the centroid overlay as the item most likely to have made that visible. Writing another
    bespoke tracker to do it would reproduce the exact failure class.

    `skimage.feature.peak_local_max` is the standard, maintained, sub-pixel-capable alternative.
    `min_distance` is what handles the overlapping-core case that broke the original: peaks closer
    than that are merged rather than reported as two, which is the honest answer when cores merge.

    Known limitation, stated rather than tuned away. On a heavily overlapped pair the detector
    reports n=2 early and n=1 once the cores merge, and on ring-structured fields (`pi`, `phi`)
    it can latch onto ring maxima and report n=3-4. Both are surfaced by the count-instability
    banner in `render_snapshots`, which is the intended behaviour: a flickering count IS the
    C2.8b signature, so it is reported in red rather than smoothed into a plausible-looking
    trajectory. Treat the overlay as a check on node observables, never as one.

    Returns [(row, col, weight)] in array index order, or [] if scikit-image is unavailable -- the
    overlay degrades rather than failing the render.
    """
    try:
        from skimage.feature import peak_local_max  # noqa: PLC0415
    except Exception:
        return []
    d = np.abs(arr) ** 2 if np.iscomplexobj(arr) else np.abs(np.real(arr))
    peak = float(d.max())
    if not np.isfinite(peak) or peak <= 0:
        return []
    # A speckle field has no nodes, so naming six of its specks is a lie the overlay must not
    # tell. Discriminator, measured on the corpus rather than guessed: fragment the
    # above-threshold mask and count components. A localised core gives 1-3; noise gives 45-67.
    try:
        from scipy import ndimage  # noqa: PLC0415
    except ImportError:            # narrow on purpose: a blanket except here once hid a NameError
        ndimage = None
    blobs = 1 if ndimage is None else int(ndimage.label(d >= rel_thresh * peak)[1])
    if blobs > max_blobs:
        return []
    try:
        # min_distance must reflect the CORE scale, not the grid scale. Too small and a
        # flat-topped overlapping pair reports phantom peaks on its own plateau -- which is the
        # C2.8b failure reproduced in the detector meant to reveal it.
        #
        # exclude_border=False is REQUIRED. The default excludes a min_distance-wide margin, so a
        # node that has drifted near the boundary silently vanishes from the overlay -- and a node
        # approaching the boundary is exactly when you most want to see it.
        pk = peak_local_max(d, min_distance=max(3, d.shape[0] // 8),
                            threshold_abs=rel_thresh * peak, num_peaks=max_nodes,
                            exclude_border=False)
    except Exception:
        return []
    return [(int(r), int(c), float(d[r, c] / peak)) for r, c in pk]


def overlay_axes(ax, arr, *, midplane=True, mask_edge=True, centroids=True):
    """The HUD part: mark the structures the observables are defined against.

    The midplane x=0 and the mask interface x=+dx/2 are exactly the surfaces the P2 momentum ledger
    integrates over, and getting their placement wrong opened that ledger at O(1). Drawing them
    makes a placement error visible instead of arithmetic.

    Centroids (6.3) mark where a node-tracking observable would say the nodes are -- so a tracker
    that has jumped, merged or latched onto a lobe is visible at a glance rather than inferred from
    an impossible scalar.
    """
    n = arr.shape[0] if hasattr(arr, "shape") else int(arr)
    if midplane:
        ax.axvline(n // 2, color="cyan", lw=0.7, ls="-", alpha=0.75)
    if mask_edge:
        ax.axvline(n // 2 + 0.5, color="cyan", lw=0.6, ls=":", alpha=0.6)
    if centroids and hasattr(arr, "shape"):
        pts = find_centroids(arr)
        for r, c, w in pts:
            # images are drawn transposed (imshow(arr.T)), so array (r,c) -> plot (r,c) directly
            ax.plot(r, c, marker="+", color="#39ff14", ms=7 + 5 * w, mew=1.3, alpha=0.95)
        if len(pts) >= 2:
            (r0, c0, _), (r1, c1, _) = pts[0], pts[1]
            ax.plot([r0, r1], [c0, c1], color="#39ff14", lw=0.7, ls="--", alpha=0.6)
            ax.text(0.02, 0.02, "sep=%.1f px  n=%d" % (np.hypot(r1 - r0, c1 - c0), len(pts)),
                    transform=ax.transAxes, fontsize=6, color="#39ff14", va="bottom")
        elif len(pts) == 1:
            ax.text(0.02, 0.02, "n=1", transform=ax.transAxes, fontsize=6,
                    color="#39ff14", va="bottom")


# ------------------------------------------------------------------ legacy packs


def pack_stem(npz_path, run_dir=None):
    """A filesystem-safe montage name that is unique WITHIN the run.

    Runs nest packs in subdirectories and reuse names across them (a `fields.npz` per case
    directory), so a basename-derived output silently overwrites. In a corpus-wide pass that
    yields montages quietly showing the wrong pack, and an image that lies about its source is
    worse than no image at all.
    """
    rel = os.path.relpath(npz_path, run_dir) if run_dir else os.path.basename(npz_path)
    return re.sub(r"[^\w.-]", "_", rel)[:-4][:120]


def render_pack(npz_path, outdir, *, max_mb=1500, dpi=110, stem=None):
    """Render every field in one arbitrary .npz as a single montage."""
    made = []
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            keys = list(z.files)
            panels = []
            for k in keys:
                try:
                    shp = z[k].shape if False else None  # noqa: F841  (kept lazy below)
                except Exception:
                    continue
                a = z[k]
                if not is_cube(a):
                    del a
                    continue
                if a.nbytes / 1e6 > max_mb:
                    panels.append((k, None, "skipped: %.0f MB > --max-array-mb" % (a.nbytes / 1e6)))
                    del a
                    gc.collect()
                    continue
                plane = centre_plane(a)
                panels.append((k, np.array(plane), None))
                del a, plane
                gc.collect()
    except Exception as e:  # noqa: BLE001
        return [], "unreadable: %s" % e

    real = [p for p in panels if p[1] is not None]
    if not real:
        return [], "no renderable fields"

    # a complex field earns two panels (density and masked phase)
    slots = []
    for name, arr, _ in real:
        cls = classify(name, arr)
        slots.append((name, arr, cls, "field"))
        if cls == "wavefunction":
            slots.append((name, arr, cls, "phase"))

    ncol = min(4, len(slots))
    nrow = int(np.ceil(len(slots) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (name, arr, cls, kind) in enumerate(slots):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        if kind == "phase":
            draw_phase(ax, arr, name)
        else:
            im = draw(ax, arr, name, cls)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
        overlay_axes(ax, arr)
    fig.suptitle(os.path.basename(npz_path), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, pack_stem(npz_path, stem) + ".png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    made.append(out)
    skipped = [p[0] for p in panels if p[1] is None]
    return made, ("skipped %d oversized" % len(skipped)) if skipped else None


# ------------------------------------------------------------------ HUD snapshots


def render_snapshots(snapdir, outdir, *, ncols=6, dpi=110):
    """Time montage from snap_*.npz: one row per field, columns are time."""
    if not os.path.isdir(snapdir):
        return [], "no such directory: %s" % snapdir
    snaps = sorted(f for f in os.listdir(snapdir) if re.match(r"snap_\d+\.npz$", f))
    if not snaps:
        return [], "no snap_*.npz"
    pick = [snaps[int(round(i))] for i in np.linspace(0, len(snaps) - 1, min(ncols, len(snaps)))]

    frames, fields = [], []
    for fn in pick:
        with np.load(os.path.join(snapdir, fn), allow_pickle=False) as z:
            d = {"t": float(z["t"]) if "t" in z.files else np.nan,
                 "scalars": {k[8:]: float(z[k]) for k in z.files if k.startswith("scalar__")}}
            for k in z.files:
                m = PLANE_RE.match(k)
                if m and m.group("plane") == "xy":
                    d[m.group("name")] = np.array(z[k])
                    if m.group("name") not in fields:
                        fields.append(m.group("name"))
            frames.append(d)
    if not fields:
        return [], "no '<field>__xy' planes"

    nrow, ncol = len(fields), len(frames)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.9 * nrow), squeeze=False)
    for r, name in enumerate(fields):
        for c, fr in enumerate(frames):
            ax = axes[r][c]
            arr = fr.get(name)
            if arr is None:
                ax.axis("off")
                continue
            cls = classify(name, arr)
            draw(ax, arr, name, cls, title=(f"{name}   t={fr['t']:.2f}" if r == 0 or True else None))
            overlay_axes(ax, arr)
    # HUD 6.3 diagnostic: a node count that flickers across consecutive frames of a fixed-N run is
    # the C2.8b signature. Whether it is the tracker or the physics, it must be SAID, not smoothed.
    warn = []
    for name in fields:
        counts = [len(find_centroids(fr[name])) for fr in frames if fr.get(name) is not None]
        if counts and (max(counts) != min(counts)):
            warn.append("%s n=%s" % (name, "/".join(str(c) for c in counts)))

    fr0 = frames[0]["scalars"]
    sub = "  ".join(f"{k}={v:+.3e}" for k, v in list(fr0.items())[:3])
    title = "%s   %s" % (os.path.basename(os.path.dirname(snapdir.rstrip("/\\")) or snapdir), sub)
    if warn:
        title += (chr(10) + "UNSTABLE NODE COUNT (tracker or merge -- check before "
                  "trusting any node observable):  " + "   ".join(warn[:3]))
    fig.suptitle(title, fontsize=10,
                 color=("#b00020" if warn else "black"))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "snapshots_%s_timeline.png" % os.path.basename(snapdir.rstrip("/\\")))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # scalar traces across every frame, so the montage has its numbers beside it
    made = [out]
    allsc = []
    for fn in snaps:
        with np.load(os.path.join(snapdir, fn), allow_pickle=False) as z:
            row = {"t": float(z["t"]) if "t" in z.files else np.nan}
            row.update({k[8:]: float(z[k]) for k in z.files if k.startswith("scalar__")})
            allsc.append(row)
    keys = [k for k in allsc[0] if k != "t"]
    if keys:
        fig, axs = plt.subplots(len(keys), 1, figsize=(7, 1.7 * len(keys)), sharex=True, squeeze=False)
        for i, k in enumerate(keys):
            axs[i][0].plot([r["t"] for r in allsc], [r[k] for r in allsc], lw=1.1)
            axs[i][0].set_ylabel(k, fontsize=8)
            axs[i][0].grid(alpha=0.25)
            axs[i][0].axhline(0, color="k", lw=0.5, ls=":")
        axs[-1][0].set_xlabel("t", fontsize=8)
        fig.tight_layout()
        o2 = os.path.join(outdir, "snapshots_%s_scalars.png" % os.path.basename(snapdir.rstrip("/\\")))
        fig.savefig(o2, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        made.append(o2)
    return made, None


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="render every .npz in sweep_runs/<RUN_ID>")
    ap.add_argument("--snapshots", help="render a HUD snapshot directory")
    ap.add_argument("--all", action="store_true", help="render every pack in sweep_runs/")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume a corpus pass: leave packs that already have a montage")
    ap.add_argument("--max-array-mb", type=float, default=1500.0)
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    jobs = []
    if args.snapshots:
        d = args.snapshots
        run_root = d
        for _ in range(3):
            run_root = os.path.dirname(run_root)
            if os.path.basename(os.path.dirname(run_root)) == "sweep_runs":
                break
        made, err = render_snapshots(d, os.path.join(run_root, "rendered"), dpi=args.dpi)
        print("snapshots: %d image(s)%s" % (len(made), "  [%s]" % err if err else ""))
        for m in made:
            print("   ", os.path.relpath(m, REPO))
        return

    if args.run:
        roots = [os.path.join(SWEEP, args.run)]
    elif args.all:
        roots = [os.path.join(SWEEP, d) for d in sorted(os.listdir(SWEEP))
                 if os.path.isdir(os.path.join(SWEEP, d))]
    else:
        ap.error("one of --run, --all or --snapshots is required")

    for root in roots:
        for dirpath, dn, fns in os.walk(root):
            # `rendered/` is output; `snapshots/` belongs to render_snapshots -- rendering
            # each snap_*.npz standalone would make thousands of near-identical single-frame
            # images instead of one timeline.
            dn[:] = [d for d in dn if d not in ("rendered", "snapshots")]
            for fn in fns:
                if fn.endswith(".npz"):
                    jobs.append(os.path.join(dirpath, fn))
    if args.limit:
        jobs = jobs[:args.limit]
    print("packs to render: %d" % len(jobs))

    n_ok = n_skip = n_have = n_err = 0
    t0 = time.time()
    for i, p in enumerate(jobs, 1):
        run_dir = p
        while os.path.dirname(run_dir) != SWEEP and os.path.dirname(run_dir) != run_dir:
            run_dir = os.path.dirname(run_dir)
        outdir = os.path.join(run_dir, "rendered")
        if args.skip_existing and os.path.exists(
                os.path.join(outdir, pack_stem(p, run_dir) + ".png")):
            n_have += 1
            continue
        try:
            made, err = render_pack(p, outdir, max_mb=args.max_array_mb,
                                    dpi=args.dpi, stem=run_dir)
        except Exception as exc:      # one bad pack must not end a corpus pass
            print("   FAILED %s: %s" % (os.path.relpath(p, SWEEP), exc))
            n_err += 1
            continue
        if made:
            n_ok += 1
        else:
            n_skip += 1
        if err and made:
            print("   %s  (%s)" % (os.path.basename(p), err))
        if i % 25 == 0:
            el = time.time() - t0
            done = max(1, i - n_have)
            print("  ... %d/%d   %.0fs elapsed, ~%.0fs left"
                  % (i, len(jobs), el, el / done * (len(jobs) - i)))
    print(chr(10) + "rendered %d pack(s); %d nothing renderable; %d already present; %d failed"
          % (n_ok, n_skip, n_have, n_err))
    print("output lands in <run>/rendered/ and is imported by tools/build_run_catalogue.py")


if __name__ == "__main__":
    sys.exit(main())
