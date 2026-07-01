# Visual Analysis & Rendering — Handover to Codex

Purpose: how the Quantule Mapper scout results are **visually analysed and rendered**, so you can
re-run, extend, or audit the figure pipeline. Covers the compute→render split, the canonical
`quantule_viz` package, the field/vorticity primitives, and the per-study data→figure map.

---

## 1. Architecture: compute in WSL, render on Windows (hard split)

The numerics and the rendering run in **different Python environments** because the libraries don't
co-exist:

| Stage | Environment | Has | Used for |
|---|---|---|---|
| **Compute** | WSL2 Ubuntu venv `~/jax_irer` (`jax 0.10.2`, **CUDA/GPU**) | jax, numpy | run S-NCGL sims, write artifacts |
| **Render** | **Windows** python on PATH | numpy, **matplotlib, pyvista, skimage, h5py, imageio, PIL** | read artifacts → figures |

- WSL invoke: `wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python <script>'`
- **Windows python has NO jax; WSL jax venv has NO matplotlib/pyvista/h5py.** Therefore **renderers
  must be jax-free** — node detection in the render layer uses numpy/skimage flood-fill, NOT
  `transfer_diag.detect_nodes` (which imports jax). This is a real footgun: importing anything from
  `jax_scout` that transitively imports `jax` will crash the Windows renderer.
- **Data contract** the compute scripts emit (this is the render layer's input):
  - **Scalar time-series / class tables** as JSON + CSV (small).
  - **Field frames** as `.npz` of `psi` (complex64) + `frames` (int snapshot indices). Renderers
    recompute density/current/vorticity from `psi` in numpy.

---

## 2. Canonical render layer: `quantule_viz` (read-only package)

Top-level package `quantule_viz/` (v0.2.0). **Read-only**: it loads saved artifacts and writes
figures; it never runs a sim. Run on **Windows python**:

```
python -m quantule_viz <subcommand> <run_dir> [--latest] [--outdir DIR] [--overwrite]
```

Subcommands (`quantule_viz/cli.py`):
| Subcommand | Renders | Run-dir patterns / input |
|---|---|---|
| `phase-c` | saturation-search panels: summary_panel, class_histogram, class_counts_by_K, node_counts_by_K, saturation_slope_scatter, best_candidates_table.csv, analysis_summary.json | `CORE_SAT_HUNT_*`, `CORE_SAT_PILOT_*` (`all_evals.csv`+`summary.json`) |
| `core-basin` | basin map: pairwise scatter, per-param histograms, basin_summary | `CORE_BASIN_*` / `CORE_BASIN_REFINE_*` |
| `core-characterize` | long-time persistence, radial profiles, perturbation | `CORE_BASIN_*/core_characterize/` |
| `feb-bound-state` | relaxation timeline, per-node profiles, tracks+slices, perturbation | `SUBSTRATE_HUNT_*/feb56dc7_bound_state/` |
| `phase-c-structured` | structured-discovery visual pack (needs `summary_csv`, `--shortlist`, `--cases-root`, `--outdir`) | `docs/phase_c_structured_discovery_B_summary.csv` |
| `frames` | generic density-slice + vector preview from ANY npz with `psi`/`frames` | any `.npz` field bundle |

Conventions (all in `quantule_viz/io.py`):
- `--latest` resolves the newest matching run dir under `sweep_runs/` (mtime).
- `guard_outputs()` refuses to overwrite existing figures unless `--overwrite` (idempotent, safe to re-run).
- `IncompleteRunError` + `renderers/__init__.py::KNOWN_INCOMPLETE_RUNS` skip crashed runs. **NOTE:**
  `CORE_SAT_HUNT_20260622_211628` is flagged there — that 6h IC-varied hunt crashed with an empty
  `all_evals.csv` (re-run needed; see §5).
- `maybe_float`/`maybe_int` tolerate blank CSV cells; `latest_run` excludes `exclude_parts`.

### The deprecated shims
`plugins/visualizers/core_characterize_render.py` and `core_basin_refine_render.py` are now thin
wrappers that print a deprecation notice and forward to `python -m quantule_viz ...`. The other
`plugins/visualizers/*_render.py` (payan_chiral_slices, payan_hifi_render, core_basin_render,
feb_bound_state_render) are the **original standalone renderers** I wrote — still functional, but
`quantule_viz` is the canonical path. Prefer migrating any remaining standalone logic into
`quantule_viz/renderers/`.

---

## 3. The visual-analysis primitives (how a field becomes a figure)

All in `quantule_viz/plots.py` — these ARE the "visual analysis" methods:

- **`as_density(psi)`** = `|psi|^2`.
- **`density_slice(field)`** → picks the axis/center through the **densest voxel** and returns a 2D
  slice (used for ρ heatmaps/contours).
- **`phase_current(psi)`** → the informational current `J = Im(conj(psi) * ∇psi)` via `np.gradient`
  (3 components). (The original `payan_chiral_slices.fields()` computes the same J + vorticity via
  **FFT** gradients — equivalent; FFT is more accurate for periodic fields, np.gradient is cheaper.)
- **`vector_slice(psi)`** → ρ slice + in-plane current (u,v) + **pseudo-vorticity** `∂v/∂x − ∂u/∂y`
  (the swirl/handedness map). This is the core "vector map" overlay.
- **`find_component_centroids(field, quantile=0.997, min_size=4)`** → **jax-free node detector**:
  threshold at a high density quantile, flood-fill connected components, return centroids. This
  replaces `transfer_diag.detect_nodes` on the Windows side (used for node tracks).
- **`render_density_slices` / `render_vector_preview` / `render_frame_pack`** → the generic
  `frames` subcommand: density slices (early/mid/late) + a final-slice current-quiver +
  pseudo-vorticity panel, with NaN/inf guarding and 99th-pct color scaling.
- **`CLASS_COLORS`** → the standard class palette (SUSTAIN/TRUE_SAT green … BLOWUP red …) used across
  all panels for consistency.

**Domain conventions:** grid `L=10.0`, `dt=0.005`; node/shell radii in **voxels**; the box is
**periodic** (use minimal-image for distances). "Vorticity" here is the curl of `v = J/ρ` (or the
2D pseudo-vorticity of a slice) — a swirl proxy, not a topological winding number.

---

## 4. The richer per-study visual analyses (what each figure means)

These are the analysis *idioms* I used (now in the renderers); reuse them:

- **⊥-axis slices** (`payan_chiral_slices`): slice perpendicular to a node/bridge axis, overlay ρ
  contour + J quiver + axial-vorticity colormap → diagnoses local vs inter-node circulation /
  handedness.
- **Vortex-core dynamics** (`payan_hifi_render`, `core_characterize`): time-series of core density,
  radial inflow `v_r`, tangential circulation `v_t`, swirl ratio `v_t/(|v_r|+v_t)` → distinguishes
  spiral-sink vs pure vortex vs decay; the **radial profiles** ρ(r)/v_r(r)/v_t(r) are the node "anatomy".
- **Resolution-convergence overlay** (`payan_hifi_render`): er(t) & n_nodes(t) across N=48/96/128 on
  one axes → real structure vs low-N artifact.
- **Node tracks**: per-frame centroids (from `find_component_centroids`), colour=time, in x–y
  projection → migration / merger / lattice.
- **Stability curve** (basin refine): `P(SUSTAIN|η)` / `P(class|η)` line + stacked-area composition.
- **Perturbation panels** (`feb_bound_state`, `core_characterize`): kicked-vs-unperturbed core
  density / v_t over time → attractor (returns) vs fragile.
- **3D density isosurface GIFs**: `plugins/visualizers/visual_analysis_suite.py` (pyvista +
  skimage marching_cubes; off-screen). Modes: `single`, `compare`, `topology`, `stitch`. Needs an
  HDF5 `rho_history` dataset; build it on Windows from a `.npz` (WSL has no h5py).

---

## 5. Compute scripts (upstream — produce the artifacts the renderers consume)

All under `jax_scout/`, run in the **WSL jax venv**. Key ones from this arc:

| Script | Produces | Rendered by |
|---|---|---|
| `core_saturation_search.py` (`--ic-counts`, `--hours`, `--N`, `--T`) | `sweep_runs/CORE_SAT_{HUNT,PILOT}_*/all_evals.csv`,`summary.json` | `phase-c` |
| `core_basin_sweep.py` | `sweep_runs/CORE_BASIN_*/all_evals.csv`,`summary.json` | `core-basin` |
| `core_basin_refine.py` | `CORE_BASIN_REFINE_*/refine_curve.json`,`all_evals.csv` | `core-basin` |
| `core_basin_validate.py` | `CORE_BASIN_*/basin_validation_N96.json` | (read directly) |
| `core_characterize.py` | `CORE_BASIN_*/core_characterize/core_characterize.json`,`core_frames.npz` | `core-characterize` |
| `feb_bound_state.py` + `feb_bond_test.py` | `SUBSTRATE_HUNT_*/feb56dc7_bound_state/feb_bound_state.json`,`feb_bond_test.json`,`frames.npz` | `feb-bound-state` |
| `payan_hifi_continuation.py` | `SUBSTRATE_HUNT_*/hifi_N*_L*_T*/hifi_series.json`,`hifi_frames.npz` | `payan_hifi_render` |
| `payan_chiral_capture.py` | `SUBSTRATE_HUNT_*/chiral_viz/chiral_fields.npz`,`chiral_meta.json` | `payan_chiral_slices` |

**Engine note:** the fast sweeps use `physics.sweep_probe` (vmap-batched bare S-NCGL; returns
`psi_mid`, `psi_final`, `energy[B,n_steps]`, `max_amp`, `finite`). The `energy[]` trajectory is what
the saturation classifier slopes; `psi_final` feeds node-count + core-density. Long single-config runs
use a jitted snapshot capture (`core_characterize._capture_bare`). Bare S-NCGL = `gamma_A=0` (no
A-field); A-coupled runs use `afield_current_coupled.capture_cc`.

**Outstanding compute task:** the 6h `core_saturation_search.py --ic-counts "1,2,3,4,6"` hunt
(dir `CORE_SAT_HUNT_20260622_211628`) **crashed with an empty CSV** — re-run it (WSL, ~6h) to get the
`sat_node_counts_by_IC_blobs` distribution, which answers "does final node count track the IC blob
count (lone solitons possible) or is the field-preferred multiplicity always 4–5?" Then
`python -m quantule_viz phase-c --latest` to render.

---

## 6. Suggested workflow for you (Codex)

1. Render an existing run: `python -m quantule_viz phase-c --latest --overwrite` (or pass a specific
   `sweep_runs/...` dir). Same for `core-basin`, `core-characterize`, `feb-bound-state`.
2. For an ad-hoc field bundle: `python -m quantule_viz frames path/to/bundle.npz`.
3. To extend: add a renderer in `quantule_viz/renderers/`, reuse `plots.py` primitives + `io.py`
   loaders/guards, register it in `cli.py`. Keep it **read-only and jax-free**.
4. Honor the discipline this project runs on: figures are **inspection aids, not proof**; keep the
   class taxonomy + colors consistent; cite N, T, seed count, and window caveats on every panel.

## 7. Gotchas
- Don't `import jax_scout.transfer_diag` (or anything pulling `jax`) in a Windows renderer.
- `psi` frames are **complex64**; cast to complex128 before FFT-gradient math.
- Periodic box → minimal-image for all distances; node radii are in voxels (`dx = L/N`).
- 99th-percentile symmetric color scaling + `nan_to_num` for vorticity maps (fields can have spikes).
- `quantule_viz` figures are guarded against overwrite — pass `--overwrite` to regenerate.
