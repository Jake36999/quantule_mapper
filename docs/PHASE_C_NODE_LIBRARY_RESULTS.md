# Phase D.3 — Phase C Node-Library, Harvest Results

**162 configs catalogued** (read-only) from 8 saved Phase C dirs via `jax_scout/node_library.py` →
`PHASE_C_NODE_LIBRARY.{csv,json}` (off-repo). No sims, no solver/gate change. The harvest is the compute-cheap first
batch (existing fields only); it is diverse *enough to start* coupling analysis, with clear, physics-driven gaps.

## Diversity of what we already have
- **Node count:** `{2:5, 3:4, 4:124, 5:5, 6:20, 7:3, 8:1}` — **dominated by 4-node**. This is **physics, not search
  bias**: the dynamics *merge* toward ~4 nodes (the feb-basin work already found K6/8 → 4 final nodes). The 6-node
  set is the seed620/621 morphology; higher counts are failures. **Node-count diversity is intrinsically capped ~4–6.**
- **Stability spectrum:** `{TRUE:129, SPIN_DOWN:23, TRANSIENT_GROWER:7, LATE_BLOWUP:3}` — a strong stable(129) +
  control(33) contrast set.
- **NN spacing (box units):** overall [0.071, 0.594]; stable 4-node feb clusters ≈0.485, seed620/621 6-node are
  **tighter** (0.21–0.35), K3→2-node **wider** (0.59). Moderate spacing diversity among stable configs.
- **Mass evenness:** feb 4-node uneven (`mass_cv`≈0.79–0.84, one dominant node), seed621 6-node even (`cv`≈0.12).
  Real morphological contrast (uneven-wide-4-node vs even-tight-6-node).

## The load-bearing finding: stable nodes are near-current-free
`stress_curr_frac` (weight of the phase-current term `J⊗J/ρ` in the info-stress) is **≈0.00 for essentially every
stable config**, and only nonzero for growers/blow-ups (0.21–0.33). i.e. the informational current `J=ρ∇φ` is
**negligible inside stable nodes** — they are near-current-free **standing** states (consistent with
[[gl-rotational-core-basin]] and the FMIA/routing nulls). Two consequences:
1. **Coupling between stable nodes cannot be current-mediated** — there is no current to carry it. Any node↔node
   coupling (D.4) must be **density- / phase- / geometry-mediated**, not current/vorticity.
2. This **explains D.2's inconclusive first pass**: in stable configs the stress tensor is dominated by the
   density-strain term (`∂√ρ ∂√ρ`), which is ≈ a density proxy — hence ‖T‖~ρ ≈0.5 and construction-fixed anisotropy.
   The current term only "lights up" where a config is *failing*.

## Diversity assessment
| axis | status |
|---|---|
| stability (TRUE / near / fail) | **good** (129 / — / 33) |
| node count | **limited** (physics merges to ~4–6) |
| spacing (stable) | **moderate** (0.21–0.59; tight-6 vs wide-4) |
| mass/morphology | **moderate** (even-tight vs uneven-wide) |
| current/vorticity (stable) | **~null** (J≈0 — a finding, not a gap to fill) |
| layout (bridged vs isolated) | **limited** (no deliberately-bridged stable configs) |

## Gaps + next steps
- **Real gap:** *spacing/layout* variety among **stable** configs (varied stable arrangements; deliberately close vs
  far node pairs). Node-count variety is physically capped, so chasing it is low-value.
- **Cheapest way to fill it:** a **small targeted seed/K batch** at the validated a\* params — different IC seeds and
  blob counts give different **stable arrangements** (spacing/layout) *without* new physics. (Each is a genuine
  stable config; this is "reuse the validated basin", not a new objective.)
- **Then D.2 re-run / D.4 first coupling analysis** on the library, focused on **density/phase/geometry** coupling
  (since current is ≈0): regress node-pair coupling proxies (density-bridge conductance, Ω-corridor smoothness,
  phase-difference, `∇·T` of the density-strain term) against **spacing** and **stability**, vs density-only nulls.

**Failure mode check (pre-registered):** "search only re-discovers the feb/a\* morphology" is **partially realised** —
the *stable* attractor genuinely favours ~4-node feb morphology; genuine diversity lives in **spacing/seed
arrangement**, not node count. So the library's value for coupling tests is in the **spacing/layout** axis, and the
current channel is a **null** to be reported, not forced.
