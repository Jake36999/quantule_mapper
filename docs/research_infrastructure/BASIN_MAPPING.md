---
tags: [record, infra, basins, continuation]
date: 2026-10-04
branch: Branch - Stability - Index
status: running
---

# Basin mapping: ensembles, clustering, continuation (Phase F)

This is the build record for Phase F of [[IMPLEMENTATION_PLAN_2026-10]]. It replaces the GA and
inverse-GP basin search (which blends distinct basins; see [[SEARCH_STACK_AUDIT_2026-10]]) with the
standard approach in three stages:
1. run many initial conditions forward per parameter point;
2. cluster the **end states** by shape;
3. follow each basin's state across parameters by numerical continuation, with Floquet stability.

**Status.** The tools are built and validated against known answers. The two GPU runs on a\* are
queued behind the B4 replay (`tools/revalidation/post_replay_queue.sh`): the basin ensemble and the
continuation pilot.

> [!info] Sources
> - F1: `jax_scout/registry.py` (`state_descriptors` v2), `tools/run_spec.py` (`final_only`),
>   `specs/approved/astar-basin-ensemble.json`
> - F2: `tools/basin_cluster.py`, `tests/test_basin_cluster.py`
> - F3: `jax_scout/continuation.py`, `jax_scout/astar_continuation_pilot.py`, `tests/test_continuation.py`
> - Evidence: `docs/instrument_integrity/evidence/phase_f/` (validation scripts, logs, KG demo basins)

## F1 — forward ensembles
An ensemble is an ordinary spec with a `sweep` over a parameter, seed or IC, plus the
`state_descriptors` observer (`final_only`).

The descriptors are **permutation-invariant**, so node ordering never matters:
- density contrast
- node count and node-size statistics
- periodic node-node spacing (mean, std, min)
- periodic radius of gyration
- power-weighted |k|
- inertia anisotropy

**v2 gates node descriptors on contrast.** A dispersed field always has speckle above the node
threshold. The first KG demo counted 15–19 "nodes" in fields that had simply spread out.

## F2 — clustering end states
`tools/basin_cluster.py run_dir --group-by param_a`:
1. **Shape (intensive) descriptors only.** Mass and amplitude vary continuously inside one basin and
   would split it.
2. **Relative scaling.** Scale-like columns use log10; the others are divided by their median
   magnitude. A distance of 0.1 means roughly "10% different in one descriptor".
3. Within each parameter point, cluster with average linkage and cut at 0.2. Then match local
   clusters across points by nearest centroid within the same radius, so one basin keeps one label
   along a parameter.

> [!warning] Three design errors found and fixed by testing (each now has a test)
> | attempt | what went wrong | fix |
> |---|---|---|
> | count nodes in every field | dispersed speckle counted as 15–19 "nodes" | gate on contrast (v2) |
> | cluster on all descriptors | 2× mass differences split one basin | shape descriptors only |
> | z-score + HDBSCAN | 1% jitter in noise-only columns was scaled up to unit variance; **2 basins came out as 12** | relative scaling + linkage cut |

**KG demonstration** (12 runs; amplitude × 3 start positions): weak packets (A ≤ 0.3) form one basin
and strong packets (A ≥ 1.6) form another. Start position never changes the basin. This matches the
spec's prediction.

## F3 — continuation of relative equilibria
`jax_scout/continuation.py` solves G(ψ, θ, a) = S_a R_θ Φ_T(ψ) − ψ = 0 on the fixed ETDRK4 (Φ_T).
- It is bordered by a phase condition and three translation conditions, and is matrix-free: J·v
  comes from `jax.linearize` and the linear solve is GMRES.
- Pseudo-arclength continuation in one physics parameter.
- Floquet multipliers from ARPACK on a jvp operator.
- One compilation serves a whole branch. The first version rebuilt a jit closure per Newton step, so
  GMRES recompiled every iteration and was unusable.

### Known-answer validation
| case | result |
|---|---|
| **uniform S-NCGL state, stable root** (g(ρ\*) = η, rotation ω0·T) | converged **quadratically** in 6 steps: ρ\* to 1.8e-12, θ = ω0T to 1.6e-14; Floquet: one neutral (phase) multiplier = 1, all others < 1, i.e. stable. Now a CI test (N=8, 5 s on CPU). |
| uniform state, **unstable** root | Newton converged, but onto **ψ = 0**: the trivial solution is also a relative equilibrium |
| exact **NLS soliton** (conservative branch) | the rotation check is exact (θ = μT to 1e-13), but Newton **stagnates** at about 7e-5 |

**What the two failures mean (limits, not bugs):**
1. **Trivial-solution attraction.** Newton finds *a* solution, not *the* one. Unstable nontrivial
   states need deflation (Farrell et al.) or an amplitude constraint. The a\* pilot reports
   `COLLAPSED_TO_TRIVIAL_SOLUTION` instead of a fake branch.
2. **Families in conservative systems.** NLS solitons form a continuous family in μ, so with θ free,
   every family member solves G = 0 and the bordered Jacobian is singular along the family. Hamiltonian
   continuation needs an extra condition (fixed mass, or an unfolding term). **Continuation on the C2/KG
   conservative branches is therefore not supported yet.** The dissipative a\* case has isolated
   solutions and does not have this problem.
3. **Breathing states**, i.e. periodic orbits whose period is itself unknown, are not implemented. A
   breathing a\* returns `NOT_A_RELATIVE_EQUILIBRIUM_AT_T_MAP`.

## Queued GPU work (after the B4 replay)
| step | spec / script | question | est. |
|---|---|---|---|
| E pilot | `specs/approved/c27-r3-soliton-transport-n1.json` | does the spec path reproduce the replay's v/2Dk = 1.0000? | 0.5 h |
| F1 | `specs/approved/astar-basin-ensemble.json` | do seeds 619 / 620 / 621 land in distinct basins (4 vs 6 nodes)? | 3.5 h |
| F3 | `jax_scout/astar_continuation_pilot.py --N 32` | is a\* a relative equilibrium, and does its branch change stability across the bracket? | 1–3 h |

## Issues raised

| # | issue | status |
|---|---|---|
| 1 | Continuation for conservative branches (soliton families) | OPEN — add a mass constraint or an unfolding parameter |
| 2 | Deflation to reach unstable nontrivial states | OPEN |
| 3 | Periodic-orbit (breathing) shooting with unknown period | OPEN |
| 4 | The a\* ensemble and continuation runs | OPEN — queued behind B4 |

## Associated docs
- [[IMPLEMENTATION_PLAN_2026-10]] · [[EXPERIMENT_SPECS_MCP_UI]] · [[SEARCH_STACK_AUDIT_2026-10]] · [[REVALIDATION_E270CDC_RESULTS]]

## Branches
- [[Branch - Stability - Index]]
