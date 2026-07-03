# Baseline Reproduction Runbook (validated Phase C baseline)

**Canonical object name (supersedes the stale "T=6000 4-node attractor" of `RUNBOOK_PHASE_C_AND_VISUALS.md`):**

`GAIN_LOSS_BALANCED_DISSIPATIVE_ATTRACTOR (a* ≈ ×1.15)`

Meaning: a reproducible **long-time (T=144000) stationary** bounded standing attractor of the *local, real
dissipative cubic-quintic-septic GL field on a density-sourced scalar conformal geometry*, governed by a
knife-edge gain/loss balance. **Cautious labels only** — not matter, not a particle, not a topological invariant,
not analogue-gravity-proven, not a molecule. It is **site-pinned** (no mobility) by construction of the
dissipative operator. See `BASELINE_AUDIT.md`, `PHASE_C_MOBILITY_ENDPOINT.md`.

## 1. Environments (three boxes)
| box | role | note |
|---|---|---|
| Windows dev (`F:\quantule_mapper`) | edit + `py_compile` only | **no** cupy/jax — cannot run the solver |
| WSL Ubuntu GPU (`~/jax_irer` venv) | **runs jax_scout** (all Phase C validation) | `source ~/jax_irer/bin/activate` |
| CuPy production GPU box | `worker_cupy` / `validation_pipeline` | separate; hosts `simulation_ledger.db` |

## 2. Frozen reference
- Geometry frozen at commit `e8d6a78ea`; solver = `jax_scout/physics.py` (ETDRK4, FP64, N=96, L=10, dt=0.005).
- **feb params** (`core_saturation_search.FEB`): `D=2.7329, eta=0.0704, rho_vac=1.1866, omega0=0.0,
  a_coupling=2.3098, s=0.0129, f=-0.4861, a=0.4802`.
- **a\*** = `param_a × 1.15 ≈ 0.5522` (the gain/loss balance; ×1.15–1.16 knife-edge, ±~0.5%).
- IC: K=6 per-blob, seed `20260619` (`IC_NORM_PER_BLOB_FIXED`). Node count is IC-set (seed 619 → 4-node; 620/621
  → 6-node) — stability is param-set, morphology is IC-set.

## 3. Launch pattern (WSL, detached — survives session/teardown)
```bash
ts=$(date +%Y%m%d_%H%M%S); out="sweep_runs/<RUN>_${ts}"
wsl.exe -d Ubuntu -- bash -c "cd /mnt/f/quantule_mapper && source ~/jax_irer/bin/activate && \
  setsid nohup python jax_scout/<script>.py [--args] --out $out \
  > runtime_logs/<script>.console.log 2>&1 < /dev/null & sleep 5; \
  pgrep -af <script> | grep -v grep || echo FAILED"
```

## 4. Reproduce each validated claim
Each run is verdict-first (writes CSV + `summary.json`); ~64 min per T=72000 cell on the GPU box.

| claim | script + args | expected result |
|---|---|---|
| **a\* exists** (late-slope→0) | `feb_gain_ladder_longt.py --T 72000` | slope /1k: ×1.10 −0.008, ×1.125 −0.005, **×1.15 −0.0004 (flat)**, ×1.16 +0.001, ×1.20 → `TRANSIENT_GROWER_REJECT` |
| **a\* confirmed** (longer-T + seed + bracket) | `feb_astar_confirm.py` | a×1.15 flat to **T=144000** (slope −0.0004); a×1.125 keeps declining; a×1.15 stable at seeds 619/620/621; ×1.16 grows |
| **no inertial mobility** | `feb_kick_inertia.py` | v_x kick-independent at noise floor (μ≈0), nodes 4→4 — `GALILEAN_KICK_NO_TRANSPORT` |
| **no relational mobility** | `feb_adiabatic_drag.py --mode static --V0s 0.075,0.1,0.15,0.2,0.3,0.4` | `BASELINE_REPRODUCED=True` (bit-identical); all-V0 accretion, origin never depletes — `ACCRETION_ONLY_NO_RELOCATION` (generalized across seed619/620/621) |

Interpret with the promotion gate (§5). The mobility runs boost/probe the **saved a×1.15 state**
(`sweep_runs/FEB_GAIN_LADDER_LONGT_T72000_20260701_175708/a1.15_ladder_T72000_probe.npz` + the
`FEB_ASTAR_CONFIRM` seed states) — no re-settle needed.

## 5. Promotion gate (the only criterion the closed claims passed)
`core_saturation_search.classify` **v3** (energy-stability): `er` band + `late_drift` + breathing exception
(`floor_ratio≥0.85`, `er_fin≤0.95·er_max`), refined by the **late-window slope→0** long-time criterion
(a\*-arc). Prime-SSE and TDA/Betti are **non-discriminating** and are *not* part of the gate
(`BASELINE_AUDIT_VALIDATION.md`). **Standing rule:** a bound-state claim requires a long-window slope→0 verdict —
short-window "saturation" is not stability (`BASELINE_AUDIT_NUMERICAL.md §7`).

## 6. Provenance & evidence
Load-bearing runs (path + sha256 + claim) are manifested in `EVIDENCE_INVENTORY.md`. Scripts are versioned (git,
frozen `e8d6a78ea`); results are on local disk (Git is intentionally lightweight). To re-derive any closed claim:
run the script above at the frozen geometry, or re-read the manifested `summary.json`.
