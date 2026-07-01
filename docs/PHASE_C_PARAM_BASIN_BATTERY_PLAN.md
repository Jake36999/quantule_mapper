# Phase C — parameter-basin battery (10–12 h, PREPARED — not launched)

**Date:** 2026-06-25
**Script:** `jax_scout/feb_param_basin.py` (resumable `--out`, `--deadline-hours` default 11.5).
**Premise:** the feb-basin study showed bound-state formation is a **parameter-regime** property (robust
to K / seed / IC-norm at feb's params). Gate **v3** now handles long-T breathing. This battery (a) closes
the loop by long-T-revalidating the discovered node family under v3, and (b) maps how wide feb's
bound-state basin is in **parameter space** — the agreed principled successor to the broad hunt.
**Discipline:** analysis/replay only — fixed solver/geometry/classifier-logic; only feb's input parameters
are perturbed via the existing `run_probe`. No PDE/solver/geometry change, no new hunt, no broad random
search. Classifier v3. **Do not launch until approved** (and prefer overnight: daytime WSL teardowns have
killed long runs twice — the script is resumable so a teardown only costs the in-progress config).

## P1 — node-family long-T revalidation (~2 h)

K ∈ {3,4,5,6,8}, feb params, per-blob, seed 20260619, **T=24000**, v3 gate. Confirms the node-count family
(found stable at T=12000) is genuinely long-time stable now that the gate is breathing-aware (the T=24000
check that previously mis-fired under v2). 5 runs × ~25 min.

## P2 — parameter one-at-a-time (OAT) basin map (~9 h, T=12000)

Center: feb (all params ×1). Then perturb **each** of feb's 8 params one at a time, holding the rest at
feb, K=6/per-blob, T=12000, v3-gated:

| param | feb value | sweep |
|---|---|---|
| param_D | 2.7329 | ×{0.5, 0.75, 0.9, 1.1, 1.25, 1.5} |
| param_eta | 0.0704 | ×{0.5 … 1.5} |
| param_rho_vac | 1.1866 | ×{0.5 … 1.5} |
| param_omega0 | 0.0 | set {0.05, 0.1, 0.2, 0.4} (turns on the vacuum oscillator) |
| param_a_coupling | 2.3098 | ×{0.5 … 1.5} |
| param_s | 0.0129 | ×{0.5 … 1.5} |
| param_f | −0.4861 | ×{0.5 … 1.5} |
| param_a | 0.4802 | ×{0.5 … 1.5} |

≈ 47 configs × ~12.6 min. Each yields class + drift + bounded_breathing + n_fin + held_mass, recorded
incrementally. This gives **1-D stability intervals** for every parameter — which params feb's bound state
is most/least sensitive to, and how far each can move before the state decays or blows up.

## Totals & order

≈ 52 configs, ≈ 11.9 h nominal; deadline 11.5 h with **P1 first** (so the revalidation is guaranteed and
only the P2 tail can be trimmed). Resumable: re-invoke with `--out <same dir>` to finish any skipped tail.

## Launch (when approved)

```bash
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && \
  python jax_scout/feb_param_basin.py' > runtime_logs/feb_param_basin.console.log 2>&1
# resume after any interruption:
#   python jax_scout/feb_param_basin.py --out sweep_runs/FEB_PARAM_BASIN_<ts>
```

## Reads / what it answers

- **P1:** is the node-count family long-time stable under the corrected (breathing-aware) gate? (expected
  yes; confirms the feb-basin conclusion at T=24000).
- **P2:** the shape of the bound-state basin in parameter space — per-param stable windows, the most
  sensitive direction(s), and whether turning on the vacuum oscillator (param_omega0 > 0) keeps a bound
  state. This is the data needed to decide whether a (later) joint multi-param search is worthwhile.

No charge / topology / proof / ground-state / black-hole / universal-law claim is made. Promotion to
"bound state" uses the v3 gate; nothing is auto-committed (generated outputs stay in `sweep_runs/`).
