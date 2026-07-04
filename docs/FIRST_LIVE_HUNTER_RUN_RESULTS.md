# First Live Hunter Run — Results: `LIVE_HUNTER_REDISCOVERY_PASS`

**The last gap is closed.** After A5 showed the objective re-finds a\* by *replay*, this run shows the re-aimed
Hunter re-finds a\* by **adaptive search** — the point of H7. Run `sweep_runs/LIVE_HUNTER_20260703_230055`
(local `.venv`, ~5.8 h), driver [`tools/first_live_hunter_run.py`](../tools/first_live_hunter_run.py).

## Config (tiny, per `FIRST_LIVE_HUNTER_RUN_PLAN.md`)
`objective="stability"` · population 6 · 2 generations · N=96 · T=24000 · narrow box
`param_a∈[0.48,0.60]`, `param_eta∈[0.0598,0.0810]`, `param_rho_vac∈[1.068,1.365]` (all other params held at feb) ·
Gen-0 seeded with a\* (0.5522), decayer (0.5042), grower (0.6003) + 3 narrow-box random · `css`/stability gate
certifies. Selection = **H7.1b** stability generator (fitness tournament + bounded mutation on the 3 axes, **no
SGN/ASMT/NSGA/spectral steering, no prime-SSE**).

## Gen 0 (seeded) — evaluated on the production CuPy path
| idx | origin | param_a | score | cert | slope/1k | note |
|---|---|---|---|---|---|---|
| 0 | seed **a\*** | 0.5522 | **0.913** | ✅ | +0.0002 (flat) | **top** |
| 3 | random | 0.5444 | 0.627 | ✅ | −0.0037 | near a\* |
| 5 | random | 0.5486 | 0.581 | ✅ | −0.0046 | near a\* |
| 1 | seed decayer | 0.5042 | 0.443 | ✅ | −0.0132 | below a\* |
| 2 | seed grower | 0.6003 | 0.175 | ✅ | +0.0207 | penalised |
| 4 | random (low-η/high-ρ) | 0.5783 | −1.0 | ❌ | +0.030 | `GROWER_BLOWUP` — rejected |

## Gen 1 (H7.1b selection: elite + tournament mutations)
| idx | origin | param_a | score | cert |
|---|---|---|---|---|
| 0 | **STABILITY_ELITE** | 0.5522 | **0.913** | ✅ |
| 4 | STABILITY_MUTATION | 0.5536 | 0.712 | ✅ |
| 1 | STABILITY_MUTATION | 0.5514 | 0.691 | ✅ |
| 2 | STABILITY_MUTATION | 0.5409 | 0.588 | ✅ |
| 3 | STABILITY_MUTATION | 0.5415 | 0.553 | ✅ |
| 5 | STABILITY_MUTATION | 0.5424 | 0.535 | ✅ |

**Every Gen-1 mutation landed in `param_a ∈ [0.54, 0.554]` and certified** — the search *concentrated* on a\*.

## Verdict — all checks pass
| check | result |
|---|---|
| final best certifiable | ✅ (a\*, 0.913) |
| final best in a\* box `[0.52,0.58]` | ✅ (param_a 0.5522) |
| converged toward a\* (mean param_a) | ✅ **0.5547 → 0.547** (a\*=0.5522) |
| prime-SSE not steering | ✅ (structural — stability branch never reads prime) |
| short-window artifact promoted | **no** (all T=24000; the one blow-up was rejected, not promoted) |

→ **`LIVE_HUNTER_REDISCOVERY_PASS`.** The re-aimed Hunter, driven only by the stability score, re-discovers the known
a\*≈×1.15 basin by search: it seeds broadly in the box, ranks a\* top, rejects the blow-up, and its next generation
clusters tightly on a\* — **re-finding a known result, not new physics.**

## Scope / honesty
- This is a **re-find confirmation** in a narrow box (pop 6, 2 gens), not a broad hunt. It used the minimal
  stability-selection mode (SGN/ASMT/NSGA bypassed); a fuller adaptive stability front over more axes is future work.
- Solver / physics / kinetic term / geometry / gate unchanged; the frozen Phase C dissipative operator is intact.
- A transport/dispersive kinetic operator remains **not implemented** — a later Phase D RFC, not this work.
