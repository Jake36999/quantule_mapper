# Phase D / C1 — Results (mixed dissipative+dispersive kinetic operator, jax_scout mirror)

**Candidate:** C1 from `PHASE_D_KINETIC_OPERATOR_RFC.md` — complex diffusion `(D_r + i·param_D_imag)∇²`, i.e. the
linear operator gains an imaginary `−D_imag·k²` (Schrödinger-like dispersive channel). Implemented default-off on the
**jax_scout mirror only**; the frozen Phase C production operator is untouched. `param_D_imag=0` reproduces Phase C
**byte-for-byte** (`phase_d_c1_parity.py` → `C1_PARITY_PASS`; all ETDRK4 operators max|Δ|=0.0, 50-step evolution=0.0).

## Q1 — does the dispersion destabilise a\*? (stability diagnostic, no kick, 2000 steps)
| D_imag | mass ratio | nodes | note |
|---|---|---|---|
| 0.0 | 0.999 | 4→4 | baseline a\* holds |
| 0.002 | 1.016 | 4→4 | coherent, tiny growth |
| 0.005 | 1.287 | 4→7 | growing + fragmenting |
| 0.01 | 1.884 | 4→**388** | fragmented (numerical noise) |
| 0.02 | 9.616 | 4→1 | blown up |

⇒ a\* stays coherent **only for `D_imag ≲ 0.002`**. The knife-edge gain/loss balance is fragile to spectral
redistribution: dispersion changes the local density ρ, which shifts the density-dependent gain/loss off balance.

## Q2 — is there mobility in the coherent range? (kick probe, a\* state, tkick=6000, N=96)
`PHASE_D_C1_TRANSPORT_20260704_105309`, mobility μ = dv/dk over kicks n=0,1,2:
| D_imag | μ = dv/dk | coherent? | mass | nodes | max disp |
|---|---|---|---|---|---|
| 0.0 | 0.00007 (≈0 — **pinned**) | ✅ | 0.99 | 4→4 | 0.001 box |
| 0.001 | 0.00069 (~10× baseline) | ✅ | 1.01–1.03 | 4→4 | 0.003 box |
| 0.002 | 0.00139 (~20×) | ❌ | 1.33–1.37 | 4→7 | 0.005 box |

- **The Phase C structural null is confirmed at `D_imag=0`** (μ≈0, a\* pinned) — reproduces the kick/drag findings.
- **The dispersive channel opens a real advective response:** μ scales with `D_imag` *and* with kick k (v∝k,
  `r²`≈1, sustained constant-velocity drift), unlike the structural null. So transport is no longer structurally
  impossible.
- **But it is a transport↔stability trade-off:** more `D_imag` → more mobility *and* more destabilisation. In the
  coherent window (`D_imag≤0.001`) the drift is tiny (<0.3% of a box over T=30); the `D_imag` that gives more
  mobility (0.002) breaks coherence. **Mass grows monotonically with `D_imag`** (0.99 → 1.03 → 1.35), i.e. the
  dispersion pushes a\* off its balance toward growth.

## Long-T confirmation (pending)
The `D_imag=0.001` coherence over T=30 already shows slow mass growth (→1.03); a long-T run
(`D_imag∈{0,0.001}`, tkick=48000) tests whether that coherence is a **stable** moving regime or merely **slow
destabilisation**. <!-- LONG_T_RESULT -->

## Interpretation (so far — honest, not over-claimed)
C1 refines the Phase C mobility null: it is **not** that transport is structurally forbidden (adding the dispersive
kinetic term *does* confer a real, k-proportional advective response), but that a\* is a **dissipative attractor
whose coherence is in tension with the dispersion needed for meaningful transport** — the more you disperse (to
move it), the faster you break the gain/loss balance. No robust *stable moving coherent soliton* regime was found in
the coherent range; a\* is **"softly pinned."** This is a genuine Phase D finding on the C1 route, and it does not
touch or weaken any Phase C result (all of which stand as the `D_imag→0` limit).

## Guardrails
jax_scout mirror only; `param_D_imag` default 0.0; frozen Phase C operator (`e8d6a78ea`) intact; nonlinearity,
geometry, gate, and the vmap sweep/Hunter path all unchanged (sweep stays at `D_imag=0`). No production/CuPy change.
