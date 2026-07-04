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

## Q3 — long-T confirmation: is the D_imag=0.001 coherence stable, or slow destabilisation? (tkick=48000, T=240)
| D_imag | T | mass ratio | nodes | v_x | com r² | verdict |
|---|---|---|---|---|---|---|
| 0.0 | 240 | **0.988** | 4→4 | +0.0002 | 1.00 | **stable + pinned** (baseline holds over the full window) |
| 0.001 | 30 | 1.03 | 4→4 | — | 1.00 | *looked* coherent |
| 0.001 | **240** | **2.518** | 4→**7** | −0.0002 | 0.80 | **runs away** — mass 2.5×, fragmenting |

**Decisive.** The `D_imag=0.001` coherence at T=30 was **slow destabilisation**: over the long window mass more
than **doubled** (1.03 → 2.52) and the node count grew (4→7), while the drift did **not** accumulate (disp <0.5% of a
box, `r²` fell to 0.80 — no sustained linear motion). The `D_imag=0` baseline stayed perfectly pinned and stable
(mass 0.99). ⇒ **there is no stable moving-coherent-soliton regime**: the dispersion that would confer mobility
instead breaks a\*'s gain/loss balance before it translates.

## Verdict — `C1_NO_STABLE_TRANSPORT`
The C1 route (adding a Schrödinger-like dispersive channel to the validated dissipative attractor) **does not yield
matter-like transport.** Precisely:
- The Phase C mobility null is **not merely a missing kinetic channel** — adding one (C1) confers only a transient,
  negligible drift and, at any dispersion strength large enough to matter, **destabilises a\*** (mass runaway,
  fragmentation).
- a\* is a **fundamentally stationary dissipative attractor**: its coherence and the dispersion needed to move it are
  **incompatible**. This *sharpens* (does not contradict) the Phase C site-pinning finding and matches the formalism
  gap review's "sectors are cleanly split" — the dissipative stability sector does not smoothly extend into a
  transport sector by a kinetic-term tweak.
- **Implication for Phase D:** matter transport, if it exists in IRER, is **not** reachable by perturbing the
  validated dissipative operator. It would require a genuinely different regime — e.g. **C2** (conservative
  nonlinear Schrödinger: matter as a norm-conserving moving soliton, which *abandons* the gain/loss balance that is
  the Phase C result) — i.e. a **different object**, not an extension of a\*. That is the natural next RFC branch, and
  a real scope decision.

Every Phase C result stands untouched (they are the `D_imag→0` limit); C1 was tested and cleanly answered on the
mirror without altering the frozen operator.

## Guardrails
jax_scout mirror only; `param_D_imag` default 0.0; frozen Phase C operator (`e8d6a78ea`) intact; nonlinearity,
geometry, gate, and the vmap sweep/Hunter path all unchanged (sweep stays at `D_imag=0`). No production/CuPy change.
