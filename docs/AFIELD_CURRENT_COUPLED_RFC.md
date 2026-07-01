# RFC — Current-Coupled A-field (the "rate of interaction" / FMIA-wire mechanism)

**Status:** DRAFT design contract. NOT yet implemented. Do not code into the solver until this
is reviewed. JAX scout branch only; production `gravity/unified_omega.py` stays untouched.
**Date:** 2026-06-20

## 1. Why (motivation from measurement)

The bridge-objective finalists are robust **coupled** multi-node structures, but the coupling is:
- **global / collective, not pairwise** — response-matrix `global_mode_fraction ≈ 0.89`
  (Stage 2), i.e. kicking any node excites essentially the same collective mode;
- **non-selective** — bridge ≈ node (selectivity 1.3-1.4); the density bridge is not a special
  routing channel;
- **slow + linear** — peak ~2600 steps, amplitude-scaling exponent ≈ 1.06 (linear response);
- **real but bridge-associated** — a no-corridor control propagates ≈ 0.

This is the textbook signature of geometry coupling to a **scalar** (density ρ): a scalar source
has magnitude but no direction, so the induced geometry pulls isotropically → a holistic web, not
directed wires. (Confirmed by audit: both the production geometry and the existing A-field
prototype are scalar/density-sourced — see §2.)

To get **directed, selective** transfer (FMIA "Informational Parallels" / wires), the field must
couple to the **informational current**, which carries direction:
```
J_info = ρ ∇φ = Im( conj(ψ) ∇ψ )      (a 3-vector field)
```
This is the IRER translation of a least-action "rate of interaction": energy/phase is carried
along the current, and a current-coupled potential (EM vector-potential analogy) forces the
least-action path into a discrete conductive channel.

## 2. Audit of the existing A-field (what it is NOT)

`jax_scout/afield_prototype.py` (`IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1`):
- **Source:** `d²A/dt² = -c²k²A + ρ_k` — sourced by ρ = |ψ|² (SCALAR, DC-removed).
- **Coupling back:** scalar modulation only — `ρ_vac_eff = ρ_vac + γ_A·A` (vacuum_ref) or
  `Ω²_eff = Ω²·exp(γ_A·A)` (additive_potential).
- **Verdict:** scalar density-sourced + isotropic scalar modulation. It is NOT current-coupled and
  CANNOT, by construction, produce directional routing. Testing it (`afield_rescue_test.py`) is an
  `A_SCALAR_RESCUE_TEST`, not a current-coupled test.

## 3. Proposed physics (ONE chosen form — minimal coupling)

Treat A as a **3-vector potential** `A = (Ax, Ay, Az)`, evolved spectrally with finite speed,
sourced by the informational current, coupling to ψ by **minimal coupling** (gauge-covariant
derivative ∇ → ∇ − i γ_A A).

**A evolution (finite-speed, causal):**
```
∂²A/∂t² = c_A² ∇²A − Γ ∂A/∂t + κ · P_T[ J_info ]
```
- `J_info = Im(conj(ψ)∇ψ)`, projected to its transverse (divergence-free) part `P_T` (Coulomb
  gauge ∇·A = 0) so A is a clean circulation/routing field; spectral projection
  `P_T[J]_k = J_k − k(k·J_k)/|k|²`, with k=0 pinned (gauge zero mode), DC removed.
- `Γ` optional weak damping for boundedness; `κ` source strength; `c_A` propagation speed.

**ψ coupling (minimal coupling in the kinetic term):**
```
(∇ − iγ_A A)² ψ  = ∇²ψ − iγ_A( 2 A·∇ψ + (∇·A)ψ ) − γ_A² |A|² ψ
                 = ∇²ψ − 2iγ_A A·∇ψ − γ_A² |A|² ψ        (Coulomb gauge ∇·A=0)
```
So the solver's diffusion/kinetic term gains exactly two extra real-space terms:
`−2iγ_A (A·∇ψ) − γ_A² |A|² ψ`, computed via FFT gradients and folded into the ETDRK4 nonlinear
operator `n_op` (the linear operator L is unchanged). The conformal Ω² covariant-Laplacian path
is left as-is for v1 (A couples to the base kinetic gradient; revisit coupling to the covariant
gradient in v2 if warranted — documented choice).

**Exact baseline guarantee:** γ_A = 0 → both extra terms vanish → `n_op`/`step` reproduce the
current solver bit-for-bit (must verify rel_L2 ~ machine eps, as the scalar prototype did).

## 4. Code contract

- New module `jax_scout/afield_current_coupled.py`; **new contract key**
  `IRER-SNCGL-CURRENT-COUPLED-AFFECT-ETDRK4-v1` (distinct from scalar keys; A-on runs NOT
  rank-compatible with γ_A=0; never mixed in any ledger ranking).
- `physics.n_op`/`step` gain an optional `a_vec` argument (default None → no-op, exact baseline);
  mirrors the existing optional `rho_vac_eff` / `omega_sq_mult` pattern.
- A updated ONCE per outer step (held across the 4 ETDRK4 substages), mirroring the scalar
  prototype + `solver/run.py` ordering.
- Safety rails: k=0 pinned; transverse projection; optional damping Γ; clip/floor as needed;
  finiteness + A-energy/A-max runaway guards.
- Production `gravity/unified_omega.py` untouched; governance gate extended with a
  "current-coupled A-field contract stamped + default-off + reserved" check.
- Add a γ_A=0 equivalence test + a CuPy parity note (CuPy has no current-coupled term yet → A-on
  is JAX-scout-only until a CuPy implementation exists; do not promote on JAX alone).

## 5. Falsification / success criteria

Run on frozen gen18/gen14/gen34, γ_A sweep {0, 0.01, 0.02, 0.05, 0.1, 0.2}, 1600 steps + the
clean phase-kick routing test. The current-coupled A-field is supported only if, vs γ_A=0:
- **global_mode_fraction DROPS** (web → wires: response becomes pairwise, not collective);
- **node_bridge_selectivity RISES** (bridge becomes a preferential channel);
- phase coupling **stays above the 0.73 floor at 1600 steps** (transient rescued);
- phase-kick response becomes **resolved + bridge-selective** (not boundary-pinned, bridge > node);
- A-field activity **localizes along the bridge** (A energy concentrated on the corridor);
- all bounded: er ∈ band, curvature bounded, no A DC/k=0 runaway, seed/perturbation robust.

Classifications: `A_CURRENT_ROUTING_VALIDATED` (JAX-tier) / `A_CURRENT_PARTIAL` /
`A_CURRENT_NO_EFFECT` / `A_CURRENT_RUNAWAY_REJECT`.

## 6. Decision gate

Implement this branch ONLY after: (a) the scalar rescue test confirms scalar A does not rescue
selective routing (expected), and (b) this RFC's coupling form is accepted. Then implement,
verify γ_A=0 equivalence, run the sweep, and report. Do not claim "A-field proves FMIA" — the
claim under test is: *does a current-coupled interaction-rate field convert the holistic web into
directed, time-resolved routing?*
