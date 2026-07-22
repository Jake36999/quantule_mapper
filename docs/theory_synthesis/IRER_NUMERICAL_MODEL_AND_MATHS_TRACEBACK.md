# IRER Numerical Model & Maths Traceback

**Report 2 of 5.** A "Phase-A-style maths sanity check" for the modern, multi-script Quantule Mapper engine: for each
sector it states the *actual equation being integrated*, the code that computes it, the observable that decides
success/failure, and the current verdict. The goal is that no future agent can wave at "IRER physics" — each claim
must name an equation, an implementation, a metric, and a result. **All function/module paths are as-of this
milestone; a Codex traceback audit should re-verify exact names (marked `NEEDS_REVIEW` where I am less certain).**
Not a claim of physical truth — a claim about *what the code integrates*.

## 0. Common substrate
- Field: complex ψ on an N³ periodic box, side L, FP64/complex128. Spectral (FFT) spatial derivatives.
- Nonlinearity (cubic–quintic–septic Ginzburg–Landau family): **g(ρ) = a·ρ + s·ρ² + f·ρ³**, ρ=|ψ|²,
  params `param_a, param_s, param_f` (feb reference: a≈0.552 (×1.15 at a\*), s≈0.013, f≈−0.486).
  Code: `jax_scout/physics.py::_nonlinear_rhs`; KG: `phase_d_c3_wave.py::_g`.
- Dealiasing: 2/3-rule spectral mask.

## 1. Sector traceback table
| # | theory claim | equation integrated | code (module::fn) | key observable | validation gate | result | caveat |
|---|---|---|---|---|---|---|---|
| **C0** dissipative baseline (Phase C) | resonant structures stabilize by gain/loss balance | i ψ_t = L_k ψ + N(ρ)ψ, L_k = −Dk² − η + iω₀ (ETDRK4) | `physics.py::_construct_ops` (dissipative branch), `::n_op`, `::step` | er-slope→0, node count | late-window slope ≤1.5e-4 | a\* stable attractor (**CONFIRMED**) | knife-edge balance; site-pinned |
| **C1** dispersive kinetic term | adding complex diffusion mobilises a\* | L_k gains −i·D_imag·k² | `_construct_ops` (D_imag branch); `phase_d_c1_transport.py` | mobility μ=dv/dk; mass | long-T stability | destabilises, mass runaway (**FALSIFIED**) | dispersion↔stability incompatible |
| **C2** conservative NLS | a Hamiltonian substrate lets structures move | i ψ_t = −D∇²ψ − g(ρ)ψ; L_k=−i·D·k², nonlinearity ×kfac=1j | `_construct_ops` (kinetic_mode="conservative"); `phase_d_c2_*` | mass, boost velocity | dt CFL; parity | supports solitons (**CONFIRMED**) after C2.6 fix | needs dt≈1e-3; quasi-conservative geometry-on |
| **C2-geom** geometry coupling | density sources a conformal metric acting on the field | RHS += D·geom_fac·(lap_cov − lap_flat), lap_cov = Laplace–Beltrami(Ω²δ) | `physics.py::_cov_laplacian`, `_nonlinear_rhs`, `_geometry_with_gradient` | norm flux; T_info | RHS flux probe | quasi-conservative (**REVISED**) | not flat-self-adjoint (Report 3 §7) |
| **C2-flat** true geometry-off | `param_geom_off` gives a genuinely flat NLS | geom_fac=0 ⇒ lap_cov−lap_flat term vanishes | `physics.py` (`Ops.geom_fac`, `build_operators` param_geom_off) | one-step phase; packet transport | Galilean identity v=2Dk | true flat (**CONFIRMED**); v=2Dk exact | replaces the buggy `a_coupling=0` path |
| **C3** Klein–Gordon / Q-ball | second-order dynamics gives the field configurational inertia | ψ_tt = c²∇²ψ − m²ψ + g(ρ)ψ | `phase_d_c3_wave.py::build_kg,::kg_evolve` (Strang: exact per-mode SHO rotation ω_k²=c²k²+m² + nonlinear half-kicks) | E, Q, P; centroid velocity | linear parity; E/Q drift | conservative + inertial transport (**CONFIRMED**) | quasi-Q-ball breathing limits v-fidelity |
| **Ω²(ρ)** conformal law | the informational manifold is a density-sourced geometry | Ω² = clip((ρ_vac/ρ)^a_coupling) (log-tanh soft-clip) | `gravity/unified_omega.py::derive_stable_conformal_factor` | Ω² field; radial profile | (characterization only) | saturation cliff, not graded (**gravity PAUSED**) | soft-clip dominates; vacuum→cap |
| **T_info** informational stress tensor | coherent loads carry an informational stress-energy | T_ij = κ(∂_iφ)(∂_jφ) − δ_ij·L, L = ½Σ|∇φ|² − ρ² | `metrics/tensor_validation.py::construct_T_info` (+ symmetry, shear tests) | shear, anisotropy, symmetry err | symmetry ~0 | read-only diagnostic (used in gravity A/D) | phase-gradient stress; not the Ω² geometry |

## 2. Observables (how success/failure is measured)
| observable | definition | code | supersedes | why |
|---|---|---|---|---|
| **momentum-density velocity** | v = 2D·(∫Im ψ*∂ₓψ)/(∫ρ) over a COM-following window | `phase_d_c2_9_two_node_robust.py::profiles/core_com`; `phase_d_c3_two_qball.py` | projected-density peak-tracking | integral, robust when cores overlap/breathe (peak-tracking gave unphysical elasticity e=3.21) |
| **windowed centroid separation** | circular-mean COM of |ψ|² per core, minimal-image separation | same | naive two-peak finder | periodic-safe; tracks through merger |
| **radiation fraction** | 1 − (mass within W_WIN of either core)/(total mass) | `phase_d_c3_collision_ladder.py::collide` | — | quantifies shed halo in violent collisions |
| **conservation telemetry** | E, U(1) charge Q=Im∫ψ*π, momentum P=−Re∫π*∇ψ, mass ∫ρ | `phase_d_c3_wave.py::invariants` | — | the Strang splitting-error meter (dE/E ~1e-13 at rest) |
| **VK stability slope** | dQ/dω from Q(ω)=ω∫|φ|² across neighbouring Q-balls | `phase_d_c3_wave.py` (G6) | — | dQ/dω<0 ⇒ linearly-stable branch |

## 3. Boost initial conditions (the transport tests — and where the bugs hid)
| substrate | correct moving-structure IC | code | failure mode caught |
|---|---|---|---|
| C2 NLS | ψ₀ = φ·e^{ikx}, k=2πn/L (Galilean boost); v_expected = 2Dk | `phase_d_c2_2_loss_source.py` boost; `_c2_9` | the "geometry-off" substrate was secretly D_eff=D/151 → v=2·D_eff·k looked like "drag" (C2.6) |
| C2 local | phase ramp ∇φ≈k across the core, zero net winding | `phase_d_c2_4_local_boost.py::local_phase` | (used to test flow-through; verdict retracted post-C2.6) |
| C3 KG | ψ₀=φ(γx)·e^{ikx}, k=γωv/c²; π₀=(−v∂ₓφ_c − iγωφ_c)e^{ikx} | `phase_d_c3_wave.py` (G4) + `contract_axis0` | naive kick π=−v∇φ−iωφ (no carrier phase) gave spurious v_frac≈0.04 |

## 4. Collision classifiers (C2.9 / C3 / C2.10 two-body)
| classifier | rule | code | outputs |
|---|---|---|---|
| static force | sep decreasing = ATTRACT, increasing = REPEL, crossover at Δφ=π/2 | `_c2_9`, `_c3_two_qball` | attract/repel/hold/merge |
| collision outcome | overlap+re-separate coherent = PASS_THROUGH; overlap+bound = CAPTURE; repel-before-overlap = BOUNCE; incoherent = DISRUPT; E/Q (or mass) drift too high = INCONCLUSIVE | `phase_d_c3_collision_ladder.py::_classify` (KG); `phase_d_c2_10_antiphase_collision.py::_classify` (NLS, RUN-2) | 5-way + elasticity, radiation, identity-ambiguity flag |

*RUN-2 classifier refinements (C2.10, both fixed + re-verified from saved trajectories):* the overlap/merge scale is
the **soliton core** (`MERGE_SEP≈3.5`), not the tracking window (`2·W_WIN=7`); and re-separation is judged by the
**post-min peak** separation, not the final frame — a periodic-box pass-through pair separates fully then wraps back,
so `sep_end` understates it. Both mislabels (BOUNCE→CAPTURE, PASS_THROUGH→INTERMEDIATE) were caught and corrected.

## 5. What this traceback establishes
1. **The engine is a small family of well-posed PDEs**, not a monolithic "adaptive" black box: dissipative
   Ginzburg–Landau (C0/C1), conservative cubic-quintic-septic NLS (C2, geometry-on and true-flat), and nonlinear
   Klein–Gordon (C3), all sharing the same nonlinearity g(ρ) and spectral machinery.
2. **Every published verdict is tied to a named observable with a conservation/parity gate.** The strongest results
   (C2.7 v=2Dk, C3 E/Q ~1e-13, VK slope) are identities checked against the code, not narrative.
3. **The geometry sector is the one place the implemented law diverges from the documented law** (soft-clip
   reshaping of Ω²(ρ)) — flagged, characterized, and the reason the gravity ladder is paused.

## Cross-references
Concept meanings: Report 1. Identities/proofs (incl. the derivations behind v=2Dk, the D/151 bug, dQ/dω): Report 3.
Why the engine is split this way: Report 4. Results/status: Report 5 + `docs/IRER_MASTER_HYPOTHESIS_CATALOG.md`.
Per-sector detail: `docs/PHASE_C_*`, `docs/PHASE_D_C1..C3_*`, `docs/PHASE_D_C2_6_GEOMETRY_OFF_BUG_REPORT.md`,
`docs/IRER_GRAVITY_RUNG_A_D_RESULTS.md`.
