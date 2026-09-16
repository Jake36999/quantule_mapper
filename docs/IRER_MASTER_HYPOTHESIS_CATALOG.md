# IRER Quantule Mapper — Master Hypothesis Catalog & Progress Tracker

**Purpose:** a single, centralized record of every hypothesis the project has tested — confirmed, falsified, null,
retracted, or open — so the whole arc is legible at a glance. Includes the quickly-falsified and the
instrument-bug-invalidated ones on purpose: the nulls and retractions are as much of the map as the positives.
Last updated **2026-07-18** after the TG-B1S/TG-B2 temporal-geometric dual-substrate implementation (§8C), Phase-R
robustness, the MC concept cross-maps, clock-migration and source-semantics sprints, the TG-B2 characterization, and
three independent review audits (external-physics-object comparison; consolidated progress/validation audit;
provenance×workflow assessment). Those audits' central finding is folded into the §8C verdicts and §11:
**numerical robustness currently leads theory fidelity — several headline characterizations advanced ahead of quieter
validation gates, and some prior prose (mass-independence, force-range, "independent" confirmations, the MC-1 yield
reframe) is stronger than the data support and is corrected here.** (Prior "Last updated 2026-07-14": Gravity D
characterization + Codex TG feedback-loop formalization.)

## 0. Executive assessment
Two of the three IRER simulation sectors are answered; the third is scoped and paused.
- **Stability sector (Phase C): CLOSED.** A real, long-time, gain/loss-balanced attractor (a\*≈×1.15) exists and is
  site-pinned. Every mobility, resonance, topology, and routing hypothesis around it came back null.
- **Transport/coupling sector (Phase D): ANSWERED POSITIVE across two conservative substrate families.** After a
  pivotal instrument bug was found and fixed (C2.6), true solitons (NLS) translate at exactly v=2Dk, and Q-balls
  (Klein–Gordon) transport inertially and are VK-stable. Both substrates share the same two-body relational law
  (π/2 phase-force crossover) AND the same collision phase×speed *diagram* (RUN-2): capture is generic, with an
  exact-anti-phase destructive-interference node that transmits (pass-through) at intermediate speed and captures
  above a sharp threshold — cross-substrate universality now spans the full collision structure, not just the static
  force.
- **Gravity-like sector: SPATIAL MIRROR CHARACTERIZED + PRODUCTION CLOSED.** The standalone Gravity D branch now
  has a reproduced, converged, robustness-closed and dynamically characterized spatial effective-medium force:
  `F = -D integral grad(N)|grad psi|^2 dV`. It is finite-width, probe-structure-dependent and non-Newtonian. The
  production gravity ladder remains closed; temporal lapse, nonlocal source fields and IRER source semantics remain
  separate open hypotheses.
- **Temporal-geometric dual-substrate (TG-B1S/B2): FIRST CLOSED LOOP IMPLEMENTED; ONE ROBUST, NON-GRAVITATIONAL
  RESULT (§8C).** The full chain `S_state → T → G → A(G) → force` now runs dynamically. On the theory-faithful
  A-well branch it produces a numerically-hardened, externally-legible **in-phase inter-node attraction** — a
  normalized-charge, extended-source, screened-medium force (BEC smeared-Yukawa family), **not gravity** (source
  charge not mass-proportional; no 1/r²; no UFF; no N_t lapse on the field). It does **not yet** establish secular
  binding, an alignment law, a resolution-rate mechanism, or a yield transition. **Numerical robustness leads theory
  fidelity**; the next work is to factor source/probe/observable apart and close the preregistered contracts (§11),
  not to tune toward gravity.
- **Scientific posture:** no matter claim, no gravity claim, no emergent-physics claim. Every "transport",
  "interaction", "response" is a specific measured observable with conservation telemetry. Three instrument bugs were
  caught by chasing contradictions rather than shipping convenient nulls.

## 1. Verdict legend
| tag | meaning |
|---|---|
| **CONFIRMED** | supported by the measured observable, reproduced/robust |
| **FALSIFIED** | tested and contradicted |
| **NULL** | tested, no signal (a real, informative negative) |
| **INCONCLUSIVE** | tested but the instrument/measurement could not decide |
| **RETRACTED** | a prior verdict later invalidated by a discovered bug/finding |
| **OPEN** | designed/motivated, not yet run |
| **DESIGN-ONLY** | RFC/plan, deliberately not implemented |
| **PAUSED** | started, halted at a documented gate/re-entry condition |

---

## 2. Infrastructure & method
| # | hypothesis / task | verdict | evidence |
|---|---|---|---|
| I1 | DC-v1.0 data contract can be hardened to compliance | CONFIRMED | 16/16 compliant; test-bench scanner |
| I2 | There is a separate "CuPy box" for production | FALSIFIED | CuPy runs on this PC via repo `.venv` (cupy 14.0.1, GTX 1080); PATH-python's cupy was just ABI-broken |
| I3 | CuPy (FP64) and jax_scout mirror share the identical operator | CONFIRMED **(re-verified 2026-09-16)** | H4/A1 parity rel-L2 1.7e-12; C1 parity byte-exact. **Re-run 2026-09-16 against current code: rel-L2 1.735e-12, max\|Δ\| 4.58e-13, `PARITY_WITHIN_TOL` — reproduces the original to two significant figures across 38 commits to `jax_scout/`.** Procedure and artifact: `docs/SOLVER_PARITY_ARTIFACT.md`, `tools/solver_parity_check.py`. |
| I4 | Production stability_metrics reach the provenance/validation path | CONFIRMED | A3/A4/A4b wiring accepted |
| I5 | The physics-identity CI guards the C2.6 / C2.8b / C3 failure classes | **FALSIFIED then REPAIRED (2026-09-16)** | `tools/mutation_probe.py`: 10 mutations shaped like the ledger's own bugs. **7 of 10 survived**, incl. all three sign flips and the `D_eff = D/151` shape. The suite pinned symmetries (off-is-off, antisymmetry, charge conservation, finiteness) and never a value, and every survivor preserved a symmetry. Two structural causes: no test ever set `geom_en = 0`, and `A` is written twice (dynamics + observer) so pinning the observer left the dynamics free. Six value-pinning tests → **10/10 caught**. `docs/gravity_maturity/H3_MUTATION_PROBE_RESULTS.md` |

## 3. Phase C — stability sector (CLOSED)
| # | hypothesis | verdict | evidence |
|---|---|---|---|
| C-1 | a\*≈×1.15 cubic gain is a genuine long-time stable attractor | **CONFIRMED** | bracketed ±0.5%, seed/N128/T144k; er-slope→0 |
| C-2 | a\* is mobile — a kick imparts net motion | **FALSIFIED** | kick null; L_k=−Dk²−η real → no inertial channel |
| C-3 | a static potential well relocates a\* | **FALSIFIED** | response = accretion/nucleation, not migration (3 morphologies) |
| C-4 | log-prime resonance (prime-SSE) predicts stability | **NULL** | 0/60; objective retired |
| C-5 | TDA / topological features predict stability | **NULL** | ~0 |
| C-6 | tensor-routing: geometric bridges route current | **NULL (NO_SUPPORT)** | proxy effect was a flat-form artifact (Stage B) |
| C-7 | Payan (axial ∇φ phase-winding) predicts bridge stability | **NULL (NO_SIGNAL)** | alignment gap −0.08 |
| C-8 | nodes are topological vortices | **FALSIFIED** | hi-fi continuation: rotational cores, not topological; stable=energy balance |
| C-9 | strong geometric bridges route (bridge/void ratio) | **NULL** | denominator invalid (no-bridge=949) |
| C-10 | GL rotational cores form a self-sustaining basin | CONFIRMED (partial) | eta-dominated band; 210/1920 sustain; dissipative solitons |
| C-11 | a re-aimed stability objective re-discovers a\* without prime-SSE | **CONFIRMED** | H7.3 RE-DISCOVERY PASS (rank 0, score 0.867) |
**Sector verdict:** a\* is a **fundamentally stationary dissipative attractor**; stability and transport are
incompatible in this sector. Every non-stability structural hypothesis (prime/TDA/routing/Payan/vortex) is null.

## 4. Phase D — dissipative sub-sector (D.1–D.6)
| # | hypothesis | verdict | evidence |
|---|---|---|---|
| D1 | C1 dispersive (complex-diffusion) kinetic term mobilises a\* | **FALSIFIED** | destabilises not mobilises; mass runaway 1.03→2.52 (`C1_NO_STABLE_TRANSPORT`) |
| D2 | the informational stress tensor T_ij is a node-coupling signal | **INCONCLUSIVE** | not a pure density proxy, but no clean coupling in feb/a\* |
| D3 | stable nodes carry current | **FALSIFIED** | near-current-free; current only in growers/blowups |
| D4 | nodes couple via a short-range density/geometry channel | **CONFIRMED** | conductance∝1/spacing, radius ~0.5 box (survives density-null) |
| D5 | two nodes drift / show relational motion | **FALSIFIED** | merge<0.3 else hold, no drift (`TWO_NODE_MERGE_OR_HOLD`) |
| D6 | the sector coarse-grains to a pinned/merging network | **CONFIRMED** | reduced model reproduces 4-node cap (tendency-level) |
**Sector verdict:** dissipative nodes **couple + merge + phase-lock but never move** — transport null at single-node
(C1) and pair (D5) level. Genuine transport needs a different substrate.

## 5. Phase D — conservative NLS sub-sector (C2 arc)
| # | hypothesis | verdict | evidence |
|---|---|---|---|
| C2-0 | a\* carries over as a conservative soliton | **FALSIFIED** | radiates ~50%, μ≈0 in the Hamiltonian substrate |
| C2.1 | the conservative substrate has its own native solitons | **CONFIRMED** | wide σ=0.15 solitons hold at N=96 |
| C2.1b | those solitons transport ballistically ("lossy ballistic") | **RETRACTED** | (C2.6 bug) the μ≈0.04 was 2·D_eff of a broken substrate |
| C2.2 | the density-sourced geometry breaks Galilean invariance → radiation | **FALSIFIED** | geometry-off identical loss |
| C2.2b | the residual loss is purely dt-numerical | **RETRACTED** | measured on the bugged substrate |
| C2.3 | an exact stationary soliton exists (imag-time/Petviashvili) | **FALSIFIED** | flows to uniform; no branch above g_max≈0.23 → metastable quasi-soliton |
| C2.3b | the velocity anomaly = topological ring-winding drag | **RETRACTED** | (C2.6 bug) |
| C2.4 | a local (zero-winding) boost moves the soliton; else "flow-through pinning" | **RETRACTED** | (C2.6 bug) the pinning was an artifact |
| **C2.6** | **`param_a_coupling=0` turns geometry OFF** | **FALSIFIED (THE BUG)** | soft-clip squash maps Ω²=1→~151 → D_eff=D/151; the "drag" μ=2·D_eff exactly |
| C2.7 | on the fixed substrate, a true soliton translates at v=2Dk | **CONFIRMED** | v/2Dk=0.9999, mass 0.9999 at N=96 |
| C2.7b | feb/a\* has native conservative solitons | **FALSIFIED** | structureless (g_max 0.23 ≪ box binding floor 0.44) |
| C2.7c | moving-soliton families exist for some (a,s,f,D) | **CONFIRMED** | 2 GALILEAN families (s<0 saturation + box-compatible D) |
| C2.8/8b | two-node elasticity from projected-density peak-tracking | **INCONCLUSIVE** | fragile tracker (unphysical e=3.21; false-repel static) |
| C2.9 | (robust momentum observable) static pair force has a π/2 crossover | **CONFIRMED** | attract Δφ<π/2, repel >π/2, crossover π/2 |
| C2.9b | collisions are capture-dominated (non-integrable binding) | **CONFIRMED** | asym 3/2 & 4/3 both capture, mass/P conserved |
| C2.10 (RUN-2) | the NLS anti-phase collision has the same node channel as C3 | **CONFIRMED** | in-phase captures at all speeds; anti-phase = BOUNCE (closing≲0.42) → PASS_THROUGH (0.47–0.57) → CAPTURE (≳0.61); phase-driven (in-phase control captures where anti-phase bounces) |
| C2.10b (RUN-2) | the NLS substrate has an anti-phase transmission (pass-through) window | **CONFIRMED** | cores reach the midplane node (sep_min≈0.1) & re-emerge coherently, ≤7% radiation; capture threshold sharp, bracketed to closing (0.565, 0.660) |
| C2.10c (RUN-2) | the collision phase×speed *diagram* (not just the static law) is cross-substrate universal | **CONFIRMED** | NLS reproduces the C3 structure — capture-generic + exact-anti-phase pass-through→capture |
**Sector verdict:** the conservative NLS substrate **supports clean coherent transport** (v=2Dk) for solitons that
satisfy s<0 + box-fit; two-body dynamics = phase-force (π/2) + capture, with an exact-anti-phase node channel
(pass-through → capture) that mirrors C3 — the whole collision diagram is cross-substrate universal (RUN-2). feb/a\*
itself is structureless there.

## 6. Geometry contract (Codex adjointness thread)
| # | hypothesis | verdict | evidence |
|---|---|---|---|
| G-1 | the C2 covariant correction is norm-conserving / self-adjoint (flat) | **FALSIFIED** | it is Laplace–Beltrami of Ω²δ — self-adjoint w.r.t. Ω³, not flat (mismatch 1.09e-1) |
| G-2 | a simple Ω-power weighted norm is the conserved invariant | **NULL (NO_WEIGHTED_INVARIANT_FOUND)** | live Ω(ρ) → no fixed weight conserved; sqrt_g near-pass 2.15e-4 |
| G-3 | C2 is "conservative" as labelled | **REVISED** | linear-conservative / nonlinear quasi-conservative (geometry-exchange) |
| G-4 | C2′ canonical (divergence-form + metric-variation) would conserve exactly | **DESIGN-ONLY** | RFC; not implemented |

## 7. Phase D — wave-kinetic KG sub-sector (C3)
| # | hypothesis | verdict | evidence |
|---|---|---|---|
| C3-1 | a 2nd-order (KG) substrate gives the field configurational inertia | **CONFIRMED (framing)** | momentum in ψ_t, not phase → flow-through impossible |
| C3-2 | the Strang stepper is exact vs analytic mode rotation | **CONFIRMED** | G1 max\|Δ\|=0 |
| C3-3 | a Q-ball exists (Petviashvili-in-ω) | **CONFIRMED** | at the C2.7-mapped point (KG stationary eq ≡ C2 soliton eq) |
| C3-4 | E and U(1) charge are conserved to machine precision | **CONFIRMED** | dE/E~9e-14, dQ/Q~1e-13 |
| C3-5 | the Q-ball transports inertially (density co-moves under boost) | **CONFIRMED** | v_frac~0.9 after fixing the carrier-phase boost IC |
| C3-5b | the missing Lorentz contraction is the transport-fidelity limiter | **FALSIFIED** | contraction added → v_frac unchanged; limiter is measurement (breathing) |
| C3-6 | the Q-ball is VK-stable (not a transient oscillon) | **CONFIRMED** | dQ/dω=−849<0 |
| C3-7 | the KG two-Q-ball obeys the same law as the NLS pair | **CONFIRMED** | π/2 static crossover + capture → cross-substrate universality |
| C3-8 | in-phase collisions have a critical velocity / pass-through | **FALSIFIED** | capture-dominated to 0.75c (relativistic); higher v → more radiation, still capture |
| C3-9 | the repulsive (Δφ>π/2) channel transmits | **FALSIFIED (as stated)** | statically-repulsive 3π/4, 7π/8 still capture |
| C3-10 | transmission exists at all in the substrate | **CONFIRMED (narrow)** | only at Δφ≈π (exact anti-phase) & v≲0.45c — a destructive-interference NODE feature |
| C3-11 | the static force law governs the collision outcome | **FALSIFIED** | static crossover π/2 ≠ collision boundary; only the exact-anti-phase node survives |
**Sector verdict:** the KG substrate is genuinely conservative, hosts a **VK-stable Q-ball that transports
inertially**, and its collisions form a **relational phase diagram** — capture generic, transmission a narrow
anti-phase node channel. Same π/2 two-body force as NLS (universality); cleaner conservation. **RUN-2 (C2.10)
confirmed the NLS substrate reproduces this collision diagram (anti-phase pass-through → capture), so the
universality extends from the static two-body law to the full collision phase×speed structure.**

## 8. Gravity-like sector (geometry-density)
| # | hypothesis | verdict | evidence |
|---|---|---|---|
| GR-A/D | a coherent load creates a persistent, geometry-dependent Ω²/T_info response | **CONFIRMED (entangled)** | geometry-off → load disperses, shear 1.35→1e-33; but entangled with load-stability |
| GR-1 | that response is a graded (gravity-like) potential | **FALSIFIED** | it is a saturation cliff (Ω²~750 core → ~9e5 at r>1.35) |
| GR-2 | de-saturating the soft-clip (β=0.75, wider window) makes it graded | **FALSIFIED** | made the spatial cliff 250× steeper; cliff is (ρ_vac/ρ)^a diverging at near-zero vacuum, not the cap |
| GR-3 | C2′ canonical geometry unblocks the gravity ladder | **FALSIFIED** | same Ω²(ρ) factor → same cliff |
| GR-4 | a bg=ρ_vac ambient gives a clean unsaturated well | **FALSIFIED** | the load dissolves on a filled background; soft-clip Ω²(ρ_vac)=727 anyway |
**Sector verdict:** **PAUSED.** Channel is live but uninterpretable in the production geometry (saturation cliff).
Re-entry condition: a **validated stable-overdense-load-on-ρ_vac-background** regime (a research sub-project) before
any rung B. Cautious framing preserved: *dense sequential geometry-mediated interaction around coherent mass-like
field loads*, not "mass attracts mass"; not yet an emergent-gravity claim.

## 8A. Gravity D spatial effective-medium mechanism (standalone mirror)
| # | hypothesis | verdict | evidence |
|---|---|---|---|
| GR-D-1 | bounded spatial coefficient gradients generate a finite-width wave force under `i d_t psi = -D div(N grad psi)` | **CONFIRMED / SUPPORTED** | GPU replication, convergence, robustness closure, and exact force contract: `d<P>/dt = -D integral grad(N)|grad psi|^2 dV` |
| GR-D-2 | the force is well described by a gradient-energy-weighted coarse-grained law | **SUPPORTED (PARTIAL REDUCED MODEL)** | dynamics characterization: `F_cg ~= -D K_grad grad N(R)`; best model R2=0.9913, normalized RMSE=0.0931; 6/193 sign errors in far-tail/near-null rows |
| GR-D-3 | the characterized spatial operator produces a Newtonian exterior vacuum field | **REJECTED FOR THIS MODEL** | inverse-square benchmark R2=0.0206; compact-source far exterior force ~1.66e-13 |
| GR-D-4 | the characterized spatial operator obeys Newtonian shell-theorem behaviour | **REJECTED FOR THIS MODEL** | smooth shell interior shows local-medium response; interior force max ~3.09e-02 |
| GR-D-5 | the characterized spatial operator establishes universal free fall | **REJECTED FOR THIS MODEL** | probe internal structure changes force by ~54.6%; width/gradient-energy dependence is intrinsic |
| GR-D-6 | a global point-ray description explains the characterized dynamics | **REJECTED FOR THIS MODEL** | ray benchmark has R2=-0.1085 and sign accuracy ~0.109 globally |
| GR-D-7 | metric-consistent temporal lapse, nonlocal environment field, and IRER-derived source semantics explain gravity | **OPEN / SEPARATE HYPOTHESES** | not tested by the spatial effective-medium campaign; production gravity remains closed |

**Sector update:** the standalone spatial effective-medium branch is now characterized, but it is not Newtonian or
relativistic gravity. It is a robust finite-width wave response to bounded coefficient gradients. Do not promote it
as geodesic validation, temporal-lapse validation, universal free fall, IRER gravity-source confirmation, or
production readiness.

## 8B. Temporal-geometric feedback loop (Codex formalization addendum, 2026-07-14)

This section records an additive formalization pass, not a new simulation result. The source trace is
`docs/theory_synthesis/IRER_TEMPORAL_GEOMETRIC_FEEDBACK_LOOP.md`. Conceptual authorship remains Jake McIntosh; Codex's
role was to consolidate notation, map the hypothesis to existing evidence, and define falsifiable next tests.

| # | hypothesis | verdict | evidence / boundary |
|---|---|---|---|
| TG-1 | OIW phase-locking, RD/PAS rise, chronology load, geometric strain, and outgoing release form one closed temporal-geometric feedback-loop hypothesis | **OPEN / FORMALIZED** | original provenance contains the ingredients across OIW/RD/PAS, Quantules, time-as-resolution, Chrono-Coherence, Payan states, FMIA, manifold deformation, and splash-like redistribution; no full closed-loop solver yet |
| TG-2 | the current C-series, Gravity D, bridge, or G1 runs already implement the full `Psi -> R_res -> T -> G -> Psi` recursion | **REJECTED AS CURRENT STATUS** | existing work tests limbs separately: temporal throttling, temporal KG response, spatial effective-medium force, and temporal-spatial decomposition |
| TG-3 | dynamic temporal load can drive a geometric response that emits or modulates an outgoing perturbation at a measurable frequency | **OPEN / DESIGN-ONLY** | proposed TG-A/TG-B scout; no modern frequency-gated two-field run yet |
| TG-4 | FMIA bridges/wires emerge as low-resistance decay routes under dynamic temporal-geometric stress | **OPEN / DESIGN-ONLY** | relates to FMIA and Informational Parallels, but previous tensor-routing/Payan predictors were null in the earlier static/proxy sector |
| TG-5 | photon-like emission is a minimal geometric relaxation mode, and emission pattern is a source-geometry projection | **OPEN / UNDEFINED OBSERVABLE** | recorded as `PHOTON_AS_MINIMAL_GEOMETRIC_RELAXATION_MODE` and `EMISSION_PATTERN_AS_SOURCE_GEOMETRY_PROJECTION`; requires precise outgoing-field observable before simulation |

**Current boundary:** this loop is relevant to the gravity-like sector because it is the first explicit notation for
mutual chronology/geometry feedback. It does not reopen production gravity, validate photons, validate geodesic
motion, or alter the banked Gravity D interpretation.

## 8C. Temporal-geometric dual-substrate — IMPLEMENTED loop (TG-B1S / TG-B2), 2026-07-14 → 07-18

The TG-1 loop (§8B) is now partly **built and run**: `state=(φ,π,T,V_T,G,V_G)`, chain `S_state → T → G → A(G) → δφ`,
with damped 2nd-order T/G fields and symmetric `−κTG` exchange. Two branches: **TG-B1S** (`A=exp(−ε_G G)` → A-hill;
the numerically-safe scaffold, later found to be the *anti-throttling* polarity) and **TG-B2** (`A=exp(+ε_G G)` →
A-well; the theory-faithful polarity: dense load → slower chronology → attraction). Conceptual authorship Jake;
implementation/measurement/review AI. Verdict language reflects three independent review audits (2026-07-18) that
corrected several over-strong prior labels — kept here deliberately, per the "nulls and retractions are the map" rule.

| # | hypothesis | verdict | evidence / boundary |
|---|---|---|---|
| TG-S1 | `S_state` (normalized energy+charge density) is an admissible **node state-load** source | **CONFIRMED (as a load)** | TG-S semantics: stable/nonzero on stationary node, phase/translation-invariant, frozen `S0`=135.686. It is a *load*, explicitly **NOT** a completed-resolution rate |
| TG-S2 | phase-locking (`L_lock`) operationalizes as a source | **FALSIFIED (semantics)** | failed the TG-S separation gate |
| TG-S3 | phase-tension relaxation (`R_relax`) is admissible as a resolution-**rate** source | **CONFIRMED (admissible-for-design only)** | passed the TG-S semantics bridge — but NOT shown to be the recovered `R_coh`, nor extensive in mass, nor correctly normalized; needs dimensional/null/sign/lag/energy contracts before B3 wiring |
| TG-B1S-1 | the `S_state→T→G→A→δφ` feed-forward chain is intervention-clean | **CONFIRMED** | source/temporal/geometric-off remove the right fields; translation-covariant; global-phase-invariant |
| TG-B1S-2 | the loop produces a bounded geometric backreaction | **CONFIRMED (robust but weak)** | SNR ~132; vanishes at zero coupling; smooth λ ladder; `STABLE_SHIFTED_NODE` |
| TG-B1S-3 | the single-node modal frequency shift (+2.15e-6) is a stable, box-independent, converged result | **FAILED (D4)** | D4 validation 5/6 pass; larger-box (L=12) fails the frequency-scale gate → `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`. On the **A-hill scaffold** branch (control sign), not the live A-well direction |
| TG-B1S-4 | the frequency shift is a **temporal** (time-dilation) effect | **FALSIFIED / REFRAMED** | FC-1 frequency contract: the shift is **geometric** (A-stiffening), bracketed by fixed-profile (0.25×) and relaxed (2.18×) analytic limits; the earlier sign "puzzle" was a reporting-convention artifact. **No `N_t` lapse acts on φ** (code-verified) — temporal effects reach the field only via geometry |
| TG-B1S-5 | the D4 larger-box failure is a resolution or local-mechanism effect | **FALSIFIED** | box-dependence discriminator: dx-independent (not resolution); local mechanism predicts the opposite sign → comparability/absorber/dynamical. It is the A-hill scaffold's frequency observable, **distinct** from the production saturation cliff (§8) and from the A-well direction |
| TG-B2-1 | the theory-faithful **A-well** polarity produces in-phase inter-node **attraction** (dynamical body force) | **CONFIRMED (implementation, robust)** | `<F_R_well>` negative at 100% of settled samples (both seps); sign-flip control (A-hill repels); exact off-null. Validates the A-well *implementation + robustness*, **not** the theoretical polarity independently |
| TG-B2-2 | that attraction survives Phase-R robustness | **CONFIRMED (scoped)** | grid N96, dt/2, larger-box L20 (2.45% drift), ε_G-linear (0.5×/1×/2×) all pass **for the in-phase body force**. SCOPE NOTE: normalization-consistency, dealiasing, placement are **not all closed**; phase-invariance is a *distinct physical hypothesis* (TG-B2-4), not a numerical control |
| TG-B2-3 | the attraction produces secular **pair binding** | **UNRESOLVED (attractive lean)** | body force attracts + separation closes (both cool_T) — but these are **downstream of the same trajectory, not independent votes**; the momentum proxy is invalid as a direction meter; cooling@T=40 did not quiet the breathing. Binding needs bound-state energy over turning cycles + a successful cooling precondition |
| TG-B2-4 | the loop force follows a clean alignment law `a·cos(Δφ)+b` | **FALSIFIED (dynamically)** | quasi-static fit failed free dynamics: attract@0, **REPEL@π/2**, noise beyond. Δφ≠0 drives relative motion (max@π/2) → motion-confounded; only Δφ=0 is clean. Falsifies the **TG-loop** cosine law only; the bare NLS/KG phase-force (§5, §7) is unaffected |
| TG-B2-5 | F(sep) identifies the force **range / falloff law** | **INCONCLUSIVE (shallow near-field)** | 4 near-field points (r 2.5–5): `|F(5)|/|F(2.5)|=0.75`; local power p≈0.41 ≈ exp λ≈8.6, indistinguishable. Asymptotic law + physical range **unresolved**; the original coupled-mode range ~0.8 expectation was **not reproduced** — needs r=6–8+ in a large box with an SNR stop rule |
| TG-B2-6 | F(mass) reveals **gravity-like mass scaling** | **FALSIFIED-as-gravity / otherwise UNRESOLVED** | no mass trend resolved (p_M≈−0.07, R²=0.11); **incompatible with equal-mass M²**. The w-sweep conflates mass/amplitude/width/breathing/source/probe — "mass-independent" overstates it |
| TG-B2-6a | the flat `M⁰` force is caused by the **normalized `S_state` source** (my earlier causal claim) | **FALSIFIED (P1 charge audit)** | audit `TG_B2_CHARGE_AUDIT_...`: normalization removes only ~0.35 powers (raw load `M^0.94` → normalized source `∫S_state ~ M^0.59`, still **scaling**); receiver `∫\|∇φ\|² ~ M^1.10`. Naive `Q_src·Q_recv ~ M^1.69` vs measured `M^−0.07` ⇒ a **~1.7-power cancellation from the near-field kernel** at sep=3.0, **not** the normalization. Mass-flatness is **separation-specific**, not intrinsic — needs a far-field mass sweep + measured-kernel convolution (Gate 3). `F_R` classified: a total half-space **force** (not acceleration/mass-normalized) |
| TG-B2-7 | the in-phase force is an externally recognizable physics object | **CONFIRMED (legible, non-gravitational)** | = a normalized-charge, extended-source, screened-medium attraction; closest analogue the BEC smeared-Yukawa (Girelli–Liberati–Sindoni 2008); **not** a gravitational force law (source charge not ∝ mass; acoustic metrics don't supply this force) |
| TG-B2-8 | an **independent** force instrument (midplane stress flux) confirms the body-force sign/scale | **OPEN** | planned in the consolidation framework; not yet run. Currently **one** well-hardened force channel, not multiple independent confirmations |
| TG-clock-1 | the G1 clock-mode migration toward the source is a real field response (not just an instrument fault) | **CONFIRMED (directional response detected)** | 5/5 near-clocks migrate toward source; flat control null 7.7e-11. **NOT** a clock-rate/lapse/time-dilation law — it is spatial relocation; needs distance/strength/grid/box scaling before informing any `N_t` |
| MC-1 | the Gravity-D non-Newtonian "failures" are IRER-**predicted** force behaviour (gradient-derived + saturating-yield + non-universal) | **PARTIAL: class-match SUPPORTED; yield/saturation SPECULATIVE + DOWNGRADED** | the gradient-derived-force *class* match holds (GR-D-2). BUT the yield-point pilot is **INCONCLUSIVE** (no onset), the production cliff has a simpler encoded cause (`(ρ_vac/ρ)^a` vacuum divergence), and the "cap out / elastic yield" language is **AI-proposed future-expansion (Weight C), not an author-derived prediction.** Non-Newtonian failures are **not** automatic IRER confirmations |
| MC-2…6 | recovered-concept cross-maps | **corrected (see crossmap)** | MC-2 DII → *candidate* operational interpretation (mixed AI lineage); MC-3 → bare phase-force supported / TG-loop cosine **unsupported**; MC-4 speculative, collision-only; MC-5 untested-in-transport, low priority; MC-6 provenance anchor / test-direction, not a confirmation |

**Sector verdict (TG dual-substrate):** the first *implemented* closed TG loop yields a **numerically robust,
externally-legible, in-phase effective-medium attraction** on the A-well branch — a real, hardened property of the
equations. It is **not gravity** (source charge not mass-proportional; no 1/r²; no UFF), **not** a temporal-lapse
result (geometric shift only; no `N_t` on the field), and does **not yet** establish secular binding, an alignment
law, a resolution-rate mechanism, or a yield transition. **Numerical robustness currently leads theory fidelity.**

**Provenance-attribution note (Weight A/B/C/D — binding for all cross-maps).** The recovered-concept work separates:
**A** Jake-originated concepts · **B** Jake-adopted AI development · **C** AI-generated formalization candidates
(includes the RFD and FMIA *names*, `C_ij`, `g_ij(RD,PAS,φ)`, `F_info=m_info·a_PAS`, the **"cap out / elastic yield"**
language, the **DII name**, and the **`cos(Δφ)` law**) · **D** 2026 retrospective cross-maps. A concept or label being
present in the provenance does **NOT** make a later matching result an "IRER prediction" — a Weight-C match earns a
controlled test, not a confirmation. This tempers MC-1 and several cross-maps, and is why the entries above avoid
"IRER predicted X" framing.

| V7-a | The TG wave operator is an EXACT acoustic metric | **CONFIRMED (2026-09-17)** | `docs/gravity_maturity/S2_V7_ACOUSTIC_METRIC_RESULTS.md`. g_uv = diag(-c^3 A^{3/2}, c sqrt(A) x3), *determined* by a determinant condition, not fitted. Reproduces f^{uv} exactly; null speed c sqrt(A) matches the operator. |
| V7-b | The Gravity-D force law is a postulate | **FALSIFIED - it is DERIVED (2026-09-17)** | Geodesic of the reconstructed metric gives a = -(3/4) grad A / A, i.e. motion toward smaller A, matching the coded body force in direction for both polarities. The project's first genuine **structure transfer**. |
| V7-c | The acoustic metric derives the force SIGN | **NOT SUPPORTED (2026-09-17)** | The metric fixes the RESPONSE, not the SOURCE. `a_sign` becomes "does a load lower the lapse" - sharper, still chosen. The phi sector is variational (which is why the metric exists); the T-G back-coupling is not. **The sign problem lives exactly in the non-variational part of the loop**, so S3/GAP-4 is now the only route, not a fallback. |
| D3 | The screening falloff is chameleon, symmetron or Vainshtein | **FALSIFIED - none of the three (2026-09-17)** | m_eff^2 = (m^2 - U')/(c^3 A^{3/2}): the MATTER field's mass varies, while the mediator G has constant mass omega_G - the opposite assignment to a chameleon. Metric is not conformally flat (-g_tt/g_xx = c^2 A), so not a scalar-tensor mass variation either. Closed at zero compute. |

## 9. Hunter / objective
| # | hypothesis | verdict | evidence |
|---|---|---|---|
| H-1 | prime-SSE optimization finds stability | **NULL** | retired; prime structure never predicted stability |
| H-2 | a stability objective re-discovers a\* | **CONFIRMED** | live Hunter rediscovery pass |

---

## 10. Instrument-integrity ledger (the meta-catalog)
Three bugs were caught by chasing a contradiction against a known identity, not by accepting a convenient null.
Recording them because they reshaped several verdicts:
| bug | symptom | root cause | how caught | fix |
|---|---|---|---|---|
| **C2.6 geometry-off** | "pinning", "flow-through", μ≈0.04 drag | soft-clip squash → `a_coupling=0` gave D_eff=D/151 | a linear packet failed to translate (violates Galilean identity) | `Ops.geom_fac` + `param_geom_off` (default byte-identical; Codex RK4-reaudited) |
| **C2.8b tracker** | unphysical elasticity e=3.21; false-repel static | projected-density peak-tracking fails when cores overlap/breathe | e>1 violates energy conservation | momentum-density observable v=2D·∫Im(ψ*∂ψ)/∫ρ (C2.9) |
| **C3 boost IC** | Q-ball density barely moved (v_frac~0.04) | naive kick lacked the carrier phase that IS the momentum | constant v_frac independent of v | ψ₀=φe^{ikx}, k=γωv/c² → density co-moves |
**Retractions triggered:** C2.1b, C2.2b, C2.3b, C2.4 (all "conservative pinning/flow-through"); C2.8 "n=4 survives";
"the repulsive channel transmits" → refined to the anti-phase node.
**Same discipline in RUN-2 (caught in-flight, before any verdict was published — so not verdict-reshaping bugs):**
two C2.10 classifier thresholds were corrected against the physics — the overlap/merge scale must be the soliton
core (`MERGE_SEP≈3.5`), not the tracking window (`2·W_WIN=7`) (a clean bounce was reading as CAPTURE); and
re-separation must use the post-min *peak* separation, not `sep_end`, because a periodic-box pass-through pair
separates fully then wraps back (a pass-through was reading as INTERMEDIATE). Both caught by trajectory inspection
contradicting the label.
**Lesson (kept as a standing rule):** every transport/interaction claim is gated on a correct boost IC, a robust
(non-peak-tracking) observable, and conservation telemetry; a null that contradicts a symmetry identity is treated as
an instrument fault until proven physical.

> [!note] Interpretation note on cross-substrate universality (2026-08-25)
> The NLS≡KG two-body law and collision-diagram universality is a **verification** result, not
> support for IRER. Karpman–Solov'ev (1981) / Gordon (1983) derive the two-soliton interaction as
> `e^{−Δx}·cos(Δφ)`, whose sign change is at exactly π/2, so standard soliton perturbation theory
> predicts it. Reproducing it across two substrates with different symmetry groups is strong evidence
> the solver is doing real soliton physics — file it beside `v=2Dk` (0.9999) and conservation to
> 1e-13. It is **not** a discriminating prediction. See
> `docs/ACTION_PLAN_2026-08.md` Phase 2 and `docs/SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW.md` §5.

## 11. Current frontier & open threads (none blocking)
| thread | status | next |
|---|---|---|
| C3 asymmetric-velocity elasticity | OPEN | quantify captured-vs-radiated energy budget with the momentum observable |
| C3 captured-remnant long-time fate | OPEN | stable "Q-ball molecule" vs slow decay (machine-clean conservation makes it well-posed) |
| ~~C2 anti-phase collision (NLS analog of C3)~~ | **RESOLVED (RUN-2 → CONFIRMED, §5 C2.10)** | NLS reproduces the collision diagram: bounce → pass-through → capture; transition mapped, capture threshold bracketed to closing (0.565, 0.660) |
| C2 bounce→transmit boundary & transmission-window width | OPEN (refinement) | pin the ≈0.42 bounce/transmit boundary; whether KG has the same low-speed bounce sub-regime (C3 didn't sample <0.15c) |
| C2′ canonical geometry | DESIGN-ONLY | RFC; theory choices (§7) pending; only if a conservation-exact geometry is needed |
| Gravity ladder rung B+ | PAUSED | re-entry = stable-overdense-load-on-ρ_vac-background pilot |
| Codex replication | AVAILABLE | finalized C2.9/C3/collision harnesses ready for independent re-run; C2.6/C3 already re-audited |
| **TG dual-substrate — factor apart & close contracts (the diagnosed next move)** | **PRIORITY** | three review audits converge: *numerical robustness leads theory fidelity.* Do these **before** any new source branch or broad sweep — **(P1) reconcile existing evidence**: audit fixed-vs-recomputed `e_ref`/`q_ref` across all mass/sep rows; expose per-row source integral, mediator amplitude/gradient, probe susceptibility, raw force, `F/M_p`, breathing amp, uncertainty; classify what `F_R_well` actually is (total force / density integral / average / acceleration). **(P2) close the planned contracts**: validate dynamical T/G profiles vs the analytic screened solution; implement the **independent midplane stress-flux** force; **close the non-variational energy/work ledger**. **(P3) factor source vs probe**: asymmetric `2×2` sweep → `F ∝ M_s^α M_p^β` (does normalization give α≈0?); then compare source candidates (normalized `S_state` vs unnormalized load vs `R_relax`) at matched coupling — do **not** retune coupling to improve an exponent. **(P4) only then** targeted characterization: far-field r=6–8+ (large box, SNR stop), phase-isolated alignment (pin centres / counter-momentum so Δφ varies without relative motion), binding via bound-state energy over turning cycles, and separate lapse/yield gates |
| **TG P1a — energy observable** | **CLOSED (2026-08-25)** | `docs/gravity_maturity/TG_P1A_ENERGY_OBSERVABLE.md`. Per-half-space KG energy decomposition added to the observation module. **(a)** `F_R` couples to **4.5% of the node's half-space energy** (E_grad 4.5%, E_kin 48.6%, E_mass 54.6%, E_pot −7.7%); the other 95.5% is invisible to it. **(b)** `M_R ≡ E_mass_R` exactly (m=1), so the historical `F/M_p` was force ÷ the mass-energy term alone. `F/E_R` now recorded — the only dimensionally coherent acceleration-like ratio. **(c) THE COUPLING FRACTION VARIES 29.1% NON-MONOTONICALLY ACROSS THE MASS AXIS** (0.0354→0.0483, peak at M≈70), computed from existing charge-audit rows with no new simulation. This quantitatively confirms P1's diagnosis: a force-vs-mass curve measured this way was never measuring a clean mass dependence. **(d) CAUTION:** `F/E ~ M^-1.00, R²=0.965` must NOT be reported as a discovered scaling law — `E_tot ~ M^0.935` at R²=1.0000 and F is flat, so F/E inherits E's smoothness by construction (R²(1/E)=1.0000). | **P1 now closed except error bars on the static mass rows.** Report grad_fraction as a standing covariate on every future force row. Next: **P1-b** frozen-reference mass sweep (better motivated now); **GAP-4** remains the sharpest open problem. |
| **TG P2 — midplane stress-flux estimator** | **CONFIRMED (2026-08-25)** | `docs/gravity_maturity/TG_P2_MIDPLANE_STRESS_FLUX_RESULTS.md`, run `TG_B2_MIDPLANE_FLUX_N64`. Verdict `TG_B2_MIDPLANE_FLUX_CONFIRMS_BODY_FORCE`; all 4 preregistered gates pass. Derived the exact momentum ledger of the coded system: `dP_x/dt = (S_out - S_in) + F_R` with `S = c²A|∇φ|² - 2c²A|∂ₓφ|² - |π|² + m²ρ - U(ρ)` — so the flux is both an independent estimator (samples a 2-D plane; carries `|π|²`, `m²ρ`, `U(ρ)`, none of which appear in `F_R`) **and** a falsifiable identity. **(a)** OFF-arm null (A=1 → F_R≡0) closes the ledger on flux alone, residual second-order in sample_dt (6.05e-2 → 9.74e-3 → 2.44e-3, ratios 6.2/4.0). **(b)** Compared as a differential (F_R is identically zero at A=1, so it is differential; F_flux is a total — paired sample-by-sample against the off arm): **agreement to 0.340%**, inside the 1.42% statistical uncertainty, 71σ, per-sample correlation r=0.89, sign reverses. **`F_R` measures what it was believed to measure.** **(c)** The BARE (A-independent) two-body force is **NOT resolved** — 0.11σ; absolute F_flux is a ~1700:1 cancellation whose scatter is 205× its mean. Common-mode cancellation across arms (×1069) is what makes the differential resolvable. | **Unchanged:** every P1 downgrade stands — `F/M_p` is still not an acceleration, the mass axis is still confounded, and the force's SIGN IS STILL AN INPUT (`a_sign` flag). With the estimator now confirmed, **GAP-4 (non-variational loop) is the sharpest open problem** — the free sign can no longer be attributed to instrument uncertainty. Next: **P1-a** energy observable (the remaining P1 blocker), then re-run this estimator at N=80/L=20. |
| **TG P1 — evidence reconciliation** | **CLOSED on the audit questions (2026-08-25); BLOCKED on instrumentation** | `docs/gravity_maturity/TG_P1_EVIDENCE_RECONCILIATION.md`. Verdict `TG_P1_MASS_AXIS_CONFOUNDED__FORCE_EXPONENTS_NOT_INTERPRETABLE_AS_MASS_SCALING`. **(a)** `F_R` classified: a **total half-space body force** weighted by the **gradient energy density only** (`c²|∇φ|²`), not the full KG density; the available mass observable is `∫|φ|²dV`, so **`F/M_p` is not an acceleration** and cannot support UFF reasoning. **(b)** `e_ref`/`q_ref` are **peak normalizers recomputed per mass row**, removing ~0.36 of the source exponent by construction (`E_tot~M^0.935` → `∫S_state~M^0.585`). **(c)** The mass axis is **confounded with morphology** — `source_rms_width` varies 25% non-monotonically; `∫S_state` and `well_integral` are non-monotonic in M. **(d)** The `F~M^-0.067` near-field exponent has **R²=0.11 over 5 points** — flat scatter, not a power law; correct statement is *flat to ±10% over a 2.74× mass range*. **(e)** Whole effect is linear response: `A_well_min≈0.99993` (7e-5 deep), so the convergence antisymmetry is an arithmetic check, not a physics result. **Downgrades:** far-field `M^1.42` is a separation contrast on the same confounded axis, not a mass exponent. **Unchanged:** converged A-well attraction, exact off-null, working sign control. | next: **P1-a** (add per-half-space energy observable — unblocks the remaining P1 rows) and **P1-b** (frozen-`e_ref`/`q_ref` mass sweep — separates normalization from physics); both queued in `RUN_QUEUE.md`. Then **P2 midplane stress-flux**, now the strongest outstanding instrument check. |
| **Documentation reconciliation** | **PARTIAL (this update)** | this catalog now covers §8C; still outstanding per the audits: correct the CROSSMAP "predicted/confirmed-analogue" labels down to Weight-C where due; add the machine-readable MC-3 verdict override; scope the "Phase R complete" label (which controls passed vs respecified vs open); add the A/B/C/D attribution field to the archive; note the concept-recovery objective is *complete and shelved* (322 dossiers intentionally deferred to provenance-only) so no agent restarts it |

## 12. Guardrails maintained throughout
Mirror-first (jax_scout); the frozen Phase C dissipative operator stays byte-identical; no production solver / Hunter
/ validation / config changes; no clipping/caps added to force outcomes; feb-external coefficients labelled
exploratory substrate variants (never merged into frozen baselines); cautious language for the gravity target
("geometry-density interaction", "candidate mechanism", "not yet an emergent-gravity claim"); no "gravity proven",
"mass attracts mass", or "matter proven".

## 13. Provenance
~35 Phase-D-era result/RFC docs under `docs/` (per-experiment detail), ~50 promoted Codex campaign reports under
`docs/codex_conservative_c2_campaign_archive/`, harnesses under `jax_scout/phase_d_*.py` + `gravity_ladder_A_D.py` +
`gravity_desaturation_pilot.py`, runs under (gitignored) `sweep_runs/`. This catalog is the index; each row's doc
carries the numbers, telemetry, and caveats.
