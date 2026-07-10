# IRER Quantule Mapper — Master Hypothesis Catalog & Progress Tracker

**Purpose:** a single, centralized record of every hypothesis the project has tested — confirmed, falsified, null,
retracted, or open — so the whole arc is legible at a glance. Includes the quickly-falsified and the
instrument-bug-invalidated ones on purpose: the nulls and retractions are as much of the map as the positives.
Last updated at the C3-collision-phase-diagram milestone.

## 0. Executive assessment
Two of the three IRER simulation sectors are answered; the third is scoped and paused.
- **Stability sector (Phase C): CLOSED.** A real, long-time, gain/loss-balanced attractor (a\*≈×1.15) exists and is
  site-pinned. Every mobility, resonance, topology, and routing hypothesis around it came back null.
- **Transport/coupling sector (Phase D): ANSWERED POSITIVE across two conservative substrate families.** After a
  pivotal instrument bug was found and fixed (C2.6), true solitons (NLS) translate at exactly v=2Dk, and Q-balls
  (Klein–Gordon) transport inertially and are VK-stable. Both substrates share the same two-body relational law
  (π/2 phase-force crossover; capture-dominated collisions with a narrow anti-phase transmission channel).
- **Gravity-like sector: SCOPED + PAUSED.** The geometry-density channel is *live* (a load produces a
  geometry-dependent response) but the production Ω²(ρ) is a saturation cliff, not a graded potential; a clean rung
  requires a stable-overdense-load-on-ρ_vac-background regime that does not yet exist.
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
| I3 | CuPy (FP64) and jax_scout mirror share the identical operator | CONFIRMED | H4/A1 parity rel-L2 1.7e-12; C1 parity byte-exact |
| I4 | Production stability_metrics reach the provenance/validation path | CONFIRMED | A3/A4/A4b wiring accepted |

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
**Sector verdict:** the conservative NLS substrate **supports clean coherent transport** (v=2Dk) for solitons that
satisfy s<0 + box-fit; two-body dynamics = phase-force (π/2) + capture. feb/a\* itself is structureless there.

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
anti-phase node channel. Same π/2 two-body force as NLS (universality); cleaner conservation.

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
**Lesson (kept as a standing rule):** every transport/interaction claim is gated on a correct boost IC, a robust
(non-peak-tracking) observable, and conservation telemetry; a null that contradicts a symmetry identity is treated as
an instrument fault until proven physical.

## 11. Current frontier & open threads (none blocking)
| thread | status | next |
|---|---|---|
| C3 asymmetric-velocity elasticity | OPEN | quantify captured-vs-radiated energy budget with the momentum observable |
| C3 captured-remnant long-time fate | OPEN | stable "Q-ball molecule" vs slow decay (machine-clean conservation makes it well-posed) |
| C2 higher-speed / off-phase (NLS analog of C3) | OPEN | does the NLS pair show the same anti-phase node channel? |
| C2′ canonical geometry | DESIGN-ONLY | RFC; theory choices (§7) pending; only if a conservation-exact geometry is needed |
| Gravity ladder rung B+ | PAUSED | re-entry = stable-overdense-load-on-ρ_vac-background pilot |
| Codex replication | AVAILABLE | finalized C2.9/C3/collision harnesses ready for independent re-run; C2.6/C3 already re-audited |

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
