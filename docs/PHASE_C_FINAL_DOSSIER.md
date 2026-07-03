# Phase C — Final Dossier

**End state:**
`PHASE_C_NUMERICAL_ATTRACTOR_RESULT_COMPLETE` ·
`STANDING_DISSIPATIVE_ATTRACTORS_SUPPORTED` ·
`MOBILE_MATTER_LIKE_TRANSPORT_NOT_SUPPORTED_IN_CURRENT_SUBSTRATE`

**Frozen geometry** `e8d6a78ea` · N96 (N128 checked) · L=10 · dt=0.005 · classifier
`PHASE_C_SATURATION_CLASSIFIER_v3` · jax-scout FP64 (CuPy-equivalent).
**Scope discipline:** numerical-attractor result. No matter proof, particle proof, topology proof, or mobile
matter claim. "Site-pinning / accretion", not "inertia".

## Headline

> The corrected IRER-derived dissipative S-NCGL / conformal-geometry solver supports reproducible long-time
> standing bound attractors governed by gain/loss balance. These attractors are robust across seed, resolution,
> and morphology checks, but they are **site-pinned**: phase-gradient kicks and static gain wells do not produce
> coherent relocation — static wells cause local accretion or nucleation rather than transport. Therefore the
> current model supports **stable standing field structures, but not mobile matter-like excitations** under the
> tested probes.

---

## 1. Stable attractor existence — SUPPORTED
The pre-existing feb56dc7 reference is a genuine long-time bound state; its **parameter regime** is a robust
bound-state basin (node families K3–8 all stable at N96/T12000; stability set by the full param vector, not IC
norm / mass / node count). Broad Option-B IC searches produced only short-window transients (the T=6000
saturation classifier over-reported bound states); tightening the validation window to T≥24000 removed the
false positives. Docs: `PHASE_C_FEB_BASIN_RESULTS.md`, `PHASE_C_N96_OVERNIGHT_REVIEW.md`.

## 2. a\* gain/loss balance — SUPPORTED (the sharpened result)
Long-T characterisation showed even feb-center (param_a ×1.0) and ×1.05 **slowly decay** to T=72000 — the
T24000 "TRUE" verdicts were themselves window artifacts. A **gain ladder** then located the true balance: the
late-window `er` slope crosses zero at **a\* ≈ ×1.15 (param_a ≈ 0.552–0.557)**:

| param_a (×feb) | late slope /1k | reading |
|---:|---:|---|
| ×1.00 | −0.0126 | decay |
| ×1.10 | −0.0081 | slow decay |
| ×1.125 | −0.0047 | slow decay |
| **×1.15** | **−0.0004** | **stationary (a\*)** |
| ×1.16 | +0.0012 | growth |
| ×1.20 | +0.0081 | growth → `TRANSIENT_GROWER_REJECT` |

a\* is a **knife-edge zero-crossing** pinned to ~±0.5% in cubic gain: below decays, above grows. Docs:
`PHASE_C_GAIN_LADDER_RESULTS.md`.

## 3. Long-time & resolution confirmation — SUPPORTED
- **T=144000** (2× window): a×1.15 stays flat (slope −0.0004) = genuine fixed point, not a plateau artifact;
  the sub-a\* survivor a×1.125 keeps declining (er_fin 1.68→1.34) = confirmed slow decayer.
- **Seed-robust:** a×1.15 stable at seeds 619/620/621 (existence robust; node count IC-set: 4/6/6).
- **N=128 resolution:** feb-center TRUE at T12000 and T24000, same morphology, er_max matching N96 — basin is
  not an N96 grid artifact. Docs: `FEB_ASTAR_CONFIRM`, `FEB_CENTER_RESOLUTION_N128`.

## 4. Gate v3 breathing correction — SUPPORTED (calibration)
The v2 late-drift gate false-rejects the downswing of a bounded breather at long T. v3 adds a breathing
exception (accept |drift|>0.15 iff er_max≤3, floor_ratio=er_min/er0≥0.85, er_fin≤0.95·er_max), rescuing feb's
breathing at T24000 while still rejecting decay (floor fails) and growers (peak-margin fails). Window-agnostic;
preserves all T12000 verdicts. Docs: `PHASE_C_GATE_V3_BREATHING_BOUND_STATE.md`.

## 5. Joint basin & matched controls — SUPPORTED (structured boundary)
Joint (param_a, param_eta, param_rho_vac) grid: the boundary is a **gain/loss balance surface**, not
independent windows — param_a/rho_vac trade off against eta (coupling confirmed). T24000-validated core is
narrower than the T12000 basin. Matched one-param-step controls flip stable↔failed under identical gate with
n_fin=4 and prime/topo=0 on both sides ⇒ near the boundary only the **dynamics** (er breathing) discriminate,
not morphology or spectra. Docs: `PHASE_C_JOINT_PARAM_BASIN_STAGE2_RESULTS.md`.

## 6. Log-prime / TDA diagnostics — NULL (honest non-result)
Post-hoc on 60 feb-basin + 45 joint-basin states: **0 prime peaks** (log_prime_sse=999 for all), ~0 persistent
topology; flat across TRUE/SPIN/GROW/BLOW. These diagnostics do **not** discriminate stability and provide **no
support** for the log-prime or topological hypotheses. The states are smooth dissipative solitons. Prime-SSE /
TDA are exploratory post-hoc only; the v3 energy-stability gate is the promotion criterion.

## 7. Kick / inertia null — NOT SUPPORTED (with operator reason)
Galilean phase-kick on a×1.15: velocity kick-independent at the noise floor, mobility ≈ 0, structure coherent
(nodes 4→4). Operator audit: `L_k = −D·k² − η + i·ω₀`, ω₀=0 ⇒ purely real/diffusive, no advective channel ⇒
inertial motion is **structurally absent** (config-independent). Docs:
`PHASE_C_KICK_INERTIA_AND_OPERATOR_FINDING.md`, `PHASE_C_MOBILITY_ENDPOINT.md`.

## 8. Static-well morphology ladder — NOT SUPPORTED (generalized)
Default-off drag variant (baseline bit-identical), V0 ladder 0.075→0.40 across three stable a×1.15 morphologies:

| state | morphology | verdict |
|---|---|---|
| seed619 | 4-node | `ACCRETION_ONLY_NO_RELOCATION` |
| seed620 | 6-node | `STATIC_WELL_NO_COUPLING` |
| seed621 | 6-node | `ACCRETION_THEN_NUCLEATION` (new 7th node at V0=0.4) |

No coherent relocation in any morphology; the field responds to a gain preference by **local accretion /
nucleation of new structure**, never by moving the existing structure. `..._GENERALIZED_ACROSS_MORPHOLOGIES`.
Docs: `PHASE_C_ADIABATIC_DRAG_DESIGN.md`, `PHASE_C_ADIABATIC_DRAG_MORPHOLOGY_RESULTS.md`.

## 9. Final SUPPORTED claims
1. Reproducible long-time standing bound attractors exist in the corrected dissipative solver.
2. Their existence/stability is governed by a **gain/loss balance** pinned by cubic gain a\* ≈ ×1.15 (±~0.5%).
3. Robust across **seed, resolution (N128), long-time (T144k), and morphology** checks; boundary is a coupled
   gain/loss surface; the discriminator is dynamical (er breathing), not morphology/spectra.
4. The field responds to a strong local gain preference by **local accretion / new-node nucleation**.

## 10. UNSUPPORTED / non-claims
1. No inertial (Galilean) mobility — structurally absent (real diffusive operator).
2. No static-well relational relocation — generalized across morphologies.
3. No mobile matter-like excitations in the current substrate.
4. No support for **log-prime resonance** as a stability mechanism (null across all states).
5. No support for **topological (TDA/Betti)** structure as a stability mechanism (null).
6. No support for the earlier tensor/anisotropic-metric **routing** proxy (Stage B `NO_SUPPORT`).
7. Not a proof of matter, particles, topology, or analogue-gravity — a numerical attractor characterisation.

## 11. Falsifiers (what would overturn each supported claim)
- Attractor existence → a\* config decays at still-longer T (T≥288k) or fails at N=192.
- a\* balance → the zero-crossing moves >±1 grid step with seed at N128, or is not reproduced in CuPy.
- Accretion-not-relocation → a stronger/non-nucleating well, or a different stable config, shows the **existing
  node structure** (origin depletes, node identity migrates) relocating without a new blob.
- Diagnostic nulls → a prime peak or persistent topology that tracks stability appears under a corrected metric.

## 12. Optional Phase D — mobility-capable formalism (future, not patched here)
If matter-like transport is a goal, it needs a deliberate formalism extension giving the substrate a
conservative/advective channel — e.g. imaginary/dispersive kinetic term (complex-GL `(1+ib)∇²` or Schrödinger
`i·D·∇²`), explicit advective/current-coupled transport (A-field in a stable basin), a moving source/pump, or
second-order time dynamics. Each is its own phase, scoped default-off / contract-stamped / falsifier-first.
See `PHASE_C_MOBILITY_ENDPOINT.md`.

---
*Supersedes the interim `PHASE_C_DOSSIER.md` (2026-06-27), which predates the a\* refinement and the mobility
arc. Reproducibility: frozen geometry `e8d6a78ea`, v3 gate, all scripts under `jax_scout/` (`feb_gain_ladder_longt`,
`feb_astar_confirm`, `feb_kick_inertia`, `feb_adiabatic_drag`), states under `sweep_runs/`.*
