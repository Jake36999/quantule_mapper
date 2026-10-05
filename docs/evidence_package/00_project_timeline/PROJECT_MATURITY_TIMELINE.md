# Project Maturity Timeline (may 2026 onwards)

The Quantule Mapper arc, told as evidence. This is deliberately honest about the maturity path — exploratory models,
a false baseline, three instrument bugs, and the corrections — because that path *is* the scientific record. Commit
hashes are the traceable anchors. Dates are approximate run dates from the sweep folders.

## The arc at a glance
```
exploratory monolith → Phase C stability CLOSED → dissipative transport NULL →
C2 conservative "false negative" period → C2.6 GEOMETRY BUG found → C2.7 corrected transport →
C2.9 robust two-body observable → C3 KG/Q-ball → C3 collision phase diagram →
gravity rung A+D → gravity PAUSED → catalog + theory synthesis + evidence package
```

## Stage 1 — Exploratory / substrate hunting (pre–Phase C)
- **What:** a monolithic adaptive engine hunting "interesting" substrates; hi-fi continuation of node dynamics;
  rotational-core basin studies; the prime-SSE stability objective.
- **Evidence (local):** `sweep_runs/SUBSTRATE_HUNT_20260621_161557/` (hifi_N48/96/128, chiral_viz, bound_state
  panels), `sweep_runs/CORE_SAT_PILOT_20260622_190340/`, `CORE_SAT_HUNT_20260623_004605/`.
- **Verdicts:** nodes = rotational cores, **not** topological vortices (hi-fi continuation); prime-SSE **NULL** as a
  stability predictor. *Design lesson that later drove the architecture: a monolith that evolves + scores + searches
  amplifies its own artifacts.*

## Stage 2 — Phase C: stability sector CLOSED
- **What:** established a\*≈×1.15 as a real long-time gain/loss-balanced attractor; falsified mobility; ran the
  routing/Payan/TDA/prime nulls; re-aimed the Hunter objective (H7).
- **Commit:** `cb347a9` (Phase C closure + baseline/provenance audit).
- **Evidence (local):** `sweep_runs/PHASE_C_OPTION_B_N96_*`, `PHASE_C_N96_*_20260625_*` (closure/current/longT
  panels), `PHASE_C_VISUAL_ANALYSIS_*` (case panels — real rendered PNGs), `PHASE_C_NODE_LIBRARY_20260704/`.
- **Docs:** `docs/PHASE_C_*` (referenced in the catalog Phase C rows).
- **Verdict:** stability CONFIRMED; a\* mobility FALSIFIED; ~9 structural hypotheses NULL. Sector **closed**.

## Stage 3 — Phase D dissipative: coupling without transport (D.1–D.6)
- **What:** C1 dispersive kinetic term; stress-tensor bridge; node library; coupling law; two-node dynamics; reduced
  model.
- **Commits:** `47f01ab` (C1 mirror + parity), `cd813e1` (D.5 merge-or-hold).
- **Evidence (local):** `sweep_runs/PHASE_D_C1_TRANSPORT_20260704_*`, `PHASE_D6_REDUCED_MODEL/`,
  `PHASE_C_NODE_LIBRARY_20260704/`.
- **Docs:** `docs/PHASE_D_C1_RESULTS.md`, `PHASE_D_NODE_COUPLING_RESULTS.md`, `PHASE_D_TWONODE_DYNAMICS_RESULTS.md`,
  `PHASE_D6_REDUCED_NODE_MODEL_RESULTS.md`.
- **Verdict:** nodes couple + merge + phase-lock but **never move** (C1 destabilises; D.5 merge-or-hold). Transport
  null in this sector → motivated a different substrate.

## Stage 4 — C2 conservative: the "false negative" period (LATER RETRACTED)
- **What:** opened the conservative NLS branch (C2), searched for native solitons (C2.1), chased the loss source
  (C2.2), hunted an exact soliton (C2.3), tested local boosts (C2.4), scouted families (C2.5). Concluded "conservative
  transport pinned / flow-through / drag μ≈0.04."
- **Evidence (local):** `sweep_runs/PHASE_D_C2_CONFIRM_20260704_*`, `PHASE_D_C2_2_LOSS_20260704_231230/`,
  `C22_dt*`, `C23_*`, `C24_LOCAL_N96/`, `C25_*` (pre-FIXED).
- **Docs:** `docs/PHASE_D_C2_*_RESULTS.md` (with retraction notes).
- **Status:** **RETRACTED** — these verdicts were artifacts of the C2.6 geometry-off bug (Stage 5). Preserved as
  maturity evidence, not deleted. *This is the false-baseline period.*

## Stage 5 — C2.6: the geometry-off bug found & fixed (THE PIVOT)
- **What:** a linear packet failed to translate at v=2Dk (a Galilean-identity contradiction) → traced to the
  soft-clip squash mapping Ω²=1→~151 → the "geometry-off" substrate was secretly D_eff=D/151. The universal "drag"
  μ≈0.036 = 2·D_eff exactly. Fixed with `Ops.geom_fac` / `param_geom_off`; Codex-re-audited (ETDRK4≡RK4).
- **Commit:** `0886dd0` (bug found+fixed).
- **Evidence (local):** `sweep_runs/PHASE_D_C2_6_CODEX_AUDIT_20260709/`,
  `PHASE_D_CODEX_REPRODUCTION_20260709_233433/c2_6_*`.
- **Docs:** `docs/PHASE_D_C2_6_GEOMETRY_OFF_BUG_REPORT.md`, `PHASE_D_C2_6_CODEX_AUDIT_HANDOVER.md`;
  proof ledger §2 (Report 3). **This is the single most important maturity event.**

## Stage 6 — C2.7: corrected conservative transport
- **What:** on the fixed substrate, a true soliton translates at exactly v=2Dk (0.9999, mass 0.9999, N=96); feb/a\*
  itself is structureless; moving families need s<0 + box-compatible D.
- **Commits:** `48ee430` (re-derivation harness), `a9ddd48` (complete).
- **Evidence (local):** `sweep_runs/C27_REDERIVE/` (r0_cfl.json, r2_feb.json, r3_n96.json), `C25_SCOUT_T1*_FIXED/`.
- **Docs:** `docs/PHASE_D_C2_7_REDERIVATION_RESULTS.md`. **Old-vs-corrected pivot vs Stage 4.**

## Stage 7 — C2.8/8b/9: two-body observable hardened
- **What:** Codex ran the first two-node (C2.8); the elasticity follow-up (C2.8b) gave an unphysical e=3.21 (fragile
  peak-tracker) → rebuilt as C2.9 with the momentum-density observable; static force crossover at π/2; collisions
  capture.
- **Commits:** `f19b1e0` (C2.8 + Codex tools), `fdd8686` (C2.9 observable), `9529388` (C2.9 results).
- **Evidence (local):** `sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/` (panels+npz), `C28B_ELAS/`,
  `C29_ROBUST/` (track_*.npz), `C29_VALIDATE/`.
- **Docs:** `docs/PHASE_D_C2_8_TWONODE_RESULTS.md`, `PHASE_D_C2_9_TWONODE_ROBUST_RESULTS.md`. *C2.8b = the second
  instrument bug.*

## Stage 8 — C3: KG / Q-ball development
- **What:** wave-kinetic RFC → Q-ball existence (maps the C2.7 point) → machine-clean E/Q conservation → fixed the
  boost-IC bug (missing carrier phase) → inertial transport → VK-stability (dQ/dω=−849).
- **Commits:** `9be25c8` (C3 RFC + C2.5 scout), and the C3 refinement commits.
- **Evidence (local):** `sweep_runs/C3_WAVE*`, `C3_WAVE_MAPPED/`, `C3_WAVE_BOOSTFIX2/` (qball.npy, summary.json),
  `C3_EXACT_VK2/`.
- **Docs:** `docs/PHASE_D_C3_WAVE_KINETIC_RFC.md`, `PHASE_D_C3_WAVE_KINETIC_RESULTS.md`. *C3 boost-IC = the third
  instrument bug (caught & fixed same-session).*

## Stage 9 — C3 two-body & the collision phase diagram
- **What:** two-Q-ball static force (same π/2 crossover as NLS → cross-substrate universality); collision ladder
  (in-phase capture to 0.75c; anti-phase transmits below ~0.5c; off-phase all capture) → full phase×speed diagram;
  transmission is a narrow anti-phase node feature.
- **Commits:** `48b51aa`, `b14c928` (two-Q-ball), `9a839c4` (--dphi + BOUNCE), `9579d4b` (anti-phase transmission),
  `f4b998e` (off-phase → complete diagram).
- **Evidence (local):** `sweep_runs/C3_TWOQBALL_FULL/`, `C3_COLLISION_LADDER_FULL/`, `C3_ANTIPHASE_LADDER/`,
  `C3_PHASE_pi2/`, `C3_PHASE_3pi4/`, `C3_PHASE_7pi8/` (collide_*.npz, summary.json).
- **Docs:** `docs/PHASE_D_C3_TWOQBALL_RESULTS.md`, `PHASE_D_C3_COLLISION_LADDER_RESULTS.md`.

## Stage 10 — Gravity ladder rung A+D → PAUSED
- **What:** a coherent load creates a geometry-dependent Ω²/T_info response (channel live) but the production Ω²(ρ)
  is a saturation cliff, not a graded potential; de-saturation makes it worse; paused with a documented re-entry
  condition.
- **Commits:** `5f48609` (rung A+D harness), `584ad26` (results), `489fb44` (pause decision).
- **Evidence (local):** `sweep_runs/GRAVITY_AD_bg0/` (radial profile + load_*.npy), `GRAVITY_AD_bgvac/`,
  `GRAVITY_DESAT_PILOT/`; `PHASE_D_CODEX_REPRODUCTION_.../gravity_geometry/` (geometry_law_curve.png, radial plots).
- **Docs:** `docs/IRER_GRAVITY_RUNG_A_D_RESULTS.md`, `GRAVITY_LADDER_GEOMETRY_DECISION.md`,
  `IRER_GEOMETRY_DENSITY_GRAVITY_RFC.md`.

## Stage 10B - Gravity D spatial effective-medium characterization
- **What:** the standalone mirror branch tested the bounded spatial coefficient operator
  `i d_t psi = -D div(N(x) grad psi)` as a spatial effective-medium mechanism. The sequence was: promising
  attraction, independent CPU/GPU replication, exact force-contract closure, GPU convergence, robustness battery,
  COM diagnostic closure, source-shape closure, dynamics characterization, coarse-grained model extraction, and
  Newtonian/universality rejection.
- **Evidence (local):** `sweep_runs/GRAVITY_D_GPU_CODEX_20260713/`,
  `sweep_runs/GRAVITY_D_ROBUSTNESS_GPU_20260713_195713/`,
  `sweep_runs/GRAVITY_D_ROBUSTNESS_CLOSURE_GPU_20260713_204559/`,
  `sweep_runs/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/`.
- **Docs:** `docs/GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_REPLICATION.md`,
  `docs/GRAVITY_D_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS.md`,
  `docs/GRAVITY_D_EFFECTIVE_MEDIUM_ROBUSTNESS_CLOSURE.md`,
  `docs/GRAVITY_D_EFFECTIVE_MEDIUM_DYNAMICS_CHARACTERIZATION.md`.
- **Verdict:** robust spatial effective-medium wave force characterized. The exact force is
  `F = -D integral grad(N)|grad psi|^2 dV`, with useful coarse-grained approximation
  `F_cg ~= -D K_grad grad N(R)`. Newtonian exterior field, shell-theorem behaviour, universal free fall, and a
  global point-ray description are rejected for this model. Production gravity remains closed.

## Stage 11 — Consolidation (current)
- **What:** master hypothesis catalog; theory synthesis bundle; this evidence package. Reproducibility/hygiene
  (short smoke modes; output-tree gitignore; 50 Codex MDs promoted to `docs/`).
- **Commits:** `6b64209` (hygiene/smoke), `a7e1aff` (catalog), theory-synthesis commit.
- **Docs:** `docs/IRER_MASTER_HYPOTHESIS_CATALOG.md`, `docs/theory_synthesis/`, `docs/PHASE_D_CLOSEOUT_CONSOLIDATION.md`.

## The maturity thesis (one paragraph)
Quantule Mapper did not progress by finding only positives. It progressed by **closing** a sector on nulls (Phase C),
**hitting a false baseline** (conservative pinning), **catching the instrument bug that caused it** (C2.6, against a
Galilean identity), **correcting it**, and then building the strongest results (v=2Dk transport, VK-stable Q-balls,
cross-substrate universality, a mapped collision phase diagram) on the corrected instrument — with two further bugs
(C2.8b tracker, C3 boost-IC) caught the same way. The retained nulls and retractions are the evidence that the method
is self-correcting.
