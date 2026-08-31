# Phase D / C2.8 — Two-Node Interaction: Assessment

**First relational-dynamics experiment on the fixed conservative substrate (GALILEAN family a=0.8, s=−0.5, f=−0.1,
D=0.3, μ=0.070; L=20, N=96). Run + handover by Codex (`sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611`);
this is Claude's scientific assessment.** No broad stability or matter claim. Two solitons interact strongly and
mass-conservingly: a **phase-dependent static force** (in-phase attract, anti-phase repel) and a
**velocity-dependent collision outcome** (slow → capture into a bound pair; fast → survive with internal
excitation). Cautious wording throughout: head-on cases labeled `TWO_CORES_SURVIVE_CLOSE_ENCOUNTER`.

## Directly supported (robust)
| case | setup | result | mass ret |
|---|---|---|---|
| T0 hold | isolated soliton, L=20 | holds (residual 1.8e-8, pedestal 9.6%) | 0.9999 |
| T1 head-on n=2 | v=0.377 each, closing 0.75 | two cores survive; end **locked at sep 1.458, amp oscillating 0.77–0.81** | 0.9991 |
| T1 head-on n=4 | v=0.754 each, closing 1.51 | two cores survive; **re-separate at reduced amp 0.4–0.5** | 0.9989 |
| T2 static in-phase | sep 6, Δφ=0 | **ATTRACT** → sep 5.83→1.46 | 0.9995 |
| T2 static anti-phase | sep 6, Δφ=π | **REPEL** → sep 5.83→8.75 | 0.9994 |
| supp static Δφ=π/2 | sep 6 | ATTRACT → 5.63→1.46 (one sample) | 0.9995 |
Every case conserves mass to ≥0.999 — **no radiative loss**; interactions redistribute energy internally, not away.

## The phase-dependent static force is the cleanest new physics
In-phase pairs attract, anti-phase repel — the **canonical NLS soliton interaction law**. Its clean appearance is
strong independent evidence these are genuine, well-behaved solitons (not numerical blobs), and it is the
conservative analog of the dissipative D.5 "merge-or-hold." The π/2 sample also attracting suggests the crossover to
repulsion sits near Δφ=π rather than at π/2 — worth a 3–4 point Δφ sweep to map the force-vs-phase curve (cheap
static runs, the single most informative cheap follow-up).

## The collision physics: capture vs inelastic survival (suggested, not yet clean)
- **Slow (n=2):** cores meet and **stay locked at sep≈1.46 with oscillating amplitude → capture into a bound
  oscillating pair** (a "breather molecule"), post-min separation slope ≈ 0.
- **Fast (n=4):** cores **re-separate** (slope +0.10) but emerge at ~half amplitude → **inelastic, with kinetic
  energy converted to internal oscillation**. (Peak tracking is noisy post-collision — one frame reads sep=10.0, a
  wrap/detection glitch — so the outgoing speed is not cleanly measured.)
- This capture-slow / inelastic-fast pattern is the hallmark of **non-integrable** soliton interactions (contrast
  1D cubic NLS, which is integrable and scatters elastically). It is fully consistent with the attractive static
  force. **Physically expected for a 3D cubic-quintic-septic substrate.**

## The identity ambiguity — reframed
Codex correctly kept head-on neutral because `rho_x` cannot label which incoming soliton became which outgoing one.
**Sharper point: for the *symmetric identical* solitons run here, "pass-through vs rebound" is not resolution-limited
— it is undefinable by symmetry.** Reflection symmetry about the collision midplane makes the two hypotheses produce
*bitwise-identical* density evolution (even full 3D density). Adding a phase/frequency tag would let you *define* a
label, but it answers a convention, not physics. **The physically meaningful, measurable question is elasticity:** do
two clean solitons emerge at the incoming speed (elastic), at reduced speed (inelastic), or not at all (capture)?
The current symmetric data already answers this qualitatively (capture at n=2; inelastic at n=4) but not
quantitatively (noisy outgoing-velocity tracking).

## Recommended next diagnostic (smallest that resolves elasticity)
**Asymmetric-velocity head-on** (v_L ≠ v_R, e.g. n_L=3 / n_R=2): breaks the mirror symmetry, so the two emergent
cores have *distinct speeds* and are trackable as continuous, unambiguous trajectories; net P≠0 gives a conserved-P
check. Measure incoming and outgoing per-core velocity by linear fit to the well-separated segments → elasticity
e = |v_out,rel| / |v_in,rel| directly. Pair it with a robust tracker (parabolic sub-cell peak interpolation +
nearest-neighbor identity association across frames, longer post-collision T). One bounded run. This is `C2.8b`.
(Add the Δφ ∈ {π/4, π/2, 3π/4} static sweep alongside — near-free.)

## C2.8b elasticity attempt — INCONCLUSIVE (instrument too fragile; do not trust its auto-verdict)
Ran `jax_scout/phase_d_c2_8b_elasticity.py` (asymmetric collisions n_L≠n_R + Δφ∈{π/4,π/2,3π/4} static sweep),
`sweep_runs/C28B_ELAS`. **Mass (0.9989) and momentum (conserved to 0.3%, P 1622→1617) are excellent** — the dynamics
are sound. But the reported `C2_8B_ELASTIC` verdict is NOT reliable:
- **asym_3_2** is plausible: clean re-separation (sep_end 8.96), relative speed retained e=|v_out,rel|/|v_in,rel|=0.91
  → roughly elastic / mildly inelastic. One believable point.
- **asym_4_3** is a tracking artifact: e=3.21 is unphysical (outgoing rel-speed 3× incoming violates energy
  conservation); sep_end=2.49 means the cores never cleanly separated, so the outgoing-velocity fit ran on
  peak-finder noise. The "ELASTIC" label came from that garbage clearing the 0.85 threshold.
- **Static Δφ sweep contradicts theory AND the earlier Codex supplement:** this run reports π/4, π/2, 3π/4 all REPEL,
  but the cos(Δφ) NLS interaction law predicts π/4 should ATTRACT and Codex's π/2 supplement measured ATTRACT. The
  likely failure: `two_peaks_tracked` picks up a spurious far-field second peak after an attracting pair merges →
  false "repel." Crossover location unresolved.
**Root cause: extracting outgoing velocities / separation from projected-density peak-finding is too fragile**
(breaks when cores are close, breathing, or merged). A robust redo needs a different observable — **local half-space
momentum** P_x integrated over x<0 and x>0 (gives each core's momentum directly, no peak identification), and
saved full separation trajectories for the static sweep. Elasticity remains OPEN. The durable C2.8 findings above
(mass/P-conserving strong interactions; in-phase attract / anti-phase repel) are unaffected.

## C2.6 fix — independently confirmed by Codex (external replication)
The Codex audit (`tools/c2_6_independent_audit.py`, run `PHASE_D_C2_6_CODEX_AUDIT_20260709`) reproduces our fix on
independent tooling: default parity max|Δ|=0.0 (L_k, E, f1); `param_geom_off=True` genuinely flat (geom_fac=0.0,
N_op = polynomial-only to 0.0); **old bug reproduced (D_eff ratio 0.006511 ≈ 1/151 = 0.006623)**; linear packet +
true soliton transport correct **and agreeing between ETDRK4 and an independent RK4 integrator** (mass 0.99992 vs
1.0000000 — different steppers, same trajectory); geometry-off flux ~1e-16; protected production/reference diff
empty. This is the independent confirmation requested in the C2.6 handover — the corrected substrate is trustworthy.

## Housekeeping
- Committed: harness `jax_scout/phase_d_c2_8_two_node.py`, Codex tools (`tools/analyze_c2_8_two_node_outputs.py`,
  `run_c2_8_bounded_supplements.py`, `c2_6_independent_audit.py`), `docs/PHASE_D_C2_8_AND_C2_6_CODEX_HANDOVER.md`.
- Run artifacts stay in `sweep_runs/` (gitignored); the run folder holds the preserved harness snapshot + manifest.
- **Flag:** `quantule_viz/outputs/` has a large untracked pile of Codex render/diagnostic artifacts (gifs, pngs,
  csvs). Recommend adding `quantule_viz/outputs/` to `.gitignore` (regenerable data products) — left for user
  decision, not unilaterally changed.
- Guardrails: mirror-only, true-flat substrate, no production/Phase C changes, no matter claims.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `7f0a976` (2026-07-09) — *Phase D C2.8 assessment report + C2.8b elasticity diagnostic*
**Revised since:** 8 commit(s), most recently `3877db6` (2026-08-31)

**Harness code changed since it was written:** 21 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate
  - *…and 16 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
