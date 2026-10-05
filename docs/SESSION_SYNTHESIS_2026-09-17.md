---
tags: [synthesis, session]
date: 2026-09-17
branch: Main branch
status: complete
---

# Session Synthesis — 2026-09-17

What was done, what it changed, what went wrong, and two structural problems the session kept
running into. Working document for review, not a results claim: every result below has its own
document, linked.

---

## 1. The short version

Tier 0 and Tier 1 of [[INTEGRATED_PLAN_2026-09]] are closed. Four results change what the plan says:

| # | result | document |
|---|---|---|
| **S3** | **The force sign is derivable, and the answer is attraction.** The coded `G` equation omits the variation of `−c²A(G)\|∇φ\|²` w.r.t. `G`. Restore it and `a_sign` enters **squared**, so it cancels. | [[gravity_maturity/S3_VARIATIONAL_SIGN_DERIVATION]] |
| **H2** | **The long-time drift is real** — converged in dt, dx *and* box size. **But ~70% of every magnitude quoted for it was box contamination.** | [[gravity_maturity/H2_BOX_LADDER_RESULTS]] |
| **S1** | The static T/G sector is **linear** and solves in closed form. The mediator is a **difference of two Yukawas**. The "screened falloff" discriminator closes **negatively**. | [[gravity_maturity/S1_STATIC_TG_GREENS_FUNCTION]] |
| **S2** | The acoustic-metric mapping is **exact**, and the Gravity-D force law is **derived, not postulated** — it is a geodesic. | [[gravity_maturity/S2_V7_ACOUSTIC_METRIC_RESULTS]] |

Plus: [[gravity_maturity/H3_MUTATION_PROBE_RESULTS|H3]] (the identity CI caught 3 of 10 bug shapes;
now 10), [[gravity_maturity/P1B_FROZEN_REFERENCE_REANALYSIS|P1-b reanalysis]], **D3 closed
negatively** at zero compute, and D1 **deliberately blocked** pending the real P1-b.

## 2. H2 — the result with a picture

![[runs/_plots/TG_H2_BOX_LADDER/h2_box_ladder.png]]

Left: the drift does not decay to zero. `a + b·exp(−cL)` fits at **0.88%** rel-rms against **22%**
for a pure power law — 25× better — with asymptote **6.45e-07**, about 30% of the L=10 value.
Right: the drift falls 3.4× while the boundary flux falls 14.4×, so it does not track the boundary.

Two of the four rows are **controls** and reproduce the frozen D4 harness **bit-for-bit, all 16
digits**, from a separately written harness.

> [!warning] This re-opens D4
> D4 failed on its `larger_box` row against a gate defined by the **L=10 reference** — a value now
> known to be ~70% contamination. The row was closer to the truth than the reference judging it.

## 3. The HUD, and what it immediately found

[[VISUAL_HUD_SCOPE_RFC|Items 6.1–6.5]] are now built, and **C1** rendered the 670-pack backlog:
445 montages, 0 failures.

![[runs/_figures/TG_SOURCE_SEMANTICS_GPU_20260714_133513/rendered__source_snapshots_density_matched_scrambled_sample020.png]]

Fourteen fields in one image with no per-campaign code — the thing the HUD existed to make possible.
Looking at these is what produced the [[runs/TG_SOURCE_SEMANTICS_GPU_20260714_133513|source-semantics
reading]] and exposed two detector defects the two-node test could not reach (a node near the
boundary was invisible; speckle fields were given six invented nodes).

---

## 4. Issues I hit

Asked directly, so answered directly.

### 4a. My own errors, all caught and recorded

| error | how it was caught |
|---|---|
| **S3 coupling ratio wrong by ~100× *and* in the wrong direction** — I said the variational term was 110× weaker; it is 1.9–2.9× **stronger** | comparing a toy Gaussian against the real `S_state`, which carries `S0=135.686` and peak normalisers; and I used the wrong propagator |
| **H2 verdict logic wrong twice** — first "shrank a lot ⟹ finite-box" (cannot tell decay-to-zero from decay-to-asymptote); then `curve_fit` landed in a local minimum with a 58% residual | fixed by making the fit deterministic — for fixed `c` the model is linear in `(a,b)`, so scan one dimension |
| **Called H2 a finite-box effect on two data points** | four points showed it asymptotes |
| **Read `corr(S_state, energy) = 0.998` as "the source is just energy"** | within one configuration that measures *shape*; the state-dependence is in the amplitude, which varies 73% across arms |
| **Labelled the position-dependent mass "the chameleon structure"** | exactly the resemblance trap the research request warned about — a chameleon varies the *mediator's* mass; here the *matter* field's varies |

The pattern is worth noting: **every one was caught by computing the thing properly rather than by
reasoning harder about the estimate.** That is the honest argument for §6.

### 4b. Friction that cost real time

- **Heredoc/escaping failures — repeatedly.** Writing Python through bash heredocs mangled `\n`,
  `\\`, `\text` (which became a literal tab) and broke string literals perhaps a dozen times. Each
  cost a repair cycle. Mitigation used: line-by-line surgery and `chr(10)`.
- **The ladder ran 12.4 hours blind.** See §5 — this is your point, and I hit it hard.
- **The test suite has 7 modules that cannot be imported** and 24 further failures, all pre-existing,
  all in the orchestration layer. A chip is queued to triage it.
- **Results stranded in the corpus.** `CORE_SAT_MASS_THRESHOLD_N96` carries `verdict: null` while its
  table contains an unambiguous negative — 3 of 3 N48 survivors fail at N96 — sitting unread since
  June.

---

## 5. Runtime observability — your first point, and I ran straight into it

**What exists now:** `jax_scout/snapshots.py` + `tools/hud_monitor.py` give a live **field** view —
slices rendered as the run writes them, reader-only, no control path.

**What does not exist: any live view of the maths.** During the 12.4-hour ladder I had exactly one
line of output per row. I could not see the drift developing, the energy ledger closing, the charge
conserving, or the fit stabilising. Had the run gone wrong at hour 3, I would have learned it at
hour 12.

That is not a missing feature so much as a missing *stream*. The harnesses **already compute** these
quantities every sample — `ledger_residual_abs`, `profile_overlap`, `boundary_flux_proxy_max`,
charge, energy — and then throw them away until the run ends.

> [!important] The cheap fix, and it is genuinely cheap
> `SnapshotWriter` already has a bounded non-blocking queue that cannot stall the simulation. Give it
> a second channel: append one JSON line per sample to `telemetry.jsonl` with whatever scalars the
> harness has in hand. Then `hud_monitor.py` tails it and plots invariants live beside the fields.
>
> Nothing new is computed. Nothing can block the run. The contract is already written and tested.
> This is a small amount of work for the difference between watching a 12-hour run and hoping.

**The stronger version** — and this is where your "active maths" framing points — is that each
harness declares its **invariants** (the identities that must hold: momentum ledger closes, charge
conserved, `A=1` when feedback is off), and the monitor shows those *live, with their residuals*.
A run that breaks an identity at hour 3 would then announce itself. The identities already exist —
`tests/test_physics_identities.py` has 18 of them — they just only run in CI, never during a run.

---

## 6. The platform / lifecycle problem — you are right, and here is the evidence

Your diagnosis matches what I kept tripping over. The evidence from this session alone:

| finding | what it indicates |
|---|---|
| **23 harnesses had each defined their own identical one-line `write_json`** | no shared seam, so provenance had nowhere to live — which is *why* 250 of 279 runs carried no commit |
| **Not one run recorded which harness produced it** | the run→code link was convention (run-id prefixes), not a record |
| **5 near-duplicate `phase_c*` renderers**, 2,093 lines | rebuild-per-campaign, retired this session |
| **`orchestrator/` + `queue_runtime.db` dead since March**, with `G:\` paths | a subsystem nobody retired, still in the tree |
| **7 test modules that cannot be imported** | a test that cannot be collected looks like coverage and provides none |
| **A June result with `verdict: null`** | no lifecycle state, so nothing flagged it as unreviewed |

**The "harness" label is doing lifecycle work it was never given the vocabulary for.** There are
~126 of them and no way to ask: *is this current? what superseded it? which enquiry does it belong
to? has anyone read its output?*

### The proposal, in the shape you suggested

Mirror the documentation pattern — **Main branch + per-enquiry branches** — for code:

```
harness manifest (a header block each harness declares, like the doc frontmatter):
    id            gravity_TG_B2_midplane_stress_flux
    branch        Branch - Gravity - Index          <- the enquiry it serves
    status        ACTIVE | SUPERSEDED | RETIRED | PROPOSED
    superseded_by <id>                              <- required when SUPERSEDED
    invariants    momentum_ledger, A_unity_when_off <- what must hold at runtime (§5)
    produces      run-id prefix(es)
```

Then a generated `HARNESS_REGISTRY.md` + index — exactly the shape of
[[DOCUMENTATION_METHODOLOGY]] and the results index — answering:

- **conception** → `PROPOSED` entries with no runs yet
- **application** → which runs each harness produced (the provenance stamp now records `harness`, so
  this closes automatically)
- **retirement** → `RETIRED` with a reason, the way the `phase_c*` renderers now carry a banner, a
  `[RETIRED]` CLI label, and a notice at point of use
- **review** → a harness whose runs have unread reading-blocks is visible as a queue

**What is already half-built**, so this is less work than it sounds: `tools/build_results_index.py`
(the manifest schema with `substrate` as the anti-pollution key), `jax_scout/provenance.py` (now
records `harness` and `argv`), `tools/build_doc_lineage.py`, and the retirement pattern demonstrated
on five modules.

**What is missing:** the manifest header itself, the registry generator, and — the part that makes it
stick — a CI check that a harness writing to `sweep_runs/` must declare one.

> [!warning] One caution from the evidence
> `orchestrator/` was an attempt at exactly this and died. The reason is recorded: it was hardcoded
> for a *finalised* system and could describe one variant of the model, so mixing eras polluted it.
> **A registry must be descriptive, not prescriptive** — it records what a harness *is*, and must
> never become something a harness has to satisfy before it can run. The moment it gates
> experimentation it will be bypassed, and then it is worse than nothing because it will be
> confidently wrong.

---

## 7. "Are we reasoning about the results enough?"

**No — and the gap is specific, not general.**

The project measures carefully and **derives rarely.** Things that had been carried as empirical
facts for months, and fell out analytically in a single session with no new compute:

| carried as | actually |
|---|---|
| "the falloff is exponential-like and short-range; the far field is not yet reached" — listed as one of **three candidate discriminators** | a **difference of two Yukawas** with both ranges fixed by the couplings (0.875, 0.484). Derivable in an afternoon. It discriminates nothing, because any theory with these couplings predicts it. |
| `kernel_screening_length = 0.9221`, treated as a constant of the model | a band-dependent single-exponential fit to a two-scale kernel; drifts 0.9275→0.9099 as the band moves. The model's actual constants were never computed. |
| "the force law" (Gravity-D), a **postulate** since introduction | a **geodesic** of an effective metric that was sitting in the wave operator. |
| `a_sign` is a free flag — **the project's #1 open problem** | a consequence of one missing variational term. |
| D4 `larger_box` **FAILED** | failed against a reference that is ~70% contamination. |

That is five months of "open problems" that were closed with `sympy` and an afternoon.

**The diagnosis is not "think harder about results."** It is that the loop
*measure → interpret → measure again* has been running without the step
**derive what the model predicts before interpreting what it did.** Where that step is missing,
measurements get interpreted against intuition, and intuition supplies the prior that the result then
appears to confirm.

> [!important] The concrete change I would make
> Before any new campaign, a **one-page "what does the model predict?"** written from the equations,
> with its parameter-free consequences stated in advance. It is cheap — S1 and S2 were each a few
> hours — and it converts campaigns from exploration into tests.
>
> And it has teeth: S1 produced a **parameter-free, falsifiable prediction** (the force-versus-
> separation curve must fit a two-scale form with both ranges fixed) that can be checked against runs
> that already exist.

**The counter-argument, which deserves stating:** the measurement discipline is genuinely strong and
is what makes this correctable at all. Three instrument bugs were caught by chasing contradictions;
the identity tests exist; the momentum ledger closes at exactly second order; the parity check
reproduces to 1.7e-12 across 38 commits. **The problem is not rigour. It is sequencing.**

---

## 8. What I would do next, in order

1. **Telemetry stream** (§5) — small, unblocks watching every future long run.
2. **P1-b proper** — decides whether the saturation cliff is physics or the source self-normalisation,
   and **D1's value depends entirely on the answer**.
3. **Re-examine D4** against an uncontaminated reference (H2 §4).
4. **Harness manifest + registry** (§6), descriptive only.
5. **The S3 decision** — yours, not mine: deriving the sign means replacing `S_state` with
   `|∇φ|²`, which is gradient energy rather than information.

---

## Associated docs

- [[INTEGRATED_PLAN_2026-09]] · [[IRER_MASTER_HYPOTHESIS_CATALOG]] · [[DOCUMENTATION_METHODOLOGY]]
- [[VISUAL_HUD_SCOPE_RFC]] · [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]]

## Branches

- [[Main branch]]
