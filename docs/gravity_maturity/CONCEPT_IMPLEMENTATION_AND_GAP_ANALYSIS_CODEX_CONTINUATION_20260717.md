# Concept Implementation and Gap Analysis - Codex Continuation

Codex continuation, 2026-07-17. Jake McIntosh retains conceptual authority over IRER interpretation. Claude's extraction and
cross-analysis work is preserved unchanged; this document is an additive implementation-facing continuation.

## Status and Boundary

This document extends `CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS.md` by cross-reading the recovered v9/extraction concepts
against the recent Gravity/TG implementation record from an implementation and review-agent perspective.

Key clarification:

> The current system implements many foundational concepts, but often as proxies or branch-specific analogues; this does
> not mean the original ontology is proven.

No production verdict, master-catalogue verdict, protected solver, Hunter path, GPU launcher, or queue row is changed here.
This is documentation and cross-analysis only.

## Source Basis

Primary extraction guides:

- `docs/gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS.md`
- `docs/theory_synthesis/irer_archive/CROSSMAP_CONCEPTS_VS_RESULTS.md`
- `docs/theory_synthesis/irer_archive/00_CORPUS_COVERAGE_REGISTER.md`
- `docs/theory_synthesis/expanded_declaration/PRELIMINARY_CONCEPT_SYNOPSIS_NOT_ARCHIVALLY_COMPLETE.md`
- `docs/theory_synthesis/expanded_declaration/09_MISSING_CONCEPTS_AND_ADDENDA.md`

Current implementation/evidence cross-checks:

- `TG_SEMANTIC_BASELINE_AND_POSTULATE_AUDIT.md`
- `TG_CONSOLIDATION_AND_VALIDATION_FRAMEWORK.md`
- `TG_B1S_D4_FINAL_ANALYSIS.md`
- `TG_B1S_FREQUENCY_CONTRACT_RESULTS.md`
- `TG_B1S_BOX_DEPENDENCE_RESULTS.md`
- `TG_B2_TWO_NODE_FORCE_RESULTS.md`
- `TG_B2_METHOD_ASSESSMENT_AND_DYNAMICAL_PLAN.md`
- `TG_B3_FOUNDATIONS_DESIGN_CONTRACT.md`
- TG-R robustness v2 returned capsule and first-pass review

The original v9 document remains the read-only conceptual source. This continuation relies on the extraction files for
source discovery and uses v9 only as targeted confirmation, not as a fresh archive pass.

## Evidence Classes That Must Stay Separate

Several recent documents use nearby language for different evidence classes. Future agents should not collapse them.

| Evidence class | What it measures | Current reading | What it does not establish |
| --- | --- | --- | --- |
| Spatial medium force | Exact coefficient-gradient wave force, `F = -D int grad(A) |grad psi|^2 dV` | Reproduced and characterized as finite-width, gradient-energy weighted, source/profile dependent | Not a universal metric force or Newtonian exterior field |
| Live-field TG-B2 body force | `F_R = -c^2 int_{x>0} d_x A |grad phi|^2 dV` on live two-node fields | A-well branch passes sign, null, sign-flip, epsilon, grid, dt, and larger-box sentinels | Not a settled two-node orbit, capture, or secular impulse result |
| Secular impulse / trajectory | Long-time `J = P_R(full) - P_R(off)` or separation dynamics | Methodologically open; earlier proxies exposed breathing and bookkeeping contamination | Not superseded by the body-force pass; it is a different observable |
| Temporal clock response | Supplied or sourced `N_t` effect on clock/KG frequency | Temporal differential detected, but G1 local clock calibration failed | Not an objective clock-law calibration yet |
| Source semantics | What a proposed source actually measures | `S_state` is node-state load; `R_relax` is phase-tension relaxation; other sources remain gated or disabled | A source cannot be called "resolution activity" unless it passes its own semantics gate |

This distinction matters because TG-R robustness strengthens the TG-B2 body-force observable, while the secular
two-node dynamical sign remains a separate method problem.

## Concept / Implementation / Failure Matrix

| concept / formalism | v9 or extraction anchor | current implementation | current evidence | missed or underweighted implication | failure mode predicted | next test / queue candidate |
| --- | --- | --- | --- | --- | --- | --- |
| Gradient-derived informational force | Concept 20; MC-1 gradient-derived, finite-range, non-universal force | Gravity-D and TG-B2 body-force law | Exact operator force closed; TG-B2 A-well body force robust in Phase R sentinels | Non-Newtonian behaviour may be a theory-native signature, not just a rejection | Force depends on probe/node structure and local gradient energy | Continue using force-density/body-force observables; avoid COM-only promotion |
| Field-rigidity / load-capacity saturation | MC-1 and missing-concept addenda around capping, yield, path saturation | Production gravity cliff and finite source response are observed, but not reframed as yield physics | Gravity-D characterization rejected Newtonian benchmarks; saturation cliff remains a blocker | Saturation may be a critical-RD/load-capacity observable that was forgotten | Attempts to "desaturate" can steepen the cliff instead of restoring a smooth potential | Characterize onset scale and yield behaviour rather than only trying to remove it |
| Payan/FMIA alignment controls coupling | MC-3, coupled rotational information indifferences, spin-locked conduits | Phase-dependent C2/C3 two-body laws; TG-B2 currently uses fixed phase choices | pi/2 force crossover and anti-phase pass-through are known | Relative phase may be the implemented Payan-alignment variable | Apparent non-universality or sign flips may track phase/alignment, not numerical error | Add phase/alignment diagnostics to two-node body-force and collision harnesses |
| Dynamic FMIA channels | MC-5; FMIA as co-evolved low-resistance path, not static route | Static routing predictors nulled; dynamic TG stress exists but path observables are not defined | Static routing nulls remain valid only for static/proxy sector | Dynamic channels were likely tested in the wrong substrate before | No channel appears in pinned dissipative/static sector even if dynamic transport channels exist | Define corridor/path persistence metrics under KG/NLS transport or TG stress |
| Temporal-geometric impedance / delay | TG loop formalization; T/G response fields with finite omega/gamma | B1S and B2 use damped second-order T/G fields | B2 method assessment shows T/G response time comparable to node breathing | Delay is not merely runtime inconvenience; it is a physical impedance analogue | Quasi-static sign can wash out or oscillate when T/G is non-adiabatic | Measure response phase lag and compare to node breathing before interpreting secular force |
| Relief/radiation channel | Photon/propagator/open chi_out hypothesis; informational propagators | No named relief channel; only boundary absorber and incidental radiation metrics | D4 larger-box failure and absorber suspicion remain unresolved | Missing relief makes boundary/absorber dependence expected | Long-time drift or frequency shifts may depend on box because the boundary is the only sink | B3 chi_out design; shell-flux diagnostics inside absorber; energy split tests |
| Node-state load source | TG-S state-load semantics; density/load source warning in loop doc | `S_state` with frozen global normalization | Feed-forward chain supported; backreaction robust but weak; D4 unresolved | Good load source, but not completed-resolution rate | Static load can create quasi-static fields and drift without event semantics | Keep as baseline branch; do not rename it to resolution activity |
| Phase-tension relaxation source | TG-S `R_relax`; continuous coherence-transition candidate | Classified but disabled in state-load branch | Semantics evidence exists but no feedback rerun with this source | This may be the closest current proxy to rate-of-resolution | Derivative noise or breathing contamination if not re-gated in new branch | B3-S re-gate with derivative/cadence audit before feedback use |
| Temporal lapse on substrate | c_emergent/load-dependent chronology; G1 and TS bridge | Not wired into the working TG loop; T reaches phi through G/A only | G1 clock calibration failed; temporal differential detected | Current TG frequency shifts are geometric, not direct chronology shifts | Clock modes can migrate toward source and confuse rate measurement | Fix local clock instrument before promoting any direct `N_t` branch |
| Clock migration toward source | G1 failure boundary; load-dependent clock response concept | Observed as calibration failure | Far eigenmode drifted toward low-lapse region | Could be a physical load-seeking/medium response signal, not only an instrument bug | "Clock" fails by becoming a probe of the field gradient | Design a deliberate migration test rather than treating it only as invalidation |
| A-hill / A-well polarity | IRER throttling sign: dense load -> slower chronology -> lower coefficient | B1S frozen A-hill scaffold; B2 A-well theory branch | FC-1 explains A-hill faster frequency; B2 body-force A-well attracts | Boundedness and attraction currently live on different branches | Agents may accidentally treat A-hill validation as theory-polarity validation | Keep branch names explicit; harden A-well boundedness separately |
| Placed nodes vs formed nodes | OIW overlap -> RD/PAS -> Quantule formation | Petviashvili Q-balls are inserted | Stationary node machinery is strong; formation remains absent | Current work tests response to load, not emergence of load | Raw superposition of placed nodes can be violently non-stationary | Cooled/relaxed pair or formation scout before claiming bound structure |
| Non-variational work channels | FMIA/least-action and loss ledger requirements | Phenomenological phi/T/G chain; T/G exchange partly action-like | Ledger residuals small but not exact | The model can be useful but claim strength is capped | Weak effects may include unbooked work terms | LEDGER-1 and VAR-1 design remain important before mature claims |
| Boundary/absorber dependence | Boundary-induced forces, relief, propagators, holographic/boundary signatures | Absorber is numerical sponge, not physical channel | D4 box row failed; FC-box ruled out simple resolution/local-mechanism explanation | Boundary sensitivity may be a clue to missing channel, not only a bug | Larger boxes or absorber shifts change long-time modal shifts | Isolate absorber position, shell flux, and reflected/absorbed power |
| Informational inertia / DII | MC-2; coherence stability as resistance to reconfiguration | KG second-order substrate provides inertia; dissipative attractor pins | Stable dissipative structures resist motion; KG transports | Stability/transport tension was predicted by DII-like reasoning | Too-stable attractor does not move; too-mobile state may not persist | Treat "where inertia lives" as a design variable in future models |

## Implementation-Led Additions The Extraction Pass Underweighted

### 1. Body force vs secular motion is now a standing methods distinction

The TG-B2 body-force observable is much cleaner than COM or half-space momentum proxies. It is sign-controlled, has an
exact off-null, and survived the compact TG-R robustness package. However, it is still a live-field force-density
measurement. It does not by itself prove that two nodes enter a quiet secular inward trajectory.

Predicted risk already visible in the formalism: weak mediated forces can be hidden under large substrate breathing,
especially when the pair state is not a stationary object. The next secular-motion test must begin from a quieter pair
or explicitly average over and model the breathing.

### 2. The "violence" of the two-node evolution is a substrate-state problem

The feedback-off arm breathes violently even without geometry acting back on phi. That means the violent evolution is not
caused by T/G feedback fighting itself. It is the bare two-node KG initial condition: two placed Q-balls are not a
stationary two-body solution at the tested overlap.

Recovered-concept reading: the system has not yet implemented "node formation" or "stable route selection"; it has placed
nodes and then watched an excited pair relax. The formalism predicts that such a configuration may seek a new attractor
or radiate load. Treating that violence as an implementation bug is too narrow.

### 3. T/G response delay is temporal-geometric impedance

The T/G fields have finite response times. The non-adiabatic mismatch between T/G ring-up and node breathing is not just
a numerical nuisance. It is the first concrete version of the "temporal-geometric impedance" idea: the geometry does not
instantaneously reflect node activity.

Predicted outcome: if response delay is load-bearing, varying `omega_T`, `omega_G`, damping, or source type should change
the phase lag and possibly convert an oscillatory loop force into a secular one, or vice versa. This should be measured as
phase lag, not tuned away.

### 4. Missing relief channel is a plausible explanation for box dependence

The original loop expects outgoing perturbation or relief. The current B1S branch has an absorber but no named `chi_out`
observable or physical relief channel. D4's larger-box discrepancy and absorber suspicion should therefore be read as a
theory-relevant warning: the boundary may be impersonating the missing relief channel.

Predicted outcome: a branch with measured shell flux or explicit relief accounting should reduce unexplained box/absorber
sensitivity, or at least move it into a measurable exported-energy ledger.

### 5. Static routing nulls do not close dynamic FMIA

Earlier FMIA routing tests failed in static/proxy settings. The extracted concepts describe FMIA as dynamically carved,
reinforced, saturated, or fractured by repeated activity. A static null therefore should remain local to that static
sector.

Predicted outcome: if dynamic FMIA exists in this implementation family, it should appear as repeated flux concentration,
phase-coherent corridors, reduced action paths, or persistent channel memory under transport or TG stress, not as a
pre-existing geometric bridge.

### 6. Clock migration may be a signal class

G1 failed because the clock mode did not stay where the local-clock instrument needed it. But a mode migrating toward a
low-lapse/high-load source may be a real field response. This should stay a calibration failure for clock-law claims, but
it also deserves a separate "migration as response" test.

Predicted outcome: if clock migration is physical, its direction and rate should scale with supplied temporal gradient,
trap strength, and clock mass/width, and should vanish in flat controls.

## Failure-Point Register

| observed or likely failure | formalism-linked cause | how to diagnose | result boundary |
| --- | --- | --- | --- |
| D4 larger-box frequency-scale failure | Missing relief channel, absorber-position coupling, or long-time dynamical sensitivity | Compare absorber/shell flux, row-local T/G tails, and phase slope windows | Keeps `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`; blocks D5-style promotion |
| Secular two-node sign ambiguous | Bare pair breathing plus non-adiabatic T/G response | Feedback-off breathing metrics, phase-lag spectra, long settled averages | Body-force attraction remains distinct from secular-trajectory claim |
| COM/masked-position sign errors | Internal reshaping and mass sloshing contaminate position proxies | Compare against force-density and momentum conservation | COM-only force labels should remain retracted |
| Source over-interpretation | `S_state` is load, not resolution rate | TG-S-style source semantics gate | Do not call load source completed-resolution activity |
| Temporal claim overreach | No direct `N_t(T)` in working loop | Inspect phi equations and clock calibration results | Frequency shift is geometric unless a direct temporal limb passes |
| A-hill/A-well branch confusion | Numerically safe scaffold sign differed from throttling sign | Explicit branch labels and sign controls | A-hill boundedness cannot be silently inherited by A-well branch |
| Static FMIA null overreach | Static tests do not represent co-evolved channels | Dynamic transport/channel metrics | Keep static nulls local to tested sector |
| Energy-ledger ambiguity | Phenomenological/non-variational chain | Book missing work terms or derive action | Limits claim strength for weak effects |

## Candidate Continuation Tests

These are documentation-level candidates only. They should become queue rows only if Jake selects them.

| candidate | smallest question | expected evidence |
| --- | --- | --- |
| Absorber/relief isolation | Is B1S D4 box sensitivity boundary-coupled? | Shell flux, T/G tail energy, absorber-distance sensitivity |
| Cooled two-node body-force plus impulse run | Does a quieter two-node state produce a secular sign matching body force? | Reduced feedback-off breathing, settled `J` average, body-force cross-check |
| Rate-source semantics bridge | Does `R_relax` behave as a resolution-rate source when embedded in TG? | TG-S derivative/cadence pass plus source-off/phase-scramble controls |
| Clock migration characterization | Is G1 migration a reproducible field response? | Trap/mass/width scaling and flat null |
| Dynamic FMIA corridor scout | Do repeated TG-stressed emissions carve persistent low-resistance channels? | Flux concentration, path persistence, rotation/translation controls |
| Load-capacity/yield map | Is the gravity saturation cliff a critical-RD yield? | Onset threshold, hysteresis, recovery/failure curve |

## Handoff Notes For Future Agents

1. Treat Claude's extraction as the canonical source-discovery layer. Codex continuation adds implementation-facing
   interpretation, not replacement.
2. Preserve the distinction between:
   - A-hill frozen B1S scaffold;
   - A-well theory-faithful B2 branch;
   - future B3 dual-coefficient/rate-source design.
3. Do not promote from body-force robustness to secular orbital binding without the corresponding impulse/trajectory
   gate.
4. Do not promote a source to "resolution activity" without a source-semantics pass.
5. Do not use direct temporal-language claims for a geometry-only frequency shift.
6. Treat predicted non-Newtonian, finite-range, saturating, and structure-dependent responses as possible IRER-native
   signatures, while still keeping ordinary physics claims bounded.
7. Keep master-catalogue updates separate from this working analysis.

## Bounded Conclusion

The extraction work shows that many behaviours now seen in the implementation record were already present as conceptual
predictions or open mechanisms in the original theory family: saturation, finite-range gradient forces, phase/alignment
dependent coupling, dynamic path formation, relief/radiation, and chronology/geometry feedback. The implementation record
has not turned those into ontology claims. It has produced a network of tested analogues, proxies, nulls, and failure modes.

The main value of this continuation is to stop future agents from interpreting every non-Newtonian or non-adiabatic result
as an accidental failure. Some of those behaviours are exactly what the recovered formalisms say should be examined next.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 4 commit(s), most recently `60093e5` (2026-08-27)

**Harness code changed since it was written:** 4 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
