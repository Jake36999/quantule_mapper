# Validation Metric Candidates

Status: provisional Codex metric-preparation notes for Claude review. These are candidate comparison formulas and measurement hooks only. They do not reinterpret IRER, change any verdict, or claim external physical correspondence.

Match levels: `EXACT EQUATION`, `SAME FAMILY`, `CLOSE ANALOGUE`, `SPECULATIVE`, `UNVERIFIED`.

## V1 - Karpman-Solov'ev / Gordon Phase-Force Fit

External source IDs: `SRC-KS-001`, `SRC-KS-002`, `SRC-KS-003`, `SRC-KS-004`.

Candidate target form:

```text
q_ddot = -C * exp(-lambda * q) * cos(Delta_phi)
Psi_ddot = -C * exp(-lambda * q) * sin(Delta_phi)
```

where `q` is a half-separation or separation coordinate depending on convention, `Delta_phi` is relative phase, and `C`, `lambda` depend on NLS normalization and soliton amplitude.

Candidate metrics:

| Metric | Extraction from IRER data | External formula target | Units/scaling caveat | Match level |
|---|---|---|---|---|
| Phase sign accuracy | Sign of early separation acceleration from C2.9/C3 static pair tracks | `sign(q_ddot) = -sign(cos(Delta_phi))` | Relative phase convention must be aligned; C2.9 pi/2 row needs careful neutral/near-neutral treatment. | SAME FAMILY |
| Crossover phase error | Fit phase where `q_ddot = 0` | `Delta_phi = pi/2` | Nonintegrable CQ/CQS solitons may shift constants or crossover slightly. | SAME FAMILY |
| Exponential decay rate | Fit `abs(q_ddot)` vs separation | `exp(-lambda q)` | Need consistent `q` vs full separation; amplitude controls canonical `lambda`. | SAME FAMILY |
| Prefactor comparison | Fitted `C` vs amplitude-derived reference | Canonical pure-NLS constants such as `C ~ 4 A^3` only under canonical normalization | Project C2/C3 are not pure 1D cubic NLS; constants are not expected to be exact. | CLOSE ANALOGUE |
| Experimental optical-force cross-check | Compare sign and trend of measured phase-force/separation behavior to optical-fiber force experiments | Mitschke/Mollenauer report attractive and repulsive soliton interactions in optical fibers | Experimental fiber units and pulse-separation observables must be mapped; formula details remain `TO_VERIFY_METADATA` before quantitative use. | SAME FAMILY |

Acceptance language if used later: "phase-force data are consistent with the standard soliton-interaction sign law and exponential-overlap family." Do not say it proves an external physical correspondence.

## V2 - C2 Galilean Transport Line

External source IDs: `SRC-NLS-001`, `SRC-NUM-001`.

Candidate target:

```text
omega(k) = D * k^2
v_group = d omega / d k = 2Dk
```

For the Galilean-invariant NLS family, adding a carrier phase with wavenumber `k` moves the soliton with velocity `v = 2Dk` under the project convention `i psi_t = -D Laplacian psi + F(|psi|^2) psi`.

Candidate metrics:

| Metric | Extraction from IRER data | External formula target | Units/scaling caveat | Match level |
|---|---|---|---|---|
| Velocity slope | Fit measured centroid/current velocity vs `k` | slope `2D` | Use project `D`; avoid old C2.6 geometry-off bug path. | EXACT EQUATION |
| Intercept | Fit `v = m k + b` | `b = 0` | Periodic wrapping and centroid convention can bias intercept. | EXACT EQUATION |
| Mass retention by k | `mass_final / mass_initial` | high retention expected for faithful transport measurement | Not an analytic law; a numerical fidelity metric. | SAME FAMILY |
| R-squared | line fit quality | linear relation | Needs enough `k` samples. | EXACT EQUATION |
| External reduced-NLS numerical sanity check | Reproduce a deliberately reduced flat-NLS boost/propagation identity in an independent NLS solver/package | Same reduced NLS equation family | This is numerical-method sanity only, not empirical validation and not a full IRER comparison. | EXACT EQUATION |

## V3 - Cubic-Quintic / Cubic-Quintic-Septic Collision and Capture Literature

External source IDs: `SRC-CQ-001`, `SRC-CQ-002`, `SRC-DATA-003`, `SRC-DATA-004`, `SRC-DATA-009`, `SRC-DATA-010`.

Candidate comparison form:

```text
i psi_t = -D Laplacian psi + (a*rho + s*rho^2 + f*rho^3) psi
```

No single universal closed-form capture law was verified in this pass. The comparison should therefore use collision observables rather than a claimed exact formula.

Candidate metrics:

| Metric | Extraction from IRER data | External target | Units/scaling caveat | Match level |
|---|---|---|---|---|
| Capture/transmission outcome grid | Classify outcome by `(Delta_phi, v)` | Published CQ/CQS and BEC/optical collision studies report phase/speed-dependent outcomes | Need same phase convention and speed normalization; optical/BEC media differ. | CLOSE ANALOGUE |
| Mass/norm retention | Track conserved mass or charge | Numerical fidelity/comparison anchor | Conservation law differs by equation family. | SAME FAMILY |
| Momentum/current telemetry | Compare incoming/outgoing current velocity | NLS/GPE current observables | Windowed current definitions differ across dimensionality. | CLOSE ANALOGUE |
| Radiation fraction | Estimate residual outside core windows | Common collision diagnostic in nonintegrable soliton studies | Requires a stable core/radiation partition. | CLOSE ANALOGUE |
| BEC phase-dependent collision analogue | Compare qualitative phase-dependent collision behavior against bright matter-wave soliton experiments | Nguyen/Hulet source trail reports relative-phase-sensitive collision dynamics | BEC experiment is not the project equation; raw/tabulated data were not verified. | CLOSE ANALOGUE |

## V4 - Complex Ginzburg-Landau Dissipative Solitons

External source IDs: `SRC-CGLE-001`, `SRC-CGLE-002`, `SRC-CQ-002`, `SRC-DATA-007`.

Generic CGLE family:

```text
partial_z A = delta A + (beta + iD) partial_xx A
              + (epsilon + i gamma) |A|^2 A
              + (mu + i nu) |A|^4 A
              + higher-order terms
```

Project relevance is family-level: gain/loss-balanced localized states, not an exact equation identity.

Candidate metrics:

| Metric | Extraction from IRER data | External target | Units/scaling caveat | Match level |
|---|---|---|---|---|
| Existence window | Phase C stable/unstable parameter runs | CGLE dissipative soliton existence regions | Parameter names and signs differ; compare qualitatively unless mapped. | SAME FAMILY |
| Attractor/pinning behavior | Standing attractor persistence and drift | Dissipative soliton attractor behavior | Optical propagation coordinate differs from simulation time. | CLOSE ANALOGUE |
| Gain/loss balance | Net mass/energy trend over late window | Dissipative balance | Needs exact project budget terms. | SAME FAMILY |
| Pulse/core morphology | Width, amplitude, occupancy | localized dissipative pulse classes | Dimensionality mismatch likely. | CLOSE ANALOGUE |

## V5 - Klein-Gordon / Q-Ball / VK Branch

External source IDs: `SRC-QB-001`, `SRC-QB-002`, `SRC-QB-003`, `SRC-QB-004`, `SRC-QB-005`, `SRC-SALIH-2026`.

Standard Q-ball ansatz:

```text
Phi(x,t) = phi(r) * exp(i omega t)
Q = integral Im(Phi* partial_t Phi) d^3x
```

Common stability/comparison hooks:

```text
dQ/domega < 0       # VK-style sign under common convention
E/Q < m             # absolute stability against decay to free quanta, convention-dependent
```

Candidate metrics:

| Metric | Extraction from IRER data | External formula target | Units/scaling caveat | Match level |
|---|---|---|---|---|
| VK branch slope | Fit `Q(omega)` from C3 branch scans | `dQ/domega < 0` under stated convention | Sign can flip if using `mu = m^2 - omega^2`; always state branch variable. | SAME FAMILY |
| Energy per charge | `E/Q` from KG telemetry | compare to mass threshold `m` where model supports it | Project potential and units must be mapped before using `E/Q < m`. | SAME FAMILY |
| Charge conservation | `Q(t)` drift | conserved U(1) charge | Exact for KG stepper if numerics are correct. | EXACT EQUATION |
| Rest/boost relation | measured density speed vs intended Lorentz boost | Lorentz covariance of KG model | Existing project notes say boost fidelity is still measurement-limited. | SAME FAMILY |

Salih 2026 handling: source existence is verified, but relevance is `TO_VERIFY_RELEVANCE`. Do not use it as a foundation until Claude reviews it.

## V6 - Analogue-Gravity Acoustic Metric / Conformal-Law Diagnosis

External source IDs: `SRC-AG-001`, `SRC-AG-002`, `SRC-AG-003`, `SRC-AG-004`.

Acoustic metric family, schematic:

```text
g_mu_nu proportional to (rho/c) *
  [ -(c^2 - v^2), -v_j
    -v_i,          delta_ij ]
```

Polytropic relation used by the local future-work plan:

```text
p = K rho^gamma
c^2 = dp/drho proportional to rho^(gamma - 1)
Omega^2 proportional to rho^((3 - gamma)/2)
IRER form Omega^2 = (rho_vac/rho)^a maps to gamma = 2a + 3
```

Candidate metrics:

| Metric | Extraction from IRER data | External formula target | Units/scaling caveat | Match level |
|---|---|---|---|---|
| Effective exponent | Fit log `Omega^2` vs log `rho` | acoustic-polytropic exponent mapping | Current production soft clip dominates; fit only where unsaturated. | CLOSE ANALOGUE |
| Saturation fraction | Count cap/floor/graded regions | no direct external law; diagnostic of comparability | This is a geometry-contract hygiene metric, not a gravity result. | CLOSE ANALOGUE |
| Radial smoothness | radial `Omega^2(r)` cliff vs graded profile | analogue metric density response should be smooth under suitable EOS | Requires filled-background design before any probe use. | CLOSE ANALOGUE |
| Photon-fluid density/dispersion comparison | Future comparison of density-dependent sound speed or Bogoliubov-like dispersion | fluid-of-light experiments report sound-speed scaling with density | Photon-fluid paraxial mapping is only an analogue; no current gravity result follows. | CLOSE ANALOGUE |

Language rule: this is a design constraint and source comparison only. Do not call it a gravity result.

## V7 - Madelung / Bohm / Quantum-Hydrodynamic Stress Tensor

External source IDs: `SRC-MAD-001`, `SRC-MAD-002`, `SRC-MAD-003`.

Core hydrodynamic identities:

```text
psi = sqrt(rho) * exp(i S / hbar)
v = grad S / m
partial_t rho + div(rho v) = 0
Q_Bohm = -(hbar^2 / 2m) * Laplacian(sqrt(rho)) / sqrt(rho)
```

Stress-tensor-style form, convention-dependent:

```text
rho * D_t v = -grad V - div(P_Q)
P_Q approximately proportional to rho * Hessian(log rho)
```

Candidate metrics:

| Metric | Extraction from IRER data | External formula target | Units/scaling caveat | Match level |
|---|---|---|---|---|
| Continuity residual | `partial_t rho + div J` | Madelung continuity | Requires time-resolved fields and consistent current definition. | CLOSE ANALOGUE |
| Quantum pressure/stress proxy | Compare project `T_info` terms to phase/density gradient stress | quantum pressure/stress tensor forms | Project `T_info` is not automatically the Bohm stress tensor. | SPECULATIVE |
| Fisher-information-like density gradient | integrate `|grad sqrt(rho)|^2` or related terms | quantum potential / Fisher information relation | Needs careful normalization; useful only as a candidate metric. | CLOSE ANALOGUE |

## Current Gap Summary

| Gap | Consequence | Recommendation |
|---|---|---|
| NLS Galilean source is currently registry-backed by a secondary summary | Exact-equation target needs publication/textbook authority before external writeup | Replace with a primary textbook or standard NLS reference in next pass. |
| VK primary source not fully verified in this pass | VK formulas included but should be citation-hardened | Add original Kolokolov/Vakhitov or modern stability-theory reference before publication use. |
| Dataset availability is uneven | Most external comparisons are future candidates, not quantitative-now | Use `DATASET_CANDIDATES.md` readiness labels conservatively. |
| Salih 2026 exists but relevance is unreviewed | Cannot support IRER claims | Keep as `TO_VERIFY_RELEVANCE` until Claude reviews. |
| Second-tier sources are mostly context/future candidates | Prevents overclaiming from analogue or paper-only data | Keep Mitschke/Mollenauer as high-priority V1 source, NLSE as numerical sanity only, and photon-fluid/BEC entries as future candidates. |

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 2 commit(s), most recently `3eb93af` (2026-08-26)

**Harness code changed since it was written:** 3 commit(s) to `jax_scout/`.
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[FUTURE_WORK_AND_EXTERNAL_VALIDATION_PLAN]], [[external_validation/CODEX_EXTERNAL_VALIDATION_REPORT]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
