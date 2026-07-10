# IRER Simulation Results & Theory Status

**Report 5 of 5 — public-facing synthesis.** Written in the disciplined style of the Declaration of Intellectual
Provenance (Tab 1), but updated with what the Quantule Mapper simulations have actually found. It states, plainly,
what is confirmed *internally* (as a numerical model), what is falsified, what was retracted for instrument bugs,
what is paused, and what remains speculative — and it points at external analogue/data searches. **This is not a
claim that IRER is physically true, nor that matter or gravity has been demonstrated.** Author of the theory: Jake
McIntosh; AI as formalization/implementation/review tool.

## 1. What IRER proposes (author's framework, in brief)
IRER models reality as emerging from an a-temporal informational substrate (AIS) carrying resonant, phase-bearing
excitations (OIW). Local resonance density (RD) and readiness-to-actualize (PAS) drive a continuous "collapse" —
Resolution Field Dynamics — that produces stable structures (Quantules) or novelty. Structures follow paths of least
informational action (FMIA) across an informational manifold whose geometry they also shape; apparent forces are
gradient-derived, and time is the ordered chronology of resolution. (Full concept meanings and their standard-language
translations: Report 1.)

## 2. What Quantule Mapper actually implements
A family of well-posed nonlinear field PDEs sharing a cubic–quintic–septic nonlinearity g(ρ) and spectral machinery:
a dissipative Ginzburg–Landau sector (Phase C / C0–C1), a conservative nonlinear-Schrödinger sector (C2, in
geometry-coupled and true-flat forms), and a nonlinear Klein–Gordon sector (C3), plus a density-sourced conformal
geometry Ω²(ρ) and an informational stress tensor T_info as read-only diagnostics. (Equation↔code: Report 2.) This is
a numerical test bench, not the full theory: several IRER concepts are only partially implemented or not implemented
(Report 1).

## 3. Confirmed internally (as measured numerical observables)
- **A stable dissipative attractor exists** (Phase C): a\*≈×1.15 cubic gain, a real long-time gain/loss-balanced
  standing structure, bracketed ±0.5%, seed/N128/T144k-robust.
- **Corrected conservative solitons transport** (C2.7): on the fixed true-flat substrate a true soliton translates at
  exactly the Galilean velocity, v/2Dk = 0.9999, mass 0.9999 (N=96) — clean coherent transport.
- **KG Q-balls are VK-stable and transport inertially** (C3): existence at the C2-mapped point; energy + U(1) charge
  conserved to ~1e-13; VK slope dQ/dω = −849 < 0 (stable branch); density co-moves under a Lorentz boost.
- **A shared two-body phase-force law** (C2.9 & C3): pairwise force attractive for Δφ<π/2, repulsive above, crossover
  at **Δφ=π/2** — the same in both first-order (NLS) and second-order (KG) substrates (cross-substrate universality).
- **A mapped C3 collision phase diagram**: capture is generic across (relative-phase × speed); transmission is a
  narrow feature confined to exact anti-phase (a destructive-interference node) below ~0.5c; the static force law does
  **not** govern collision outcomes.

## 4. Falsified / retired (real negatives)
- **Prime-SSE as a stability predictor** — 0/60; the objective is retired.
- **a\* mobility** — a kick imparts no net motion (the dissipative operator has no inertial channel); static wells
  cause accretion, not migration.
- **Dispersive mobilization (C1)** — adding a dispersive kinetic term destabilises a\* (mass runaway) rather than
  mobilising it.
- **feb/a\* as a conservative soliton family** — the feb coefficients are structureless on the true-flat substrate
  (g_max ≈ 0.23 below the box binding floor); the conservative movers need s<0 saturation + box-compatible D.
- **Nodes-as-topological-vortices**, tensor-routing, Payan-alignment-as-stability-predictor — all null (Phase C).
- **"Broad repulsive-channel transmission"** — refined away: statically-repulsive phases still capture; only exact
  anti-phase transmits (a node feature).
- **The simple de-saturated gravity fix** — loosening the soft-clip steepens the cliff (the divergence is from
  (ρ_vac/ρ)^a at near-zero vacuum, not the cap).
- **C2′ as a direct gravity-ladder unblocker** — it changes conservation form, not the Ω²(ρ) factor.

## 5. Retracted due to instrumentation bugs (a methodological achievement)
Three instrument bugs were caught by chasing a contradiction against a known identity, and their verdicts corrected:
- **C2.6 geometry-off (D_eff = D/151):** the "conservative pinning / flow-through / drag μ≈0.04" verdicts (C2.1b,
  C2.2b, C2.3b, C2.4) were artifacts of a substrate that was never actually flat. Fixed and Codex-re-audited.
- **C2.8b fragile peak-tracker:** the two-node "elasticity" (unphysical e=3.21) was retracted; replaced by the
  momentum-density observable (C2.9).
- **C3 boost-IC (missing carrier phase):** the "density barely moves (v_frac 0.04)" reading was a malformed boost;
  the corrected Lorentz boost shows the density co-moving.
The lesson is now a standing rule (Report 4 §3.6): correct boost IC + robust observable + conservation telemetry gate
every transport/interaction claim.

## 6. Paused
- **The geometry-density / gravity-like ladder** (rungs B onward). The channel is *live* (a coherent load produces a
  geometry-dependent Ω²/T_info response), but the production Ω²(ρ) is a saturation cliff, not a graded potential.
  **Re-entry condition:** a validated stable-overdense-load-on-ρ_vac-background regime (a research sub-project) that
  makes Ω²(ρ) graded, before any neutral-probe rung.

## 7. Speculative / not yet implemented
- Full **Payan-state mechanics** (quantized internal spin modes as named primitives) — only phase/winding/current
  analogues exist.
- **Angular-deficit** quantule theory and **full manifold topology** beyond the conformal factor.
- The **observer / Observer-Resolution Loop**, Chrono-Coherence Fields, Chorotic Boundaries, SNRC.
- Any **external empirical correspondence** — untested; a search direction, not a claim.

## 8. External analogue & data candidates to search next
The strongly-implemented, confirmed core maps onto standard model families; these are the places to look for prior
theory and comparable phenomenology (as *candidates*, not asserted correspondences):
| implemented phenomenon | model family to search | possible external data analogues |
|---|---|---|
| conservative solitons, v=2Dk transport | nonlinear Schrödinger solitons; Gross–Pitaevskii | BEC bright/dark solitons; optical spatial/temporal solitons |
| VK-stable Q-balls, inertial transport | complex Klein–Gordon Q-balls; non-topological solitons | field-theory Q-balls; oscillons; (analogue) relativistic solitons |
| π/2 two-body phase-force; capture/transmission | soliton interaction forces; Peierls–Nabarro | colliding BEC/optical solitons; soliton molecules |
| dissipative standing attractor a\* | dissipative solitons; complex Ginzburg–Landau | dissipative optical solitons; reaction–diffusion patterns |
| density-sourced conformal geometry Ω²(ρ) | scalar-tensor / dilaton; analogue gravity | analogue-gravity (BEC/optics) experiments |
| informational stress tensor T_info | scalar-field stress-energy | — (diagnostic; no direct experiment yet) |
VK stability, U(1) charge conservation, and the Galilean/Lorentz transport identities are the quantitative hooks a
comparison would test first.

## 9. Cautious-language contract (binding on this bundle)
- No claim that **IRER is proven**. No claim that **matter** or **gravity** has been demonstrated. No
  **emergent-physics** proof.
- Every result is a **measured observable in a numerical model**, with its conservation/parity gate and caveats.
- The gravity target is stated only as a **candidate mechanism** ("geometry-density interaction", "sequential
  tensor-mediated response", "not yet an emergent-gravity claim"); never "mass attracts mass" or "Newtonian law
  reproduced".
- Nulls, falsifications, and retractions are **first-class results**, preserved, not hidden.
- Conceptual authorship is **Jake McIntosh's**; AI assistance is formalization/implementation/review.

## 10. One-paragraph status
The stability sector is **closed** (a real pinned attractor; its mobility and structural hypotheses null). The
transport/coupling sector is **answered positive across two conservative substrate families** (NLS solitons at v=2Dk;
VK-stable KG Q-balls transporting inertially), which share a single π/2 two-body phase-force law and a mapped
collision phase diagram. The geometry-density gravity-like sector is **live but paused** at a characterized gate. The
project's credibility rests as much on the three instrument bugs it caught and corrected as on the positives. No
matter claim, no gravity claim, no emergent-physics claim.

## Cross-references
Concepts & translation: Report 1. Equation↔code: Report 2. Identities/proofs: Report 3. Architecture: Report 4.
Verdict ledger: `docs/IRER_MASTER_HYPOTHESIS_CATALOG.md`. Sector detail: `docs/PHASE_D_CLOSEOUT_CONSOLIDATION.md`
and the per-experiment docs. Original theory & authorship: `docs/_Declaration of Intellectual Provenance v9.txt`.
