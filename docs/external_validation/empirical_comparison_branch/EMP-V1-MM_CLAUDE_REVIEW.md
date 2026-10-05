# Claude Review — EMP-V1-MM (Mitschke & Mollenauer 1987, Fig. 3)

**Verdict:** `WEAK_CLOSE_ANALOGUE_SIGN_LAW_SUPPORTED_NO_QUANTITATIVE_CLAIM`
**Status:** review complete; the comparison stays `PROVISIONAL` context — no verdict/gate/framing change.
**Match level:** `CLOSE ANALOGUE` (ceiling; not raised).

## What the source is

Fig. 3 of Mitschke & Mollenauer (Opt. Lett. 12, 355, 1987): output pulse separation `σ_out/τ` vs relative phase
`φ` for two initial spacings — **close** `σ_in/τ=1.53` (→ q=0.765) and **wide** `σ_in/τ=3.82` (→ q=1.91). Codex
digitized it to a signed separation-change proxy `interaction_measure = σ_out/τ − σ_in/τ` (>0 grew/repel-like,
<0 shrank/attract-like).

Codex's execution was diligent and honest: complete provenance, explicit extraction-uncertainty estimates, and a
candid note that the **raw figure bitmap could not be downloaded** (host blocked it) so extraction was from a
rendered/indexed view — recorded rather than papered over. Copyright handled correctly (only derived data points
committed; raw off-git). No production/config/Hunter changes; protected diff empty.

## The decisive structure: the two spacings are in different physical regimes

| series | q | sign-law accuracy | reading |
|---|---|---|---|
| **wide** (weak interaction) | 1.91 | **7/7 = 1.00** | in-phase attracts, out-of-phase (weakly) repels — follows `cos Δφ` |
| **close** (strong / merger) | 0.765 | 9/15 = 0.60 | attractive branch corrupted; every miss is at φ≈0 |

The **close series must be excluded** from the force-law comparison. At small spacing the in-phase pulses attract,
**collide, and re-separate within the fiber length**, so `σ_out` reflects the phase of a nonlinear oscillation, not
the instantaneous two-body force. That is why its attractive branch reads *positive* (grew) where attraction is
expected — the same merger/self-frequency-shift caveat the paper and the provenance flag. Output-separation is not
a valid force proxy in this regime; counting it against the sign law is a category error, not a falsification.

The **wide (weak-interaction) series is the legitimate comparison**, and it supports the law:
- **In-phase attraction is clearly confirmed** — `measure = −0.37, −0.17, −0.32` at φ≈0…π/4, unambiguously
  negative and above the extraction noise.
- **The crossover to repulsion at anti-phase is consistent but sits at the noise floor** — `measure = +0.03…+0.08`
  at |φ|≈3π/4…π, within the stated ±0.08–0.12 extraction uncertainty. Physically expected (wide pairs interact
  weakly, so anti-phase repulsion is tiny), but I will not overstate it: the *attractive* branch carries the
  signal, the repulsive branch is merely not contradicted.

## No quantitative (λ) comparison is supported

There is **no valid decay-rate comparison** from this figure: only two initial spacings, in two different dynamical
regimes, with an output-separation proxy. The raw 2-point fit (`λ≈1.57`, ratio 0.79 vs our C2 `λ≈1.98`,
`R²=0.025`) is statistical noise that happens to look plausible — precisely the trap the match-level discipline
exists to catch. The comparator has been **hardened** to suppress it: `experimental_lambda` is now `NOT REPORTED`
unless the fit has ≥3 distinct q and `R²≥0.5`, and sign accuracy is broken out per spacing so regime structure is
visible rather than blended. (Self-test still recovers λ on clean synthetic data.)

## The honest statement to carry forward

> An independent 1987 optical-fiber soliton experiment, in its weak-interaction branch, reproduces the
> in-phase-attract / out-of-phase-repel phase dependence — a `CLOSE ANALOGUE` consistent with the V1 / C2.9 π/2
> two-body sign law. The strong-interaction branch is dominated by collision/merger dynamics and is excluded. No
> quantitative decay-rate (λ) correspondence is claimed from this figure.

## What this does and does not establish

- **Does:** a real experiment's weak-interaction data is *consistent with* our sign law — a modest, honest external
  corroboration of the qualitative π/2 structure.
- **Does not:** establish physical correspondence, a λ match, or anything about the close/merger regime; it does
  not touch matter, gravity, or unification, and changes no verdict.

## Next steps (optional)

1. A **quantitative λ** would need a dataset with **≥3 initial spacings in the weak-interaction regime** (or a
   direct force/acceleration measurement), not an output-separation figure. Flag as a future target requirement.
2. **EMP-V5-NG** (Nguyen matter-wave collisions) remains the better next target — a phase-dependent *outcome*
   diagram maps more directly onto our C3 phase×speed grid — but needs the V5 comparator path built first.
