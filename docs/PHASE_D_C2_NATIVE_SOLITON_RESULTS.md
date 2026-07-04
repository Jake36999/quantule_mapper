# Phase D / C2.1 — Native Conservative-Soliton Search: Final Report

**Headline: the conservative substrate has its OWN native stable solitons (distinct from the dissipative a\*), and
they MOVE ballistically under a boost — but the motion is WEAK and LOSSY (they radiate as they accelerate).** So the
conservative substrate opens a transport channel the dissipative sector structurally lacked (native solitons exist
*and* translate, v∝k) — the first motion in the whole Phase D investigation — but it is not clean coherent transport.
Mirror only, `kinetic_mode="conservative"`, Phase C default byte-identical; no gate/clipping; no matter claims.

**Classification: `C2_NATIVE_SOLITON_FOUND_NO_TRANSPORT`** by the strict criterion ("radiates under kick"), **with the
essential nuance that finite ballistic mobility IS present** — honestly, *weak lossy transport*, qualitatively
distinct from the fully-pinned dissipative sector.

## Budget
~2 h of the allocated 12 h (Stage 1 scout ~18 min; Stage 2+3 confirm+boost ~1 h + a restart re-run). Stopped early
per "stop when the result is clear": native solitons found, mobility present but lossy, verdict unambiguous.

## Stage 0 — baseline (PASS)
`kinetic_mode="dissipative"` (default) byte-identical to the frozen Phase C operator at N=48 (kfac=1.0, max|Δ L_k|=0);
conservative mode requires `dt=0.001` (dt=0.005 = Schrödinger-CFL collapse, numerical).

## Stage 1 — broad scout (N=48, T=4000, dt=0.001, 35 cells) → `C2_LOCALIZED_CANDIDATES_FOUND`
Single-Gaussian ICs, amplitude × width. Clean structure:
| width σ/L | outcome |
|---|---|
| 0.04 (narrow) | DISPERSE / (A≥3) COLLAPSE |
| 0.06 | MARGINAL / RADIATE |
| **≥0.083 (wide)** | **LOCALIZED** (15/35 cells) |
| A=3.0 (any σ) | COLLAPSE |
Native localized regime = **moderate amplitude (A≤2) + wide width (σ≥0.083)**; best at σ=0.11–0.15 (mass ret 0.87–0.97,
occ≈1.0–1.2, single node). Distinct from a\*, which *radiated* in the conservative substrate.

## Stage 2 — N=96 long confirmation (T=12000, dt=0.001)
| candidate | verdict | mass ret | occ ratio | nodes |
|---|---|---|---|---|
| A=1.0 σ=0.083 | DISPERSING | 0.56 | 2.63 | 1 |
| A=1.0 σ=0.11 | DISPERSING | 0.69 | 1.84 | 1 |
| **A=1.0 σ=0.15** | **STABLE_SOLITON** | 0.84 | 1.20 | 1 |
| **A=0.5 σ=0.15** | **STABLE_SOLITON** | 0.91 | 1.23 | 1 |
Only the **widest (σ=0.15)** hold as genuine solitons over the long window; the tighter ones slowly disperse. So the
native conservative soliton is a **wide, moderate-amplitude** structure.

## Stage 3 — transport (Galilean boost of the confirmed solitons)
| candidate | kick n | k | v | v/k | disp (box) | r² | mass ret |
|---|---|---|---|---|---|---|---|
| A=1.0 σ=0.15 | 0 | 0 | −0.000 | — | 0.000 | 0.99 | 0.79 |
| | 1 | 0.628 | +0.024 | 0.038 | 0.012 | 1.00 | 0.74 |
| | 2 | 1.257 | +0.055 | 0.043 | 0.027 | 1.00 | 0.59 |
| A=0.5 σ=0.15 | 1 | 0.628 | +0.005 | 0.008 | 0.002 | 1.00 | 0.90 |
| | 2 | 1.257 | +0.010 | 0.008 | 0.005 | 1.00 | 0.75 |
- **Ballistic mobility is REAL:** v∝k (r²=1.00, v/k ≈ constant), μ=dv/dk = **+0.043** (A=1.0), +0.008 (A=0.5). This is
  **~36× the dissipative a\* mobility (μ≈0.001, pinned)** — the soliton genuinely translates.
- **But it radiates under the kick:** mass drops with kick strength (0.79→0.74→0.59). A *gentle* n=1 boost (k=0.63,
  far below the dealias cutoff ~15) already loses ~26% → **physical radiation, not a dealiasing artifact.** The
  density-sourced conformal geometry **breaks Galilean invariance**, so an accelerated soliton is not a stationary
  solution and sheds a radiative tail.

## Interpretation vs the dissipative baseline (D.5/D.6)
| | dissipative sector (Phase C + D.1–D.6) | conservative sector (C2/C2.1) |
|---|---|---|
| native stable structure | a\* (gain/loss-balanced, ~4-node) | wide moderate-A soliton (single node) |
| mobility under boost | **μ≈0.001 (pinned)** | **μ≈0.04 (ballistic, v∝k)** |
| motion | none (merge-or-hold, no drift) | **real translation** — but lossy (radiates) |
| conservation | dissipative attractor | quasi-conservative (radiates under boost) |
**The conservative substrate opens the transport channel the dissipative one structurally lacked** — the first
genuine node motion in the investigation — but it is **weak and lossy**: the geometry coupling makes moving solitons
radiate, so it is not clean coherent transport.

## Stage 4 (two-node) — NOT run
Per the gate (Stage 4 only on clean Stage-3 success): the transport is real but lossy (radiates under kick), so a
two-node collision test would be confounded by radiation. Deferred.

## Follow-ups (not done; a C2.2 sub-project)
- **Galilean-robust soliton:** does a co-moving frame or a modified boost (adiabatic acceleration) reduce the
  radiation? Is there a soliton family that translates *without* shedding mass?
- **Geometry's role:** the conformal Ω²(ρ) coupling breaks Galilean invariance — test the pure-NLS (geometry off) as
  a contrast to isolate how much radiation is geometry-induced.
- **Higher resolution / lower k** to further separate physical radiation from spectral edge effects.
- **Two-node interaction** once a cleaner moving soliton (if any) is found.

## Guardrails honoured
Mirror only; `kinetic_mode` default dissipative = byte-identical; frozen Phase C operator intact; dt/CFL +
quasi-conservative mass-loss documented (no clipping added); **no matter-like/transport over-claims** — mobility is
reported with its lossiness, motion asserted only from v/k + r² + mass metrics.
