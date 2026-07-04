# Phase D — C2 Conservative Branch, First-Pass Results

**Classification: `C2_ASTAR_NOT_A_TRANSPORT_SOLITON`** — the conservative substrate is implemented, parity-proven, and
numerically viable, but the **a\* dissipative attractor is not a conservative soliton**: dropped into the conservative
substrate it **radiates (~50% mass loss) and does not translate** under a Galilean boost (μ≈0). Demonstrating (or
refuting) conservative transport therefore needs the substrate's **own** solitons — a search, not a\*-reuse. Mirror
only, `kinetic_mode` default = dissipative byte-identical; no solver default / gate change.

## Implementation + parity (PASS)
`kinetic_mode` flag on the mirror (`physics.py`): `"conservative"` → `L_k = −i·D·k²` (dispersive, no gain/loss η) and
the nonlinear RHS ×`kfac=1j` (Hamiltonian). Parity: `kinetic_mode="dissipative"` (and the default) reproduce the
frozen baseline **byte-for-byte** — `max|Δ|=0.0` for `L_k, E, E2, Q, f1, f2, f3`; `kfac` 1.0 (dissipative) vs 1j
(conservative); conservative `L_k` is pure-imaginary (`max|Re|=0`). C1 (`D_imag`) parity also intact.

## Numerical behaviour
- **dt matters (Schrödinger CFL).** At `dt=0.005` the conservative run goes non-finite within ~1000 steps —
  **numerical**, from the stiff dispersion (`|D·k²|` up to ~7500), not physics. At **`dt=0.001` it is stable** at all
  tested amplitudes (a\*×{1.0, 0.5, 0.3, 0.1}) — 4 nodes persist, bounded.
- **Only quasi-conservative.** Mass is **not** conserved (drops to ~0.5–0.7): the dealias mask removes the high-k
  radiation the dispersion generates, and the density-sourced conformal geometry coupling is not exactly
  self-adjoint. So this is a *dispersive-with-radiation* substrate, not a clean norm-conserving NLS.

## The transport test (boost) — NO ballistic motion for a\*
Galilean boost `ψ·exp(i k x)` on the a\* node, conservative, `dt=0.001`:
| kick n | k | v_x | v/k | disp | mass |
|---|---|---|---|---|---|
| 0 (control) | 0 | +0.0097 | — | 0.006 box | 0.53 |
| 2 | 1.257 | +0.0112 | +0.009 | 0.005 box | 0.46 |

**mobility μ = dv/dk = +0.0012 ≈ 0.** The boost barely changes the velocity (n=0 and n=2 nearly identical), and the
node sheds ~50% of its mass → it **radiates and stays pinned**, it does not translate as a coherent soliton.

## Interpretation (honest)
- The **a\* object does not carry over** to the conservative substrate — it is a *dissipative* gain/loss-balanced
  attractor, not a conservative soliton, so in the Hamiltonian substrate it **radiates** rather than forming a stable
  movable structure. This mirrors the whole Phase D lesson: **the dissipative and conservative sectors have
  different stable objects** (RFC: C2 is "a different object, not an extension of a\*").
- Therefore **C2 transport is neither demonstrated nor refuted here** — the experiment used the wrong object. The
  correct test is: **search the conservative substrate for its native stable solitons** (self-consistent NLS ground
  states of the IRER cubic-quintic-septic nonlinearity, likely lower-amplitude), then boost / interact *those*.

## Next (a real sub-project, not done here)
`C2.1 — conservative-soliton search`: at `dt≈0.001`, scan amplitude / width IC families in `kinetic_mode="conservative"`
for a **stable, non-radiating** localised structure (bounded mass, fixed profile); if found, run the boost (does it
move ballistically?) and the two-node interaction (attract/scatter?) — the actual C2 transport test, compared against
the dissipative D.5/D.6 baseline. If no stable conservative soliton exists for the feb coefficients, that is itself
the finding.

## Guardrails
Mirror only; `kinetic_mode` default dissipative = byte-identical; frozen Phase C operator intact; quasi-conservative
mass-loss + CFL documented (no clipping added to force conservation); **no matter-like/transport claims** — μ≈0 for
a\* is reported as a null *for that IC*, not a universal transport verdict.
