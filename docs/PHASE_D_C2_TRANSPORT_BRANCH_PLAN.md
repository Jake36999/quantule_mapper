# Phase D — C2 Conservative / NLS Transport Branch — Plan

**The separate transport-substrate branch** opened after the dissipative sector was fully characterised
(D.1–D.6: IRER dissipative nodes couple + merge but do **not** move — no current, no drift). C2 tests whether a
**conservative** substrate — where |ψ|² is (near-)conserved and structures *can* carry momentum — gives the node
**motion** the dissipative baseline structurally lacks. **This is a DIFFERENT physics regime, not an extension of
a\***; D.5/D.6 are the dissipative *baseline comparison*.

## The substrate (mirror only, default-off)
A `kinetic_mode` flag on the jax_scout mirror (`physics.py`), **default `"dissipative"` = frozen Phase C baseline,
byte-identical** (`kfac=1.0`, `L_k` unchanged). `kinetic_mode="conservative"`:
- `L_k = −i·D·k²` — a **dispersive** kinetic (Schrödinger-like), with **no gain/loss η** (η dropped);
- the whole nonlinear RHS is multiplied by **`kfac = 1j`** → the flow is `ψ_t = i·[D∇²ψ + N(ψ,ρ)]` with the *same*
  IRER cubic-quintic-septic nonlinearity + conformal geometry, now as a **Hamiltonian** operator → |ψ|² conserved.
It records `kinetic_mode` in provenance; the sweep/Hunter path stays dissipative. No solver default / gate change.

## Why this can work (and can fail)
A cubic *focusing* NLS in 3D is supercritical → **wave collapse (blow-up)**. But the IRER nonlinearity has
**quintic + septic** saturation (`s·ρ², f·ρ³`) — the cubic–quintic(–septic) NLS is known to support **stable 3D
solitons** — so the saturating terms *may* arrest collapse and give localised, movable conservative nodes. Whether
they do, for the feb coefficients, is an empirical question. Failure modes: **collapse** (focusing blow-up) or
**dispersion** (no localised nodes) — either is a legitimate finding.

## Tests (mirror; pure measurement)
1. **Parity** — `kinetic_mode="dissipative"` (and the default) reproduce the frozen baseline **byte-for-byte** (L_k,
   E, Q, f-coeffs, kfac, and a short run); C1 (`D_imag`) parity still holds.
2. **Conservative single-node** — evolve an a\* node in conservative mode: does |ψ|² **conserve**? does it stay a
   **localised soliton**, **disperse**, or **collapse**?
3. **Conservative two-node** (the key contrast) — two nodes at a separation that **held** in the dissipative
   substrate (D.5): do they now **drift / attract / orbit / scatter** (genuine transport), or still hold?
4. **Galilean boost** (if solitons are stable) — a phase ramp should translate a conservative soliton ballistically
   (v ∝ k), the mobility the dissipative substrate lacked (C1).

## Success / failure
- `C2_CONSERVATIVE_TRANSPORT` — stable localised conservative nodes that **move** (ballistic boost) and **interact**
  (two-node drift/attract/scatter) → a transport regime exists, in contrast to the dissipative pinned/merging baseline.
- `C2_COLLAPSE` — focusing blow-up (saturation insufficient).
- `C2_DISPERSE` — no localised nodes (defocusing / radiating).
- `C2_NO_MOTION` — localised but still pinned (transport absent even conservatively — a strong null).

## Guardrails
Mirror only; `kinetic_mode` default = dissipative, byte-identical; Phase C frozen operator intact; no gate/provenance
break; **no matter-like claims** (motion asserted only from metrics); records `kinetic_mode`. The dissipative
sector (Phase C + D.1–D.6) stands unchanged as the baseline this branch is compared against.
