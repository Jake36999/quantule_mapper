# A-field (causal-affect) readiness gate

Governs when the causal A-field coupling (`γ_A ≠ 0`) may be promoted from a **passive,
monitored diagnostic** to **active feedback** in the solver. Enforces the rule:
*stabilise and contract-version the A-field/tensor layer before active feedback.*

## Intended coupling (RESERVED — not yet wired)

```
A  : finite-speed causal field-of-affect (already integrated in solver/core.update_field_of_affect,
     decoupled from ψ; k=0 secular mode gated).
ρ_vac_eff = max(ρ_vac + γ_A · Ã, ε)        # γ_A = param_affect_coupling, default 0.0
Ω²        = (ρ_vac_eff / ρ)^a                # γ_A = 0 reproduces the current LOCAL-RHO solver exactly
```

The coupling line is **not implemented in `gravity/unified_omega.py`** (verified by the
governance check "A-field coupling still RESERVED"). `update_field_of_affect` computes `A`
but does **not** feed it back into the geometry. This is intentional.

## Current state (2026-06-19) — STABLE, gate NOT yet satisfied for activation

| Item | Status | Evidence |
|---|---|---|
| Contract versions reserved | ✅ | `CAUSAL_AFFECT_CONTRACT_VERSION`, `ADDITIVE_POT_CONTRACT_VERSION` |
| Discriminators in ledger/identity | ✅ | `variant_label`, `affect_topology`, `affect_strength` (DATA_CONTRACT §3.1) |
| `param_affect_coupling` default-off | ✅ | schema `REAL DEFAULT 0.0`; `affect_strength_for_params({})==0.0` |
| Topology → contract mapping | ✅ | `run_identity`: vacuum_ref→CAUSAL-AFFECT, additive_potential→ADDITIVE-POT |
| k=0 A-field secular gate | ✅ | `solver/core.update_field_of_affect`; `test_run_identity::TestK0Runaway` |
| Coupling still reserved (γ_A inactive) | ✅ | governance check on `unified_omega` |
| SDG/emergent-geometry diagnostic | ✅ passive, stamped `IRER-SDG-DIAG-v1`, falsifiable | `jax_scout/geometry_diag.py`; `test_sdg_diagnostic.py` |
| Contract/identity tests | ✅ 86 pass | `test_data_contract`, `test_run_identity`, `test_ledger_identity` |

## Gate to ENABLE γ_A > 0 (all must hold)

1. `tools/preflight_governance.py` PASS, including the 6 A-field/SDG checks.
2. A-field contract + identity tests green (86).
3. SDG diagnostic falsifiable + passive (`test_sdg_diagnostic.py` green) — it must demonstrably
   distinguish coherent structure from noise / runaway, else it cannot falsify anything.
4. When the coupling is wired into `unified_omega` (`ρ_vac_eff` line):
   - bump `solver_contract` → `IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1` (vacuum_ref) /
     `…-ADDITIVE-POT-…` (additive_potential) when `γ_A ≠ 0`;
   - stamp `variant_label` / `affect_topology` / `affect_strength` into `/identity`;
   - keep `param_affect_coupling` default 0.0 (explicit opt-in only);
   - update the "still RESERVED" governance check (it will, by design, fail once the
     coupling appears — replace it with an "A-coupling wired + contract-bumped" check).
5. Run γ_A>0 in **monitor-only** mode first (diagnostic records `T_info`/Ω²/curvature/SDG
   residual into provenance; **no** change to ranking/promotion). Promote to active feedback
   only after the coupled runs show **bounded emergent curvature following resonance density**
   (the headline question), stable across seeds/params — i.e. NOT runaway curvature or
   decoupled tensor noise.

## Why this matters now

The γ_A=0 deep-dive showed multi-node states are **independent self-focusing condensates**
(no mutual support, mostly runaway/decoupled geometry). The causal A-field is the candidate
**long-range coupling** that could produce genuine mutual support — but it must clear this
gate first so an unstable tensor layer is never mistaken for physical evidence.
