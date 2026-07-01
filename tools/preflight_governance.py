"""
Pre-flight governance scanner (Step 4 gate).

Verifies the corrected-physics contract surface is present on the F: Quantule Mapper
root BEFORE any sweep is allowed to start. Exits 0 only if ALL checks pass.

Adapted from the test-bench contract assertions into a single hard gate covering the
seven required invariants.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(rel):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# (label, file, required_substrings[all must be present], forbidden_substrings[must be absent])
CHECKS = [
    ("corrected solver contract present", "orchestrator/contracts.py",
     ['SOLVER_CONTRACT_VERSION = "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1"'], []),
    ("compatibility gate present", "orchestrator/result_processor.py",
     ["def _evaluate_compatibility_gate", "self._evaluate_compatibility_gate("], []),
    ("/identity contract present", "orchestrator/run_identity.py",
     ["def write_identity_group"], []),
    ("param_omega0 / param_rho_vac split present", "orchestrator/contracts.py",
     ["DEFAULT_PARAM_OMEGA0", "DEFAULT_PARAM_RHO_VAC"], []),
    ("param_omega0 used independently in solver", "solver/core.py",
     ["params.get('param_omega0'", "1j * self.omega0"], []),
    ("A_dot_k_final naming present", "solver/run.py",
     ["'A_dot_k_final'"], ["'A_dot_final'"]),
    ("k=0 A-field gate present", "solver/core.py",
     ["self.A_k[0, 0, 0] = 0", "self.A_dot_k[0, 0, 0] = 0", "rho_k_safe[0, 0, 0] = 0"], []),
    ("no contract regression (linear_operator + contract version stamped)", "solver/run.py",
     ['"linear_operator": "-D*k^2 - eta + i*omega0"',
      '"solver_contract_version": SOLVER_CONTRACT_VERSION'], []),
    ("legacy-label regression guard present", "mcp_server/data_access.py",
     ["A_dot_k_final", "legacy label"], []),
    # --- A-field (causal-affect) layer: contract-versioned, default-off, still RESERVED ---
    ("causal-affect contract versions present", "orchestrator/contracts.py",
     ['CAUSAL_AFFECT_CONTRACT_VERSION = "IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1"',
      "ADDITIVE_POT_CONTRACT_VERSION"], []),
    ("A-coupling param default-off (gamma_A=0 baseline)", "orchestrator/schema_utils.py",
     ['"param_affect_coupling", "REAL DEFAULT 0.0"'], []),
    ("affect discriminator/topology mapping present", "orchestrator/run_identity.py",
     ["affect_strength_for_params", "param_affect_topology", "vacuum_ref"], []),
    ("A-field coupling still RESERVED (not wired into geometry)", "gravity/unified_omega.py",
     ["(rho_vac / rho_capped) ** a"], ["param_affect_coupling", "rho_vac_eff", "gamma_A"]),
    # --- SDG/emergent-geometry diagnostic: contract-stamped + PASSIVE ---
    ("SDG diagnostic contract stamped", "jax_scout/geometry_diag.py",
     ['DIAG_CONTRACT_VERSION = "IRER-SDG-DIAG-v1"'], []),
    ("SDG diagnostic is passive (no solver feedback)", "jax_scout/geometry_diag.py",
     ["never alters solver"], ["physics.step", "import physics"]),
    # --- FMIA transfer / interaction-rate diagnostic: contract-stamped + PASSIVE ---
    # (this one DOES run the unmodified solver to capture trajectories, so it is not held
    #  to geometry_diag's "no physics.step" rule; it must instead read the single-source-of-
    #  truth geometry path and declare it never alters solver physics).
    ("FMIA transfer diagnostic contract stamped", "jax_scout/transfer_diag.py",
     ['TRANSFER_DIAG_CONTRACT_VERSION = "IRER-FMIA-TRANSFER-DIAG-v2"'], []),
    ("FMIA transfer diagnostic passive (reads SoT geometry, never solver feedback)",
     "jax_scout/transfer_diag.py",
     ["never solver feedback", "param_skip_topology_cap", "derive_stable_conformal_factor"], []),
]


def main():
    print(f"PRE-FLIGHT GOVERNANCE SCAN  (root: {ROOT})\n")
    results = []
    for label, rel, required, forbidden in CHECKS:
        try:
            text = read(rel)
        except FileNotFoundError:
            results.append((label, rel, False, "FILE MISSING"))
            continue
        missing = [s for s in required if s not in text]
        present_forbidden = [s for s in forbidden if s in text]
        ok = not missing and not present_forbidden
        detail = ""
        if missing:
            detail = "missing: " + " | ".join(missing)
        elif present_forbidden:
            detail = "forbidden present: " + " | ".join(present_forbidden)
        results.append((label, rel, ok, detail))

    width = max(len(r[0]) for r in results)
    all_ok = True
    for label, rel, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        line = f"  [{mark}] {label:<{width}}  ({rel})"
        if detail:
            line += f"\n         -> {detail}"
        print(line)

    print("\n" + ("GOVERNANCE GATE: PASS — sweep authorized." if all_ok
                  else "GOVERNANCE GATE: FAIL — DO NOT START SWEEP."))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
