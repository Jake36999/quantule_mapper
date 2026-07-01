---
name: test-bench-adaptation
description: "The F:\\Testing_suite \"test bench\" — what it is and how the scanner was adapted to the current system"
metadata: 
  node_type: memory
  type: project
  originSessionId: bdc77455-e79d-49ed-be46-edb9996fa77e
---

`F:\Testing_suite` is a separate legacy "test bench" repo (NOT under quantule_mapper): a governance **scanner** (`irer-test-bench-scanner/`) + FastAPI **middleware** (`bench-middleware/`) + Next.js "God View" **UI** (`workbench-ui/`). It was built to audit a retired JAX / `IRERProcessor` system; much of it was stubbed/legacy.

**Adapted 2026-06-18 (scanner only, fully runnable + tested):**
- `rules/governance.yaml` rewritten to encode the **current DC-v1.0 Golden State** as Governance-as-Code (16 rules: A_dot_k_final naming, k=0 gate, omega0 split, /identity write, composite PK, compatibility gate `_champion_eligible`, rho_vac>=0.05 config bound, eval/exec/secret checks). The legacy `IRERProcessor`/`jax.lax.scan` rules are gone.
- New `core/governance_engine.py` (dep-light: yaml+ast+re+json, NO chromadb/termcolor/KEL) that **actually applies** the rules — the legacy `scanner_main.py` took `--rules` but never loaded them.
- `scanner_main.py` rewritten: LLM args optional (default `quick`), runs from any cwd, only `.py` AST-parsed, writes `scan_results.json` + `scan_summary.md`, removed the `sys.exit` KEL hard-import. Exit codes: 0=COMPLIANT, 2=NON-COMPLIANT(valid), other=failure.
- `bench-middleware/app/routers/scanner.py`: treats exit 0 AND 2 as "results available" (was 500 on any non-zero); fixed missing `version` field in `ScanResponse`.
- Tests: `tests/test_governance_engine.py` (9, all violation types proven) + existing `test_bundler.py` (4) = 13 pass.

**Result:** running `python scanner_main.py --path F:/quantule_mapper` reports **16/16 COMPLIANT** — the current system provably meets its own DC-v1.0 contract.

**Still legacy / not adapted:** the `bench-middleware` (Redis/SSE, needs uvicorn+redis to run) and `workbench-ui` (Next.js; see `current_issues.json` — chaos-mode isolation, forensics dock, hydration race, hardcoded JAX jargon ISSUE-005). The KEL librarian (chroma vector store) is unused by the quick scan. See [[dc-v1-hardening-state]].
