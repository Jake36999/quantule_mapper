# GPU-Box Runbook — DC-v1.0 cutover & γ_A=0 regression

**Version:** RB-v1.0  **Date:** 2026-06-18
**Audience:** operator on the CuPy/GPU box (this box, `F:\quantule_mapper`, has **no** CuPy — these steps run only where a GPU + CuPy are present).
**Companions:** docs/DATA_CONTRACT.md, docs/OUTPUT_HIERARCHY.md, docs/MCP_TOOLS_SPEC.md

This runbook covers the two operations that must happen on the GPU box before any
A-coupling work (the user's locked steps 2–3), plus the smoke check and how to
launch the read-only MCP server.

---

## 0. Preconditions

```bash
# On the GPU box, from the repo root, on the same commit as the dev box.
python -c "import cupy; print('cupy', cupy.__version__)"      # must succeed
python -c "import h5py, numpy, pydantic; print('deps ok')"
git rev-parse HEAD                                            # record the code hash
```

All identity/ledger/provenance logic is CuPy-free and was already verified on the
dev box (113 tests). On the GPU box you are validating the **numerical** path only.

---

## 1. Step 2 — Ledger composite-PK migration (one-time, per live ledger)

Pre-DC-v1.0 ledgers key `runs` on `config_hash` alone, which collapses multi-seed
runs. The migration rebuilds `runs` with PRIMARY KEY `(config_hash, seed)` and is
**non-destructive** (preserves rows; legacy rows get `seed=0`). It is a no-op if
already migrated.

```bash
# 1a. BACK UP the ledger first (always).
cp simulation_ledger.db "simulation_ledger.db.bak.$(date -u +%Y%m%dT%H%M%SZ)"

# 1b. Inspect the current PK (expect {'config_hash'} on a pre-migration DB).
python -c "from orchestrator.schema_utils import runs_primary_key as pk; print(pk('simulation_ledger.db'))"

# 1c. Run the migration (idempotent; returns True when PK is composite).
python -c "from orchestrator.schema_utils import migrate_runs_to_composite_pk as m; print(m('simulation_ledger.db'))"

# 1d. Confirm.
python -c "from orchestrator.schema_utils import runs_primary_key as pk; print(pk('simulation_ledger.db'))"
#   -> {'config_hash', 'seed'}
```

If the hunt uses a different ledger filename (e.g. `sqlite_database.db` from
`burn_in_config.json:db_path`), pass that path instead. Repeat per ledger.

> Note: fresh ledgers created after this commit already have the composite PK and
> all discriminator columns; the migration is only for pre-existing DBs.

---

## 2. Step 3 — γ_A=0 baseline regression

**Goal:** confirm the modularization (`solver/` package), the k=0 affect-field gate,
and the `param_rho_vac`→`param_omega0` split reproduce the pre-refactor **ψ physics**
when the A-coupling is off (`param_affect_coupling` absent/0) and `param_omega0`
is left unset (so it defaults to `param_rho_vac`).

### 2a. Determinism self-check (cheap, do first)

Same code, same seed → byte-identical artifact.

```bash
python worker_cupy.py --params configs/config_true_golden.json --output /tmp/det_a.h5
python worker_cupy.py --params configs/config_true_golden.json --output /tmp/det_b.h5
python - <<'PY'
import h5py, numpy as np
a = h5py.File('/tmp/det_a.h5'); b = h5py.File('/tmp/det_b.h5')
for k in ('psi_final','omega_sq_final'):
    print(k, 'max|Δ| =', float(np.max(np.abs(a[k][:]-b[k][:]))))
PY
#   expect 0.0 for both
```

### 2b. Refactor-fidelity check (modular vs pre-split monolith)

Find the last commit where `worker_cupy.py` was the monolith (before the `solver/`
split), extract it standalone, and diff ψ against the current modular solver.

```bash
# Find the pre-split monolith revision (the commit before solver/ was introduced).
git log --oneline -- worker_cupy.py | head
# Extract that monolith to a temp file (it is self-contained: imports cupy + unified_omega):
git show <PRE_SPLIT_REV>:worker_cupy.py > /tmp/worker_OLD.py

# Run both on the SAME small config + seed:
python /tmp/worker_OLD.py --params configs/config_true_golden.json --output /tmp/old.h5
python worker_cupy.py      --params configs/config_true_golden.json --output /tmp/new.h5

python - <<'PY'
import h5py, numpy as np
old = h5py.File('/tmp/old.h5'); new = h5py.File('/tmp/new.h5')
def d(k): return float(np.max(np.abs(old[k][:]-new[k][:])))
print('psi_final      max|Δ| =', d('psi_final'))        # expect 0.0
print('omega_sq_final max|Δ| =', d('omega_sq_final'))   # expect 0.0
# Telemetry energy:
print('energy         max|Δ| =', float(np.max(np.abs(old['telemetry/energy'][:]-new['telemetry/energy'][:]))))  # 0.0
PY
```

### Pass / fail criteria

| Quantity | Expected | Meaning if it differs |
|---|---|---|
| `psi_final` max\|Δ\| | **0.0** | refactor/omega0-split changed the ψ physics — investigate before proceeding |
| `omega_sq_final` max\|Δ\| | **0.0** | geometry path changed |
| `telemetry/energy` max\|Δ\| | **0.0** | integration changed |
| `A_final`, `A_dot_k_final` | **WILL differ** ✅ | expected — the k=0 secular-runaway gate now zeroes the DC mode; A is passive so this does not touch ψ. This difference is *correct*, not a regression. |

> Do **not** treat an SSE change as the success signal here. This step is purely a
> byte-level physics-equivalence check of the refactor with the coupling off.
> Record the diffs in the run log.

---

## 3. Smoke check (optional sanity before a campaign)

A capped run (`N_grid ≤ 32`, `T_steps ≤ 100`) that never touches the ledger.

```bash
python - <<'PY'
from mcp_server.config import McpConfig
from mcp_server import write_tools as wt
cfg = McpConfig()   # repo root
out = wt.run_smoke_simulation(cfg,
    params={"param_D":1.5,"param_eta":0.3,"param_rho_vac":0.8,"param_a_coupling":2.0},
    seed=42, N_grid=16, T_steps=50)
print(out)
PY
#   -> status SUCCESS, artifact under runs/_smoke/, no ledger row
```

---

## 4. Staged production run via MCP (stage → review → run)

The write path is deliberately two-step (MCP_TOOLS_SPEC §4). Staging is GPU-free
and safe; running requires `confirm=true` and a staged manifest < 30 min old.

```bash
python - <<'PY'
from mcp_server.config import McpConfig
from mcp_server import write_tools as wt
cfg = McpConfig()
staged = wt.stage_simulation_manifest(cfg,
    params={"param_D":1.5,"param_eta":0.3,"param_rho_vac":0.8,"param_a_coupling":2.0},
    hunt_name="IRER_HUNT_001", generation=3, seed=42,
    N_grid=128, T_steps=1200, dt=0.005)
print("REVIEW:", staged["message"])
print("output:", staged["expected_output_path"])
print("warnings:", staged["compatibility_warnings"])
print("errors:", staged["validation_errors"])
# --- review the above, THEN run ---
if staged["staged"] and not staged["validation_errors"]:
    res = wt.run_simulation_manifest(cfg, staged["staged_manifest_path"], confirm=True)
    print(res)
PY
```

The output path is taken from the staged manifest and cannot be redirected.

---

## 5. Launch the read-only MCP server

```bash
# stdio transport; register in your MCP client (Claude Code / Desktop).
python -m mcp_server.server
```

Read-only tools (safe, no GPU): `get_run_status`, `query_ledger`, `read_audit_log`,
`read_provenance`, `list_artifacts`, `inspect_hdf5_schema`, `summarise_generation`,
`audit_data_contract`. Point it at a specific ledger via `ASTE_LEDGER_DB=<path>` or
`QM_ROOT=<repo>`.

---

## 6. Gate checklist before A-coupling

- [ ] Ledger migrated to composite PK (`runs_primary_key` → `{config_hash, seed}`)
- [ ] Determinism self-check: `psi_final` byte-identical across two same-seed runs
- [ ] Refactor fidelity: `psi_final` / `omega_sq_final` / `energy` max\|Δ\| = 0 vs pre-split monolith
- [ ] A-field difference confirmed *expected* (k=0 gate), not a ψ regression
- [ ] Smoke run clean (no sentinel)
- [ ] Read-only MCP tools answering against the live ledger

Only after these are green should A-coupling (vacuum-reference modulation) be
implemented — the k=0 merge gate is already cleared.
