# Legacy Script Inventory (H8 — docs-only; NO files moved)

Candidates for *future* archival — the `jax_scout` experiment runners from the **falsified-hypothesis eras**
(A-field/current-coupling, anisotropic-metric/Stage-B routing, Payan, bridge/corridor, transfer diagnostics).
**No files are moved or deleted.** This inventory exists so a later, explicitly-approved archival pass (H8) has a
vetted list — and, critically, so the two **live dependencies** are protected.

## ⚠ MUST STAY — live dependencies (do NOT archive)
| module | why it must stay |
|---|---|
| `jax_scout/afield_current_coupled.py` | provides `L`, `dt`, `multiseed_ic` — **imported by `core_saturation_search.py`** (3 live importers) |
| `jax_scout/transfer_diag.py` | provides `detect_nodes` — imported by `core_saturation_search.py` **and the mobility scripts** `feb_kick_inertia`, `feb_adiabatic_drag` (4 live importers) |

These two are ordinary-named "legacy" files but are load-bearing for the current baseline. Any archival must first
relocate their still-used symbols (`L`, `dt`, `multiseed_ic`, `detect_nodes`) into a core module.

## Archival candidates (standalone; not imported by live code) — leave in place for now
Grouped by the hypothesis they explored, each **falsified / no-support** per the memory record:

**A-field / current coupling** (`gamma_A` branch — `stage-b`/A-field era):
`afield_prototype.py`, `afield_current_tune.py`, `afield_rescue_test.py`, `afield_validate.py`,
`afield_microsweep.py`, `afield_substrate_hunt.py`.

**Anisotropic metric / Stage-B tensor routing** (`NO_SUPPORT`):
`afield_anisotropic.py`, `afield_aniso_deconfound.py`, `afield_aniso_strongbridge.py`, `anisotropy_diag.py`,
`render_aniso_proxy.py`, `afield_routing_gate.py`, `afield_routing_hunt.py`, `afield_routing_validate_pool.py`,
`afield_bridge_hunt.py`, `bridge_hunt.py`.

**Payan / phase-alignment** (`NO_SIGNAL` / `PAYAN_COUPLING_NOT_JUSTIFIED`):
`afield_payan_diagnostic.py`, `afield_payan_balance_test.py`, `payan_chiral_capture.py`,
`payan_hifi_continuation.py`.

**Corridor phase/window probes** (superseded):
`corridor_longwindow.py`, `corridor_phasetest.py`, `corridor_retest.py`.

**Transfer diagnostics** (exploratory; the *live* `transfer_diag.py` is excluded above):
`transfer_deepdive.py`, `transfer_null_control.py`.

## Disposition — EXECUTED 2026-07-03 (H8)
- **24 scripts moved** to `jax_scout/_legacy/` via `git mv` (history preserved); cross-imports to moved siblings
  (`afield_anisotropic`, `afield_current_tune`, `afield_payan_diagnostic`) repointed to `jax_scout._legacy.*`.
- **3 excluded (stayed in `jax_scout/`):** `afield_current_coupled` + `transfer_diag` (live core deps, above) and
  **`afield_prototype`** — imported by `tests/test_afield_prototype.py` (`from jax_scout import afield_prototype`),
  so it is a test dependency, not fully dead.
- **Verified:** live core (`core_saturation_search`) and `afield_prototype` still import on WSL; all 24 moved
  files compile; no live-code or test importer references a moved file.
- **Not done:** deletion (these document falsified directions and keep provenance value). Any later resurrection
  of a `_legacy` script needs its internal imports checked (siblings are now under `jax_scout._legacy.*`).
