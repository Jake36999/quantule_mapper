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

## Disposition
- **Status:** inventory only. Files remain in `jax_scout/`.
- **Proposed later action (H8, on explicit approval):** move the standalone candidates to `jax_scout/_legacy/`
  (keeping git history), after (a) confirming none has become a dependency, and (b) extracting the two live
  modules' still-used symbols into a core location if they are ever included in a move.
- **Not proposed:** deletion. These document falsified directions and have provenance value.
