---
name: no-cupy-dev-box
description: This dev box (F:\quantule_mapper) has no CuPy/GPU; solver integration must be verified elsewhere
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e9a0bad-bd6b-44f4-aea3-dc2feb389751
---

The **Windows** side (`F:\quantule_mapper`, the python on PATH) has numpy/h5py/pydantic/sqlite3 but **no CuPy and no jax** — anything importing `cupy` (`solver/core.py`, `solver/run.py`) or `jax` can only be `py_compile`-checked from Windows, never run there.

**BUT the JAX scout DOES run, on GPU, via WSL2.** There is a venv `~/jax_irer` in WSL Ubuntu (`wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python ...'`) with `jax 0.10.2` reporting `CudaDevice(id=0)` — so all `jax_scout/` scripts and `tests/test_*` that import jax run FP64 on GPU there (verified 2026-06-21 running the Stage B anisotropic-metric panels). The proxy/Stage-B results were produced this way. The production **CuPy** solver is the part with no runtime here (CuPy not installed); its γ_A=0 byte-identical regression still needs a CuPy/GPU box.

Strategy that works: keep contract/identity/ledger/provenance logic in pure-Python modules (e.g. `orchestrator/run_identity.py`) so they're fully unit-testable from Windows; run JAX-scout numerics via the WSL `jax_irer` venv; prove cupy-array math via numpy mirrors in tests.

The project moved drive `E:`→`F:` around 2026-06-18 (prior session ended with "working directory no longer exists: E:\quantule_mapper"); memories under the old path did not carry over. See [[dc-v1-hardening-state]].
