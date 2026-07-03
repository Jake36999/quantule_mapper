# Production Alignment Plan (Track A) — bring the CuPy production stack up to the re-aimed objective

**Why this exists.** The H7 re-aim was validated on the **jax_scout** path because that is the only runtime
reachable from the dev session (no CuPy in the Windows python, the WSL `jax_irer` venv, or the WSL system). That
is a real *split-brain risk*: the objective logic is proven on the mirror, but the **production CuPy stack** —
`worker_daemon.py` → [`worker_cupy.py`](../worker_cupy.py) → `solver/` + `gravity/unified_omega.py` → HDF5
provenance → Hunter orchestration — has not yet been re-aimed end to end. This plan closes that gap **without any
physics change**. (Track B — the *kinetic-operator* formalism question — is a separate, design-only RFC; see the
bottom. It is **not** part of this track and must not touch the frozen Phase C operator.)

## Two facts that frame the work (verified 2026-07-03, at source)
1. **Both engines are FP64.** jax_scout runs `jax_enable_x64=True` / `complex128`; CuPy is `complex128`
   ([`solver/core.py:97`](../solver/core.py)). "JAX is limited to FP32" is not the situation — the re-discovery
   ran at full double precision. CuPy is the *production authority* for **orchestration** reasons (it is the wired
   worker, owns HDF5 provenance/identity, is the Hunter's eval target, runs the 3D `N_grid³` grid), not precision.
   (Perf caveat: GTX 1080 FP64 ≈ 1:32, so FP64 is slow on both — a hardware limit.)
2. **The two engines already share the kinetic term.** CuPy `L_k = -D·k² - η + i·ω₀` (`solver/core.py:97`,
   `complex128`, ω₀=0 at feb) is identical to the jax_scout mirror (`jax_scout/physics.py`). So there is **nothing
   to update in the CuPy kinetic term** for alignment; parity is confirmed at the operator/RHS-code level. Only the
   *bit-level output* run (A1) is still pending, and that needs the CuPy box, not a code change.

## Track A steps

| # | step | state | needs the CuPy box? |
|---|---|---|---|
| A1 | **H4 CuPy bit-parity run** — `python tools/solver_parity_check.py run --backend cupy` then `compare` vs `parity/jax_ref.npz`; record `BIT_PARITY` / `PARITY_WITHIN_TOL` / `PARITY_FAIL` in `SOLVER_PARITY_ARTIFACT.md` | recipe + jax reference staged; **run pending** | **yes** — run-and-record only |
| A2 | **Operator audit** — `worker_cupy.py` / `solver/core.py` / `solver/run.py` / `gravity/unified_omega.py` vs the frozen Phase C operator (`e8d6a78ea`) | done at code level (BASELINE_AUDIT §1.4: shared local cubic-quintic-septic RHS, splash→s/f alias, uncoupled field-of-affect); re-confirmed `L_k` this session | no |
| A3 | **CuPy `stability_metrics` emission** — production run emits the objective's metrics into provenance, no physics change | **DONE this session (code)** — see below; **needs a box run** to produce a real artifact | code done; **box run pending** |
| A4 | **Wire Hunter to consume production metrics** — objective fitness from `prov_data["stability_metrics"]` | `aste_hunter._stability_fitness_from_provenance` already reads `prov_data["stability_metrics"]` (H7.1); remaining = confirm the provenance-assembly path carries it from the worker payload / HDF5 into `prov_data` | no (wiring check) |
| A5 | **Production H7 re-validation** — a small CuPy hunt/replay must: re-find a\*≈×1.15; recover matched controls; **not** promote T12000 short-window artifacts; and keep `css.classify` as the certifier | gated on A1+A3+A4 | **yes** |

### A3 detail — what shipped this session (code-only, no GPU)
- **`solver/stability_metrics.py`** — pure-numpy (no cupy/jax) reduction of a run's **raw** energy trajectory
  (`Σ|ψ|²`, no rho-floor, no dV — the css/objective convention) to the exact metrics
  `tools.stability_objective.stability_score` consumes. It mirrors
  `jax_scout.core_saturation_search.classify`'s er-math **line-for-line** (`er = energy/ic_e`,
  `ic_e = Σ|ψ₀|²`, late-half linear fit → `late_slope_50pct_per1k` / `late_drift`, `floor_ratio = er_min/|er₀|`),
  and fits against **real step numbers** so the every-~10-steps production cadence yields the same per-1000-steps
  slope the objective saw on the mirror.
- **`solver/run.py`** — three **read-only observer** insertions (no physics touched): capture `ic_e_raw = Σ|ψ₀|²`
  before the loop; record `raw_energy = Σ|ψ|²` on the existing telemetry cadence (reusing the already-computed
  `psi_real` — no extra FFT); after the loop, reduce to `stability_metrics` and write it into the HDF5
  (`/stability_metrics`, both the fail and success artifacts) **and** the worker `result_payload`.
- **`tests/test_stability_metrics.py`** — 5 tests, all pass on the dev box: **exact-parity** vs an embedded
  replica of `classify`'s math; cadence-independence (every-step vs every-10-steps); correct feed into
  `stability_score` (flat a\* > slow-decayer > penalized grower; true blow-up hard-rejected); `from_history`
  reads `raw_energy` not the legacy floored/dV `energy`; graceful on empty history.
- **Verification limit:** `py_compile` only for `solver/run.py` (cupy not importable here); the metric math itself
  is fully unit-tested. A real HDF5 artifact with a populated `/stability_metrics` requires a box run (A3→A1/A5).

## Sequencing & guardrails
Recommended order: **A1 (parity) → A3 box run → A4 (consume) → A5 (production re-validation) → only then Track B.**
Hard rules (unchanged): **no solver/geometry/gate/physics change** on this track; the metric emission is
observation-only; `css.classify` v3 stays the certifier; sweep/parity artifacts stay off-repo. Nothing here claims
the production Hunter is re-aimed/validated until A5 passes on the box.

## Track B — Phase D kinetic operator (separate, design-only, DEFERRED)
Changing the kinetic term (imaginary/dispersive `iD∇²`, second-order-in-time, Hamiltonian/advective) is a **Phase
D formalism decision**, not a hardening patch — it changes *what the simulator tests* (the transport sector vs the
validated stability sector). It belongs in a dedicated `docs/PHASE_D_KINETIC_OPERATOR_RFC.md` (design-only, no
implementation, does not touch the Phase C production engine). Deferred by decision this session; noted here so the
two tracks are not conflated.
