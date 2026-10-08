---
tags: [record, infra, basins, search]
date: 2026-10-08
branch: Branch - Stability - Index
status: running
---

# Quality-diversity explorer: searching for *different* end states

The old search stack chased one score. This explorer looks for **different kinds of end state** and
spends its samples where behaviour changes. It is built to run **in the background for days**,
stopping and resuming without losing work, so the GPU produces data whenever the PC is idle.

> [!info] Sources
> - `tools/qd_explore.py`: driver, boundary sampler, export
> - `specs/qd/wide-net-v1.qd.json`: the first wide-net configuration
> - `tools/qd_background.ps1` + `tools/qd_run_wsl.sh`: background launch
> - `tests/test_qd_explore.py`
> - `jax_scout/registry.py`: compiled batch functions are now cached across batches (CL-015)
> - Dependencies (WSL `~/jax_irer`): `ribs` 0.12 (pyribs), `scikit-learn` 1.9

## 1. Review of the old search scripts (2026-10-08)
The October audit ([[SEARCH_STACK_AUDIT_2026-10]]) already covers the Hunter, `predator_sweep.py` and
`fss_scaling_analyzer.py`. This review adds the orchestrator and decides what to keep from each.

| script | what it does | verdict |
|---|---|---|
| `aste_hunter.py`, prime mode | NSGA-II-style GA on a hand-weighted log-prime score | **Retired** (audit §1). The crossover collapses diversity, `dominates()` is not Pareto, and the null test is a no-op. |
| `aste_hunter.py`, stability mode | elitist GA on 3 axes, ±15% around FEB | **Kept for refinement only.** It re-finds a\* inside a box built around a\*. |
| `adaptive_hunt_orchestrator.bak` | per-generation driver for the Hunter | **Retired.** It survives only as a `.bak`, and nothing calls it. See below. |
| `predator_sweep.py` | basin check on Hunter elites | **Retired.** It drops `param_a`, so every child ran at a=0 (audit §3). |
| `fss_scaling_analyzer.py` | "scaling probe" feeding the orchestrator | **Retired.** Its design matrix is rank-deficient, so it can never pass its own confidence gate (audit §3). |
| `backlog_orchestrator.py` | linear runner for a fixed backlog (not a search) | Out of scope. Superseded by `tools/run_spec.py` sweeps. |

**Orchestrator findings:**
- **It averages over seeds.** `evaluate_robustness` scores a configuration by the *mean* SSE over 3
  seeds. Two seeds in different basins become one blended number. Multistability is exactly what this
  hides.
- **Failure and bad physics look the same.** A missing seed file or artifact scores 999, the same as a
  terrible run.
- **It deletes evidence.** The "JSON micro-purge" removes per-seed provenance and parameter files after
  scoring.
- **Grid invariance used the retired objective.** `fss_grid_invariance_harness` (N=32/64) gated on
  log-prime SSE.
- **One of its candidate sources never fired.** Scaling-probe ingestion waits on
  `fss_scaling_analyzer`, which can never produce a candidate.
- **Worth keeping**, and reused here: generation-level checkpointing, a stop file checked between
  generations, and keeping heavy artifacts off the hot path.

**The search box never contained the answer.** The Hunter's prime-mode default box was D∈[0.01,2],
ρ_vac∈[0,0.5], a_coupling∈[−2,2] and splash_fraction∈[0,1]. The known FEB/a\* point (D=2.73, ρ_vac=1.19,
a_coupling=2.31, f=−0.49) lies **outside that box on 4 of its 7 axes**. a\* came from the later
substrate tuning, not from the box search, which supports the concern that we are sitting in "an
optimised zone of an optimised zone".

## 2. What the explorer does
Every generation runs a batch of 20 configurations (one vmapped call):

- **CMA-MAE emitters (pyribs), 15 of the 20.** They fill an *archive* of behaviours.
  - The archive is a 16 × 24 × 12 grid over end-state measures: localisation (log contrast), node
    count, and log energy ratio.
  - The emitters are rewarded for **improving the archive**, and especially for reaching empty cells.
  - The objective is *stationarity*: a standing end state scores 0, a drifting one scores below zero.
    It only ranks states **within** a cell, so the search cannot collapse onto one optimum.
- **Boundary sampler (active learning), 5 of the 20.**
  - An extra-trees classifier maps parameters to a coarse regime: `decayed`, `blowup`, or
    `{localised|dispersed}/{standing|breathing|growing|decaying}`.
  - It is refitted on every evaluation so far.
  - New points go where the classifier is least certain, with a distance bonus toward empty regions.
- **A fresh IC seed for every evaluation.** One parameter point can therefore land in two cells, and
  multistability shows up instead of being averaged away.
- **Seeding.** The first `seed_evals` (200) are a scrambled Sobol spread. They start with the anchors
  (a\* and FEB), which must reappear in the archive as a sanity check.
- **Stopping.** Progress is tracked as filled cells, plus a **Chao1** estimate of reachable cells: the
  unseen-species estimator from ecology. When new cells stop appearing and Chao1 stops rising, this
  resolution is exhausted.

**The first box** (`wide-net-v1`) has 8 dimensions:
- log D ∈ [0.1, 10]
- η ∈ [−0.2, 0.4]
- ρ_vac ∈ [0.2, 3]
- a ∈ [−0.5, 1.5]
- a_coupling ∈ [−1, 4]
- s, f ∈ [−1, 1]
- blob count K ∈ 1..10

ω0 is held at 0, because a uniform frequency is probably removable by a gauge rotation (not yet
verified).

**Protocol: N=96, T=72, fp32**, with generations of 10 stepped in chunks of 5. N=96 rather than N=48 is
forced by the anchors (§4).

## 3. Running it in the background
```
powershell -File tools/qd_background.ps1 specs/qd/wide-net-v1.qd.json [-Hours 20]
python tools/qd_explore.py status sweep_runs/QD_WIDE_NET_V1
python tools/qd_explore.py pause|resume|stop sweep_runs/QD_WIDE_NET_V1
```

**How it runs:**
- The launcher starts a hidden `wsl.exe` running the driver in the foreground at `nice 10`. A live
  `wsl.exe` keeps the WSL VM up; a `nohup` child of a closed shell did not reliably survive here.
- GPU memory is allocated as needed rather than preallocated: about 2 GB at N=96 in chunks of 5,
  against 4.2 GB in chunks of 10. The PC stays usable.
- **`pause` and `stop` are files** checked between generations. They take effect when the generation
  in flight finishes (minutes). `SIGTERM` does the same.
- **Crash-safe.** `evals.jsonl` is the only state, one fsync'd line per evaluation. A restart with
  the same config rebuilds the archive from it, and the CMA emitters re-adapt within a few
  generations. A crash or reboot loses at most one generation.
- **Changing the configuration needs a new id.** A different box, measures or protocol under the same
  run directory is refused, so archives are never mixed.

**What it stores:**
- **Per evaluation:** parameters, IC seed, regime, measures, descriptors, er statistics, and its cell.
  That is about 1 KB, or roughly 3 MB a day.
- **Per archive cell:** the current elite's downsampled |ψ|² (24³ float16, about 28 KB). The worst case
  is about 130 MB for a full archive.
- Both live under `sweep_runs/` (gitignored).

**Export:** turns any evaluation, or the best evaluation in every filled cell, into a normal spec in
`specs/proposed/`. Run that with `tools/run_spec.py`, with `--fp64` to verify. It reproduces the
evaluation exactly (tested to 1e-8), and from there the viewer can re-run it with visuals.

## 4. Measured
- **CPU test** (N=16, real end to end):
  - It runs, resumes with continuous numbering, and refuses a changed box.
  - Exported evaluations reproduce through `run_spec` to 1e-8.
  - The elite export rebuilds one point per filled cell.
- **GPU smoke test at N=48** (T=72, batch 20, two generations through the background launcher):
  - It ran hidden, and `STOP` ended it cleanly after the generation in flight.
  - The Sobol seed generation filled 11 cells. **The first CMA-MAE + boundary generation filled 17
    new cells out of 20**, so the emitters do reach new behaviour.
  - Throughput was 108 evaluations per hour. About half of each wide-box batch diverged, and those
    members wasted their batch slot until it ended. **Fixed:** diverged members now leave the batch
    at the sample where they die, and keep their `t_death`.
- **Finding: N=48 does not reproduce a\*.**
  - Run to T=360 in fp32, the a\* anchor's er keeps growing: 3.4 at T=72, 7.3 at T=360, still
    rising. The field never localises (contrast 8, 0 nodes).
  - At N=96, which is validated in fp64, the same point is a bounded state with 4–6 nodes.
  - fp32 is not the cause: fp32 and fp64 agree to 0.1% at N=48 ([[BATCHED_RUNS]]).
  - **So N=48, like N=32, is under-resolved for a\*-class localised structures on L=10.** The anchor
    check caught this before a week of screening was spent at the wrong resolution.
- **N=96 cost check** (10 evaluations, T=7.2):
  - Both anchors are already `localised` at T=7.2.
  - Diverging members die at the first sample, so they are almost free.
  - Peak GPU memory was 4.2 GB stepping 10 together. Chunks of 5 cost no throughput, because
    batching gives no speed-up at N≥48, and halve the memory.
  - Expect about **190 s per surviving evaluation at T=72, or roughly 500–650 evaluations a day**.
    The first 3,000-evaluation wide net (a ≥0.1%-volume detection floor) takes about 5 days.
- Each evaluation row also stores a 60-point er(t) series, so regimes can be relabelled later (other
  thresholds, transient detection) without re-running anything.

## 5. Status and next steps
- [ ] Start `wide-net-v1` in the background and leave it for days.
- [ ] After the seed phase, check that the a\* anchor sits in a `localised/*` cell. At T=72 it is
  still transient, so expect `localised/growing`.
- [ ] Possible upgrade: a two-tier explorer. Explore at N=48 for throughput, and automatically re-run
  every new localised cell at N=96, so that resolution artifacts are labelled rather than avoided.
- [ ] Export new `localised` cells and verify them in fp64 at N=96 before reading anything into them.
- [ ] Follow-ups:
  - verify the gauge argument for ω0;
  - remove the redundant scalings from the box;
  - add a homotopy axis across substrates (temporal → kinetic → dual);
  - run continuation and deflation from confirmed cells ([[BASIN_MAPPING]] F3).

## Issues raised

| # | issue | status |
|---|---|---|
| 1 | The adaptive orchestrator averaged seeds, scored failures as bad physics and deleted provenance | NOTED — retired; the explorer keeps every evaluation and a fresh seed per evaluation |
| 2 | The prime-mode box did not contain FEB/a\* on 4 of 7 axes | NOTED — the wide-net box contains both |
| 3 | `provenance.stamp()` starts JAX, so a driver that stamps before importing the registry preallocated 75% of the GPU | RESOLVED for the explorer (it disables preallocation first) |
| 4 | `BatchedETDRK4` recompiled for every new batch | RESOLVED — compiled functions cached by key (CL-015) |
| 5 | N=48 (like N=32) does not reproduce a\*: er grows to 7, no localisation | NOTED — wide net runs at N=96 (about 5× the cost) |
| 6 | About half of a wide-box batch diverges and used to hold its batch slot | RESOLVED — dead members leave the batch |

## Associated docs
- [[SEARCH_STACK_AUDIT_2026-10]] · [[BASIN_MAPPING]] · [[SCREEN_AND_VERIFY]] · [[BATCHED_RUNS]] ·
  [[SOLVER_AND_RUNTIME_CHANGELOG]]

## Branches
- [[Branch - Stability - Index]]
