# Phase C And Visuals Runbook

Status: 2026-06-23 00:08 +01:00 local inspection. This runbook is for Windows PowerShell from
`F:\quantule_mapper` unless a command explicitly says WSL.

## Scientific Framing

Use cautious labels. The current earned object name is:

`LONG_TIME_STABLE_4_NODE_ATTRACTOR`

Meaning: a T=6000-confirmed, bounded, non-rotating, repulsively-stabilized 4-node attractor under
tested conditions. Do not call it infinite-stable, a thermodynamic ground state, a topological
invariant, matter proven, or a molecule unless the phrase is explicitly marked as analogy.

The current Phase C question is empirical:

`P(TRUE_SATURATED_BOUND_STATE | K, params)`

where `K` is the number of initial Gaussian blobs. The key readout is:

`sat_node_counts_by_IC_blobs`

This tests whether final node count tracks the initial condition or whether the field tends toward
4-5 nodes across K in this sampled regime.

## Environment Notes

- The Phase C search is implemented in `jax_scout/core_saturation_search.py`.
- The Phase C search now accepts `--ic-seed` so structured discovery runs can vary IC seed without
  changing the parameter RNG seed or classifier contract.
- The known working launch path from the Claude run was WSL2 plus the JAX venv:
  `source ~/jax_irer/bin/activate`.
- Windows `.venv` and `venv` were checked in this Codex session and did not have `jax` installed.
- On this Codex session, `wsl.exe --list --verbose` reported no installed distributions, so the
  current 6-hour hunt could not be relaunched faithfully from here.
- GPU inspection uses `nvidia-smi`; low GPU utilization alone is not proof of failure.

## Repo Status

```powershell
git status --short --branch
git log --oneline -8
```

Recent commits relevant to this work:

```text
3ce05e5cf Phase C pilot: feb56dc7 regime is a real saturated-bound-state basin (generically 4-5 node)
d1039fee4 feb56dc7 characterized: T=6000 stable non-rotating 4-node attractor (repulsive, not bonded)
2cc8ad46e Decisive long-time test: feb56dc7 SATURATES (true steady multi-node state), rotation is transient
4559102f9 Single-core characterization: long-time test REFUTES steady solitons (bare S-NCGL)
```

## Check Active Python, WSL, And GPU Processes

```powershell
Get-Process python,python3,wsl,wslhost,bash -ErrorAction SilentlyContinue |
  Select-Object Id,ProcessName,CPU,StartTime,Path
```

```powershell
nvidia-smi
```

If WSL is available:

```powershell
wsl.exe --list --verbose
wsl.exe -d Ubuntu -- bash -lc "ps -eo pid,ppid,etime,pcpu,pmem,args | grep -E 'core_saturation_search|python|jax' | grep -v grep"
```

## Locate Latest Phase C Output Directory

```powershell
Get-ChildItem -Directory sweep_runs -Filter 'CORE_SAT_*' |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10 FullName,LastWriteTime
```

Latest known directories from the handoff:

```text
sweep_runs/CORE_SAT_PILOT_20260622_190340
sweep_runs/CORE_SAT_HUNT_20260622_211628
```

## Inspect Incremental CSV, Summary JSON, And Logs

Set a run directory:

```powershell
$run = 'sweep_runs\CORE_SAT_HUNT_20260622_211628'
```

File modification times:

```powershell
Get-ChildItem $run -Force | Format-List Name,Length,LastWriteTime,FullName
```

CSV row count:

```powershell
if (Test-Path "$run\all_evals.csv") {
  (Import-Csv "$run\all_evals.csv" | Measure-Object).Count
}
```

CSV tail:

```powershell
Get-Content "$run\all_evals.csv" -Tail 5
```

Summary:

```powershell
Get-Content -Raw "$run\summary.json"
```

Run log:

```powershell
Get-Content -Tail 40 runtime_logs\core_sat_hunt.log
```

Claude task output from the original launch, if still present:

```powershell
Get-Content -Raw 'C:\Users\jakem\AppData\Local\Temp\claude\F--quantule-mapper\5cf36465-e51a-450c-b166-5b68adb5100c\tasks\bqrunsq1s.output'
```

## Current Hunt Status From Codex Inspection

Verdict: `RUN_CRASHED`

Evidence:

- `runtime_logs/core_sat_hunt.log` contains only the launch banner.
- `sweep_runs/CORE_SAT_HUNT_20260622_211628/all_evals.csv` is zero bytes.
- No `summary.json` exists in that run directory.
- `Get-Process` showed no active Python/JAX worker.
- `nvidia-smi` showed no compute Python process, only desktop/Codex/Claude graphics users.
- `wsl.exe --list --verbose` reported no installed WSL distributions in this Codex environment.
- The expected launch command required `wsl.exe -d Ubuntu`.

Do not analyze `CORE_SAT_HUNT_20260622_211628` as data; it contains no completed evaluations.

## Short Smoke Test

Use this only after WSL/JAX is available. It verifies the classifier and includes the feb56dc7 reference
when the first batch uses `K=6`.

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && timeout 900 python jax_scout/core_saturation_search.py --calibrate --batch 8'
```

Seeded smoke variant for structured discovery:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_search.py --hours 0.25 --N 48 --T 4000 --batch 8 --max-batches 1 --ic-counts "1,2,3,4,6" --ic-norm total_mass_fixed --target-initial-mass 1200 --ic-seed 20260621'
```

Expected output shape:

```text
=== PHASE C PILOT DONE: 8 evals ...
counts={...}
-> /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_CALIB_...
```

Then inspect:

```powershell
Get-ChildItem -Directory sweep_runs -Filter 'CORE_SAT_CALIB_*' |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 FullName
```

## Launch 2-Hour Pilot

This is the command Claude used for the completed pilot, with default `K=6`:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_search.py --hours 2 --batch 64 > runtime_logs/core_sat_pilot.log 2>&1; echo "PILOT EXIT $?" >> runtime_logs/core_sat_pilot.log'
```

Completed pilot result:

```text
sweep_runs/CORE_SAT_PILOT_20260622_190340
N=48, T=4000, 256 evals
TRUE_SATURATED_BOUND_STATE: 33
NEAR_SATURATED_BOUND_STATE: 12
TRANSIENT_GROWER_REJECT: 10
SPIN_DOWN_REJECT: 76
LATE_BLOWUP_REJECT: 125
TRUE final node counts: 27 four-node, 6 five-node
```

Caveat: the pilot used fixed `K=6`, so the absence of 1/2/3-node saturators is not decisive.

## Launch 6-Hour K-Varied Hunt

This is the intended relaunch command once WSL/JAX is available:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_search.py --hours 6 --batch 64 --ic-counts "1,2,3,4,6" > runtime_logs/core_sat_hunt.log 2>&1; echo "HUNT EXIT $?" >> runtime_logs/core_sat_hunt.log'
```

Structured multi-seed discovery example:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_search.py --hours 1.25 --N 48 --T 4000 --batch 8 --max-batches 5 --ic-counts "1,2,3,4,6" --ic-norm total_mass_fixed --target-initial-mass 1200 --ic-seed 20260621'
```

Expected first lines in `runtime_logs/core_sat_hunt.log`:

```text
=== PHASE C SATURATION SEARCH N=48 T=4000 | regime eta[-0.02, 0.15] a[0.2, 0.5] | IC blob counts [1, 2, 3, 4, 6] ===
[batch 1 K=1 ...]
```

Expected output directory pattern:

```text
sweep_runs/CORE_SAT_HUNT_YYYYMMDD_HHMMSS
```

The script writes:

```text
all_evals.csv
summary.json
```

`summary.json` includes:

```json
"sat_node_counts_by_IC_blobs": {}
```

## Launch A Small Total-Mass Threshold Pilot

This keeps scope narrow and balanced across `K=1` and `K=6` without launching a broad hunt.

Chosen target masses:

```text
291.882452
500
800
1200
1600
2050.293702
```

Each mass is run as two batches:

- batch 1: `K=1`, `8` configs
- batch 2: `K=6`, `8` configs

That gives:

```text
6 masses x 2 K values x 8 samples = 96 configs
```

One fixed-mass run command:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_search.py --hours 1 --N 48 --T 4000 --batch 8 --ic-counts "1,6" --max-batches 2 --ic-norm total_mass_fixed --target-initial-mass 800'
```

Balanced six-mass PowerShell loop:

```powershell
$masses = 291.882452,500,800,1200,1600,2050.293702
foreach ($mass in $masses) {
  wsl.exe -d Ubuntu -- bash -lc "source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_search.py --hours 1 --N 48 --T 4000 --batch 8 --ic-counts '1,6' --max-batches 2 --ic-norm total_mass_fixed --target-initial-mass $mass"
}
```

## Run The Threshold Diagnostic Pass

Replay-backed threshold summary for the six `K in {1,6}` mass-grid runs:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_collapse_diag.py --mode threshold-pilot --n-snap 24 --outdir /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_THRESHOLD_DIAG_20260623_180519'
```

Outputs:

- `threshold_diagnostics.csv`
- `threshold_summary.json`

## Run The Threshold Branch Robustness Layer

Dry-run manifest only:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_robustness.py --dry-run --outdir /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_<timestamp>'
```

Serial robustness batch:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_robustness.py --run --outdir /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_<timestamp> --runtime-log /mnt/f/quantule_mapper/runtime_logs/phase_c_threshold_robustness_<timestamp>.log'
```

Post-run analysis:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_robustness.py --analyze --outdir /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_<timestamp>'
```

Outputs:

- `robustness_manifest.csv`
- `robustness_manifest.json`
- `run_status.json`
- `threshold_branch_robustness.csv`
- `threshold_branch_robustness_summary.json`
- `docs/PHASE_C_THRESHOLD_BRANCH_ROBUSTNESS.md`

Notes:

- Cross-resolution rows use explicit `--target-initial-mass-override`.
- The robustness driver reuses `core_saturation_replay.py`; it does not duplicate the solver or classifier path.
- Existing complete row directories are skipped on rerun.

## Build The Four-Case Trace Comparison Pack

Step 1: replay and save traces in WSL:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_collapse_diag.py --mode trace-comparison --n-snap 60 --failure-n-snap 120 --outdir /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450'
```

Step 2: render PNG panels from the saved trace JSON on the host Python:

```powershell
python jax_scout/core_saturation_collapse_diag.py --mode render-trace-plots --outdir sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450
```

Outputs:

- `mass_threshold_trace_comparison.csv`
- per-candidate `trace_panel.png`
- `trace_overlay.png`

Expected runtime:

- about `8` to `12` minutes per mass after startup
- about `50` to `90` minutes total for all six masses

Do not commit the generated `sweep_runs/CORE_SAT_HUNT_*` directories unless explicitly requested.

## Stop A Stalled Run Safely

Prefer interrupting the terminal that launched the run with `Ctrl+C`.

If the process is detached and WSL is available:

```powershell
wsl.exe -d Ubuntu -- bash -lc "ps -eo pid,etime,args | grep core_saturation_search | grep -v grep"
```

Then send SIGINT to the specific PID:

```powershell
wsl.exe -d Ubuntu -- bash -lc "kill -INT <PID>"
```

If it does not exit after a reasonable wait:

```powershell
wsl.exe -d Ubuntu -- bash -lc "kill <PID>"
```

Avoid deleting partial output. Preserve the run directory and relaunch to a new timestamped directory.

## Resume Or Relaunch

`core_saturation_search.py` does not currently implement resume-from-partial CSV. Relaunching creates a
fresh timestamped directory and preserves prior partial output.

Before relaunch:

```powershell
Get-ChildItem -Directory sweep_runs -Filter 'CORE_SAT_HUNT_*' |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 FullName,LastWriteTime
```

Then run the 6-hour command above.

## Replay A Saved Phase C Row At Higher Fidelity

Verified helper:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --csv /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_004605/all_evals.csv --idx 623 --N 96 --T 6000 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_VALIDATE_FEB_CONTROL/idx_623 --overwrite'
```

Reference control:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --ref feb56dc7 --N 96 --T 6000 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_VALIDATE_FEB_CONTROL/ref_feb56dc7 --overwrite'
```

Saved replay artifacts:

```text
summary.json
probe_data.npz
```

`summary.json` records:

- source row or reference name
- `K`
- `N`, `T`, `L`, `dt`
- class and late-window metrics
- IC seed
- classifier thresholds
- current git commit
- exact replay command

## Run The Post-Run Collapse/Runaway Diagnostic

This is a replay-backed analysis pass only. It does not modify the PDE, the solver path, or
`gravity/unified_omega.py`.

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_collapse_diag.py --outdir /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000'
```

Expected outputs:

```text
sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000/collapse_runaway_diagnostics.csv
sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000/<candidate>/diagnostic_summary.json
sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000/<candidate>/frames.npz
```

Use cautious diagnostic labels only:

```text
COLLAPSE_LIKE_RUNAWAY
FRAGMENTING_BLOWUP
DELOCALIZED_GROWTH
HIGH_K_NUMERICAL_ARTIFACT_SUSPECT
SATURATED_BOUND_STATE
SPIN_DOWN_DECAY
INCONCLUSIVE_FAILURE_TRACE
```

## Phase A/B feb56dc7 Characterization Renderer

Data directory:

```text
sweep_runs/SUBSTRATE_HUNT_20260621_161557/feb56dc7_bound_state
```

Renderer:

```powershell
python plugins\visualizers\feb_bound_state_render.py
```

Expected outputs in the data directory:

```text
relaxation.png
profiles.png
tracks_slices.png
perturbation.png
```

Supporting Phase A/B scripts:

```text
jax_scout/feb_bound_state.py
jax_scout/feb_bond_test.py
plugins/visualizers/feb_bound_state_render.py
```

## Generate Standard Figures From Phase C Results

New reusable CLI:

```powershell
python -m quantule_viz phase-c sweep_runs\CORE_SAT_PILOT_20260622_190340
```

Expected outputs:

```text
summary_panel.png
class_histogram.png
node_counts_by_K.png
saturation_slope_scatter.png
best_candidates_table.csv
```

For a frame bundle such as `frames.npz`:

```powershell
python -m quantule_viz frames sweep_runs\SUBSTRATE_HUNT_20260621_161557\feb56dc7_bound_state\frames.npz --outdir sweep_runs\SUBSTRATE_HUNT_20260621_161557\feb56dc7_bound_state
```

Expected outputs:

```text
frame_density_slices.png
frame_vector_preview.png
```

## Separation Of Responsibilities

- Simulation computes dynamics only.
- Validation classifies outcomes.
- Optimization/search proposes and evaluates configs.
- Visualization reads saved outputs and renders summaries.
- Documentation reports evidence, inference, caveats, and next actions.

The visualization package must not change parameters, run simulations, or assign new physics classes.
