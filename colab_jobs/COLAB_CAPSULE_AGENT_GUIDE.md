# Colab Capsule Agent Guide

This guide is for agents preparing a Quantule Mapper run for Google Colab. The normal deliverable is a JSON manifest plus a generated notebook. Do not modify scientific source code unless the user explicitly asks for that work.

## Mental Model

The Colab lane is a manual burst-compute path:

```text
agent writes JSON manifest
    -> notebook_packager_v4.py compiles one .ipynb
    -> user uploads .ipynb to Colab
    -> notebook extracts and verifies files under /content/qm_job
    -> user runs T1, T2, T3, then J1
    -> notebook archives results to Drive
    -> user downloads archive into colab_jobs/results
    -> agent reviews result capsule
```

The notebook is not a second repository and not an orchestration service. It is a sealed job capsule.

## Run Class Must Be Explicit

Before writing a manifest, classify the intended Colab action. Do not infer this from words like "test", "trial", or "handoff"; those words can describe either a procedural UI test or a real scientific reproduction.

Use one of these run classes:

```text
package-only              build and validate a notebook locally; no Colab execution expected
ui-smoke                  exercise buttons, terminals, Drive copy, archive and review path; minutes
short-scientific-pilot    real scientific code with deliberately short duration; usually under 30 minutes
full-reproduction         frozen-baseline reproduction; often 1-2+ hours
full-campaign             multi-row validation campaign; use only with explicit approval
```

If the user asks for a "trial", "UI test", "handoff test", or "workflow test", default to `ui-smoke` or `short-scientific-pilot` unless the queue row or current prompt explicitly says `full-reproduction` and the expected runtime is acknowledged.

The manifest should carry this metadata even though the packager treats it as descriptive:

```json
{
  "run_class": "full-reproduction",
  "estimated_runtime": {
    "a100_minutes_min": 90,
    "a100_minutes_max": 120,
    "basis": "Prior TG-B1S-D 100P Colab reproduction took 6250.04 seconds."
  },
  "operator_confirmation": {
    "required_before_j1": true,
    "message": "J1 is a full TG-B1S-D 100P reproduction and may run for about 1.5-2 hours on A100."
  },
  "stop_rule": "If J1 shows no new log output for 30 minutes, or total runtime exceeds 2.5 hours without checkpoint/archive progress, stop and return the logs for review."
}
```

For full reproductions, put the runtime class in the notebook/operator-facing label too, for example `Run TG-B1S-D 100P Reproduction (~1.5-2h)`. For UI smoke tests, never point `J1` at a known multi-hour scientific entrypoint just to test the dashboard.

## Manifest-First Rule

For normal jobs, write one manifest and avoid custom packager changes. Future agents should only create helper scripts when comparison or analysis logic genuinely does not already exist.

Prefer `rows[].stages[]` when the workflow naturally has more than one step:

```json
{
  "rows": [
    {
      "row_id": "example_row",
      "stages": [
        {
          "stage_id": "simulation",
          "entrypoint": {
            "path": "path/to/run_simulation.py"
          },
          "arguments": [
            "--out",
            "{job_root}/results/example/simulation"
          ],
          "expected_outputs": [
            "results/example/simulation/metrics.csv"
          ]
        },
        {
          "stage_id": "analysis",
          "entrypoint": {
            "path": "path/to/compare_results.py"
          },
          "arguments": [
            "--input",
            "{job_root}/results/example/simulation",
            "--out",
            "{job_root}/results/example"
          ],
          "expected_outputs": [
            "results/example/comparison_summary.json",
            "results/example/completion_status.json"
          ]
        }
      ]
    }
  ]
}
```

Supported template tokens in stage arguments:

```text
{job_root}
{job_id}
{row_id}
{stage_id}
```

## Required Manifest Sections

Use JSON. YAML is not needed for this lane.

Core identity:

```json
{
  "job_name": "Readable Job Name",
  "job_id": "stable_machine_id",
  "job_root": "/content/qm_job"
}
```

Workflow and result policy:

```json
{
  "workflow": {
    "mode": "manual_colab_upload",
    "local_results_dir": "F:\\quantule_mapper\\colab_jobs\\results",
    "intended_use": "Targeted burst-compute run; not a broad hunt."
  },
  "result_policy": {
    "ideal_archive_bytes": 1000000000,
    "hard_archive_bytes": 5000000000
  }
}
```

Files:

```json
{
  "source_files": [
    "jax_scout/simulation.py",
    "colab_jobs/analysis.py"
  ],
  "config_files": [
    "colab_jobs/baselines/example_baseline_manifest.json"
  ],
  "external_inputs": []
}
```

Only include explicit files. Do not package the entire repository unless the dependency closure genuinely requires it.

Precision and environment:

```json
{
  "precision": {
    "jax_enable_x64": true
  },
  "environment_policy": {
    "require_gpu": true,
    "accepted_gpu_names": ["A100"]
  },
  "runtime_environment": {
    "env": {
      "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
      "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90"
    }
  }
}
```

Runtime environment overrides are applied before JAX imports and inherited by child subprocesses. They are useful for Colab-specific memory policy without editing scientific files.

Resource profile:

```json
{
  "resource_profile": {
    "class": "a100_aggressive",
    "push_level": "aggressive",
    "xla_mem_fraction": "0.90",
    "xla_preallocate": "false",
    "autotune_before_j1": true,
    "autotune_target_mb": 2048,
    "autotune_max_seconds": 30,
    "warn_if_gpu_memory_used_fraction_below": 0.15,
    "notes": "Use Colab A100 resources aggressively, but do not change scientific arguments unless declared in the queue."
  }
}
```

The packager resolves this profile, applies `XLA_PYTHON_CLIENT_PREALLOCATE` and `XLA_PYTHON_CLIENT_MEM_FRACTION` before JAX imports, writes `resource_profile_resolved.json`, captures CPU RAM/disk/GPU memory/utilization in preflight, and runs a bounded generic JAX float64 resource probe during `T2`.

This is a runtime/profiling policy. It does not automatically make the scientific simulation larger or faster. If a Colab row should use a longer `T`, larger grid, larger box, more rows, denser cadence, or batched physics path, that must be explicit in the run queue and manifest arguments.

Use these push levels consistently:

```text
conservative  preserve headroom; use for fragile first deployments
balanced      default; 0.90 JAX memory cap, bounded probe
aggressive    use for trusted A100 runs where larger grid/T is requested
max-safe      only when the queue explicitly asks to push Colab hard; still keep archive <=5GB
```

When reviewing low GPU memory use, do not assume failure. A run using 1-2 GB of GPU RAM may still be compute-bound and healthy. Treat low GPU memory as actionable only if GPU utilization is also low, logs show host-transfer stalls, or the row was explicitly meant to use a larger grid/batch.

## Visual Analysis Stage

Colab's extra RAM, GPU memory, and disk are especially useful when a run can save larger HDF5/NPZ field snapshots for post-processing. Prefer this pattern:

```text
simulation stage
    -> writes metrics plus HDF5/NPZ field snapshots under /content/qm_job/results/<run>
visual-analysis stage
    -> runs quantule_viz field-suite over the saved artifact
archive stage
    -> returns metrics, HDF5/NPZ sample artifact, PNGs, GIF, CSV topology table, manifest and logs
```

Use the visual suite entrypoint when the capsule needs topology, vector, density, energy and GIF outputs:

```json
{
  "stage_id": "visual_analysis",
  "command": [
    "python",
    "-m",
    "quantule_viz",
    "field-suite",
    "{job_root}/results/example/field_snapshots.h5",
    "--outdir",
    "{job_root}/results/example/visuals",
    "--overwrite"
  ],
  "expected_outputs": [
    "results/example/visuals/density_slices.png",
    "results/example/visuals/density_evolution.gif",
    "results/example/visuals/vector_current.png",
    "results/example/visuals/topology_active_set.png",
    "results/example/visuals/topology_components.csv",
    "results/example/visuals/energy_timeseries.png",
    "results/example/visuals/visual_suite_manifest.json"
  ]
}
```

Keep solver loops focused on computation. Save sampled fields during the run, then render images/GIFs after the run completes. If larger HDF5 output is scientifically useful, declare that in the run queue's `colab push target` and keep the returned archive under the 5 GB hard cap.

Drive paths:

```json
{
  "drive_results_dir": "/content/drive/MyDrive/QuantuleMapperRuns/example_job",
  "drive_checkpoint_dir": "/content/drive/MyDrive/QuantuleMapperRuns/example_job/checkpoints"
}
```

Control panel:

```json
{
  "control_panel": {
    "schema_version": "qm-colab-controls/1.0",
    "reserved_test_slots": ["T1", "T2", "T3"],
    "tail_lines": 800,
    "actions": [
      {
        "action_id": "T1",
        "kind": "system_test",
        "label": "Run Preflight",
        "handler": "runtime_preflight",
        "output_channel": "terminal_T1",
        "required": true,
        "fixed": true
      },
      {
        "action_id": "T2",
        "kind": "system_test",
        "label": "Run Smoke Test",
        "handler": "numeric_smoke_test",
        "output_channel": "terminal_T2",
        "requires_pass": ["T1"],
        "required": true,
        "fixed": true
      },
      {
        "action_id": "T3",
        "kind": "system_test",
        "label": "Test Persistence",
        "handler": "persistence_test",
        "output_channel": "terminal_T3",
        "requires_pass": ["T1", "T2"],
        "required": true,
        "fixed": true
      },
      {
        "action_id": "J1",
        "kind": "job",
        "label": "Run Example Job",
        "handler": "run_registered_job",
        "row_id": "example_row",
        "output_channel": "terminal_J1",
        "requires_pass": ["T1", "T2", "T3"]
      }
    ]
  }
}
```

Keep `T1`, `T2`, and `T3` reserved for tests. Job buttons start at `J1`.

## Buttons And Terminals

Each control-panel action has three names that future agents should keep aligned:

```text
action_id       button/status identity, such as T1 or J1
handler         trusted notebook function to run
output_channel  human-readable terminal slot, such as terminal_J1
```

The generated notebook renders a Markdown table before the widget UI:

```text
Slot | Label | Handler | Output | Requires
T1   | Run Preflight | runtime_preflight | terminal_T1 | -
T2   | Run Smoke Test | numeric_smoke_test | terminal_T2 | T1
T3   | Test Persistence | persistence_test | terminal_T3 | T1, T2
J1   | Run Job | run_registered_job | terminal_J1 | T1, T2, T3
```

The widget below that table provides the actual buttons and bounded readout terminals. The terminal shown in the notebook is only a rolling tail so the user does not need to scroll through thousands of notebook lines. The full log is persisted separately.

For an action `J1`, the stable locations are:

```text
button:        button_J1
output widget: output_J1
status label:  status_J1
context:       context_J1
full log:      /content/qm_colab/jobs/<job_id>/run/logs/J1.log
status file:   /content/qm_colab/jobs/<job_id>/run/status/J1.json
```

Agents should use `control_panel_ui_bridge.json`, the status files, and the archived logs as evidence. Do not scrape notebook display text as the source of truth. A green button state is not enough; completion evidence comes from the returned result archive, completion status, and integrity hashes.

## Packaging Commands

From the project root:

```powershell
python F:\quantule_mapper\colab_jobs\notebook_packager_v4.py F:\quantule_mapper --manifest F:\quantule_mapper\colab_jobs\example_manifest.json --dry-run
```

Then:

```powershell
python F:\quantule_mapper\colab_jobs\notebook_packager_v4.py F:\quantule_mapper --manifest F:\quantule_mapper\colab_jobs\example_manifest.json -o F:\quantule_mapper\colab_jobs\example_job.ipynb
```

Local validation before handoff:

```powershell
python -m py_compile F:\quantule_mapper\colab_jobs\notebook_packager_v4.py
```

Also parse the manifest JSON and compile generated notebook code cells. Do not claim A100 or Drive validation until the user returns a live result archive.

## Result Review Checklist

After the user downloads the Drive archive into `F:\quantule_mapper\colab_jobs\results`:

1. copy/extract the archive into a unique review folder;
2. verify `result_integrity_manifest.json`;
3. read `completion_ledger.json`;
4. read `completion_status.json`;
5. inspect `comparison_summary.json`;
6. check expected output tables;
7. summarize runtime, environment, archive hash, failed observables, and live-runtime caveats.

Acceptable reproduction statuses are defined by the analysis script, not by the dashboard color alone. For TG-B1S-D the final machine status was:

```text
REPRODUCTION_WITHIN_DECLARED_TOLERANCES
```

## Boundaries

Do not:

- invent new scientific tolerances;
- weaken gates to make a comparison pass;
- silently substitute a different baseline;
- modify simulation equations for Colab convenience;
- treat dashboard success as a physics conclusion;
- run broad hunts through this lane.

Do:

- keep capsules small;
- use explicit allowlists;
- record source hashes;
- keep Drive as persistence and `/content` as execution workspace;
- report live-runtime checks as live checks, not assumptions;
- document successful returned capsules as examples.
