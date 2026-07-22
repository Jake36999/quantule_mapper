# CL TG-B2 Definitive Force IPYNB Provenance Report

## Capsule Identity

- Queue row: `CL_TG_B2_DEFINITIVE_FORCE`
- Job name: `CL TG-B2 Definitive Force - Dynamical Body-Force Observable`
- Job id: `cl_tg_b2_definitive_force`
- Row id: `tg_b2_definitive_force_seps_3_4`
- Notebook: `F:\quantule_mapper\colab_jobs\cl_tg_b2_definitive_force.ipynb`
- Manifest: `F:\quantule_mapper\colab_jobs\cl_tg_b2_definitive_force_manifest.json`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cl_tg_b2_definitive_force_operator_instructions.md`
- Local validation report: `F:\quantule_mapper\colab_jobs\cl_tg_b2_definitive_force_local_validation_report.md`
- Generated for manual Google Colab upload and A100 execution.

## Notebook Hash

| Artifact | Size bytes | SHA-256 |
|---|---:|---|
| `colab_jobs/cl_tg_b2_definitive_force.ipynb` | 219303 | `b92dbc559c4af103e5f4c587441d68fd54f9cfdb175dfb5ab3a68ab279f4521c` |
| `colab_jobs/cl_tg_b2_definitive_force_manifest.json` | 6466 | `27496ac963823bb7f3e41f517caba8c91bece3ebd56431c858da7f1649879fb6` |

## Bundled Files

The notebook embeds exactly the following source allowlist. No full repository directory was bundled.

| Bundled path | Size bytes | SHA-256 | Role |
|---|---:|---|---|
| `jax_scout/__init__.py` | 94 | `ed23112aa9afa87ff78171fa36b063a296cb786354053562117b7fb2e605ebc5` | Package marker/import support |
| `jax_scout/phase_d_c3_wave.py` | 16850 | `aacc2c709c1eefe98b9fed2a8ef9aa9b2ca2c439d8d982717a96aeb8bee2ccff` | KG grid/Q-ball support imported by B1S/B2 scripts |
| `jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py` | 39939 | `85215ef2f634820aae806dcd799d4fcb78ececab3224b994f696fb817bdeebf1` | Frozen TG-B1S state-load feedback helpers and single-node normalization |
| `jax_scout/gravity_TG_B2_two_node_awell.py` | 12975 | `6f6b9f0fad2bf5e27378c3efc3620ee6d79a577978a9c0a7533c0648e38012be` | Two-node A-well dynamics helper used by definitive-force script |
| `jax_scout/gravity_TG_B2_definitive_force.py` | 12784 | `b000cc031d9fc7f088bbafac60da02cd3b3c9ee5609d87bde6619d53178ec19c` | Main simulation entrypoint |

## What Test Is Being Run

The packaged `J1` action runs:

```text
python jax_scout/gravity_TG_B2_definitive_force.py --seps 3.0,4.0 --T 180 --out /content/qm_job/results/cl_tg_b2_definitive
```

This is a full-fidelity TG-B2 dynamical body-force campaign over separations `3.0` and `4.0`. For each separation it runs:

- `off`: feedback off, built-in `A=1` null arm;
- `well`: A-well branch, theory-faithful `a_sign=+1`;
- `hill`: A-hill branch, sign-control `a_sign=-1`.

The primary observable is:

```text
<F_R> = time average of -c^2 int_{x>0} d_x A |grad phi|^2 dV
```

over the settled window after discarding the first 40 percent of the trajectory.

Sign convention:

```text
F_R < 0 means attraction.
```

## Runtime And Environment Contract

- Run class: `full-campaign`
- Expected A100 runtime: `150-210 minutes`
- Hard operator stop rule: stop if no `J1` log output appears for 30 minutes, or if A100 runtime exceeds 5 hours without `RUN_COMPLETE.json` or archive progress.
- Script-level stop rule: each arm stops if node separation drops below `1.2` or becomes invalid.
- Required backend: JAX GPU
- Accepted GPU: `A100`
- Precision: `jax_enable_x64=true`
- Runtime memory policy:

```text
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
```

This memory policy is Colab resource plumbing only. It does not alter TG-B2 or frozen TG-B1S scientific source or gates.

## Expected Outputs

The capsule fails closed if these declared outputs are missing:

- `results/cl_tg_b2_definitive/config.json`
- `results/cl_tg_b2_definitive/scalars_sep3.00_off.csv`
- `results/cl_tg_b2_definitive/scalars_sep3.00_well.csv`
- `results/cl_tg_b2_definitive/scalars_sep3.00_hill.csv`
- `results/cl_tg_b2_definitive/ROW_sep3.00_STARTED.json`
- `results/cl_tg_b2_definitive/ROW_sep3.00_COMPLETE.json`
- `results/cl_tg_b2_definitive/scalars_sep4.00_off.csv`
- `results/cl_tg_b2_definitive/scalars_sep4.00_well.csv`
- `results/cl_tg_b2_definitive/scalars_sep4.00_hill.csv`
- `results/cl_tg_b2_definitive/ROW_sep4.00_STARTED.json`
- `results/cl_tg_b2_definitive/ROW_sep4.00_COMPLETE.json`
- `results/cl_tg_b2_definitive/summary.json`
- `results/cl_tg_b2_definitive/RUN_COMPLETE.json`

The final Colab archive should also contain capsule-level provenance and integrity files such as `completion_ledger.json`, `result_integrity_manifest.json`, control-panel status files, and full `J1` logs.

## Reason

The queue request exists because the prior TG-B2 overnight dynamical proxies disagreed:

- momentum-slope proxy suggested repulsion;
- separation-difference proxy suggested attraction.

Both proxy paths were considered vulnerable to contamination from violent breathing/transient reshaping. This capsule therefore runs the cleaner dynamical Gravity-D body-force observable on the live fields, with explicit off-null and well/hill antisymmetry controls.

The scientific question is narrow:

```text
Does the A-well branch produce a consistent attractive, repulsive, null, or inconsistent dynamical body-force sign across separations 3.0 and 4.0?
```

Reviewers must not treat the script's auto-label as sufficient. The returned archive should be reviewed by directly checking:

- `<F_R_well>` sign;
- `<F_R_hill>` opposite-sign control;
- `<F_R_off>` null behavior;
- node distinctness gate;
- charge conservation gate;
- cross-separation consistency.

## Results

Status: `RETURNED_FIRST_PASS_REVIEWED`

First-pass review:

```text
F:\quantule_mapper\colab_jobs\Colab_runs\cl_tg_b2_definitive_force_20260716_113033\FIRST_PASS_REVIEW_20260716.md
```

Archive:

- filename: `cl_tg_b2_definitive_force_20260716_113033.tar.gz`
- size: `199447 bytes`
- SHA-256: `f678d77ec5e02382005148ccab5a84d5fae4df3c6dc8a71482f433514ab4180e`
- sidecar hash check: passed
- result-integrity manifest check: passed, `0` mismatches

Environment:

- GPU: `NVIDIA A100-SXM4-40GB`
- JAX/JAXLIB: `0.7.2` / `0.7.2`
- Python: `3.12.13`
- x64 enabled: yes
- float64 kernel dtype: `float64`
- runtime memory policy applied: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`

Run status:

- capsule status: `RUN_COMPLETE_CAPSULE`
- completed row: `tg_b2_definitive_force_seps_3_4`
- failed rows: none
- script return code: `0`
- runtime: `2718.65 s` (`0.755 h`, about 45.3 minutes)
- script verdict: `TG_B2_DEFINITIVE_AWELL_ATTRACTION_CONFIRMED`

First-pass observable summary:

| sep | n | t_end | <F_R_well> | <F_R_hill> | <F_R_off> | well/hill antisym rel | charge drift max | sep_min | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3.0 | 361 | 180.0 | `-5.4501234166498155e-05` | `+5.45001593440529e-05` | `0.0` | `1.97e-05` | `2.13e-13` | `2.9824448062083118` | pass |
| 4.0 | 361 | 180.0 | `-4.8525703120004155e-05` | `+4.8524721724240265e-05` | `0.0` | `2.02e-05` | `8.40e-14` | `2.6633355426968413` | pass |

First-pass conclusion:

```text
The returned Colab capsule is technically valid and internally consistent.
The archive verifies, A100/JAX/x64 execution is recorded, all declared outputs
are present, and independent CSV recomputation supports the script verdict
TG_B2_DEFINITIVE_AWELL_ATTRACTION_CONFIRMED for this configured mirror test.
```

Caution: this is a clean first-pass result for the configured TG-B2 body-force observable. It should still receive deeper scientific review for time-series stability, breathing sensitivity, and scope boundaries before being promoted beyond this mirror-test result.
