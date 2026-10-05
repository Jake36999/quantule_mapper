# Quantule Mapper Colab Jobs

This directory contains the project-local Colab capsule lane for targeted burst-compute runs.

Use this lane when a run is already well specified, has a known baseline or clear analysis contract, and would otherwise keep the local GPU busy for hours. Do not use it for broad hunts, exploratory parameter searches, or jobs that cannot persist useful partial results.

## Start Here

Future agents should usually read these files in order:

1. `COLAB_CAPSULE_AGENT_GUIDE.md`
2. `COLAB_CAPSULE_WORKFLOW.json`
3. A recent manifest example, such as `tg_b1s_d_reproduction_manifest.json`
4. The success example in `TG_B1S_D_COLAB_SUCCESS_EXAMPLE.md`

Agents should not need to inspect `notebook_packager_v4.py` for normal use. The expected task is:

1. choose a targeted run;
2. freeze the baseline and comparison contract;
3. write or update one JSON manifest;
4. run the packager dry-run;
5. compile the `.ipynb`;
6. hand the notebook to the user for manual Colab upload;
7. review the returned result archive.

## Active Packager

```text
F:\quantule_mapper\colab_jobs\notebook_packager_v4.py
```

The packager builds a deterministic, self-extracting notebook. It embeds allowlisted files, verifies hashes after extraction, applies manifest runtime environment overrides before JAX imports, exposes the Colab control panel, runs registered jobs, and archives results to Drive.

## Successful First Live Run

The first completed live Colab reproduction was:

```text
TG-B1S-D 100P reproduction
row_id: tg_b1s_d_100p_primary
target_run: D3_100P_lam1
status: REPRODUCTION_WITHIN_DECLARED_TOLERANCES
runtime: 6250.04 seconds
archive: colab_jobs\results\tg_b1s_d_100p_reproduction_20260715_132529.tar.gz
```

See `TG_B1S_D_COLAB_SUCCESS_EXAMPLE.md` for the evidence and exact files.
