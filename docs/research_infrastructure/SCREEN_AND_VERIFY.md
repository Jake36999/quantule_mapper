---
tags: [record, infra, basins, performance]
date: 2026-10-08
branch: Branch - Stability - Index
status: complete
---

# Screen in fp32, verify in fp64

fp32 runs 3.7–4.5× faster than fp64 on the GTX 1080 ([[BATCHED_RUNS]]). It drifts by about 0.1–0.5%
over an a\* replay, which is small next to the gaps between basins but not small next to a basin
*boundary*. The workflow below spends fp32 on the bulk of a sweep and fp64 only on the points where
precision could change the answer. **Measured on a real a\* screen, every re-checked point landed in
the same basin in fp64.**

> [!info] Sources
> - `tools/screen_verify.py` (`plan`, `compare`), `tests/test_screen_verify.py`
> - Uses `tools/basin_cluster.py` ([[BASIN_MAPPING]]) and `protocol.precision` (CL-013)
> - Demo evidence: `docs/instrument_integrity/evidence/screen_verify/`

## The workflow
```
python tools/run_spec.py specs/approved/<ensemble>.json --precision fp32       # 1. screen
python tools/basin_cluster.py sweep_runs/<SCREEN> --group-by param_a          # 2. cluster
python tools/screen_verify.py plan sweep_runs/<SCREEN> --group-by param_a     # 3. choose fp64 re-runs
python tools/run_spec.py specs/proposed/<screen-id>-verify.json               # 4. a human starts it
python tools/screen_verify.py compare sweep_runs/<SCREEN> sweep_runs/<VERIFY> # 5. compare
```

**Step 3 (`plan`) selects these points:**

| reason | which points |
|---|---|
| `boundary` | Adjacent parameter points (along one `--group-by` axis) whose basin sets differ. One replicate per basin is taken at both points. |
| `ic_split` | A parameter point whose replicates land in more than one basin. One replicate per basin. |
| `outlier` | Every member of a basin smaller than `--min-basin` (default 2). |
| `representative` | The member nearest each basin's centroid, to confirm that every basin exists in fp64. |
| `screen_failed` | Points that stopped early or produced no descriptors. |

**What `plan` writes:**
- One fp64 spec: a zip sweep that rebuilds exactly those points. `--N` can raise the grid. It goes to
  `specs/proposed/`.
- `verify_plan.json`, written into the screen run, which maps each verify point back to its screen
  point.
- It **never launches** anything.

**Warning.** If more than half the screen is selected, `plan` warns that the screen has little basin
structure. At that point verification costs about as much as an fp64 sweep would have.

**Step 5 (`compare`):**
- Places each fp64 end state in the *screen's* descriptor space, using the same columns and scaling.
- Assigns it to the nearest screen basin within `--match-tol`, or labels it `NEW`.
- Writes `VERIFY.md` and `verify.json` into the verify run.
- A `DISAGREE` row means the screen's map is wrong at that point. The output is descriptive: nothing is
  corrected automatically.

## First real use: the a\* neighbourhood (2026-10-08)
**Setup:**
- 5 gains (param_a 0.52–0.59, with a\* = 0.55223) × 4 seeds
- K=6, N=32, full T=360 (72,000 steps)
- the screen ran in fp32 as one batch of 20

**Steps:**
1. **Screen:** 20 points in 12.5 min. The fp32 screen took **740 s for 20 points**.
2. **Cluster:** 9 basins.
3. **Plan:** 17 of 20 points selected, so the structure warning applies (see the finding below).
4. **Verify:** 17 points in fp64 at N=32, 44 min. The fp64 verify took **2,614 s for 17 points**, about
   4.2× the cost per point.
5. **Compare:** **AGREE 17 / 17.**

**How close fp32 came to fp64:**
- **Each point to itself:** the fp32→fp64 distance in relative shape units has a median of 0.0014 and
  a maximum of 0.0082.
- **Against the basins:** the basin radius is 0.2, and the closest pair of basin centroids is 0.19
  apart. Precision moved end states about **25× less** than the smallest gap between basins.
- **Energy ratio:** er differed by 0.06–0.8% at T=360.

**Finding: N=32 does not resolve a\*.** Every end state has 0 nodes and contrast below 12. The field
never localises into the 4- or 6-node states seen at N=96, so the 9 "basins" are a smear of dispersed
states rather than distinct objects. That is why `plan` chose almost everything.

So this demo validates **precision**: fp32 does not move states between basins, even over a full
replay. It does **not** validate N=32 as a screening resolution for a\*. **For a\*, screen at N≥48**
(1.7 ms/step in fp32). Basin structure is the screen's job, and resolution, not precision, is the
limit here.

## Issues raised

| # | issue | status |
|---|---|---|
| 1 | Screening a\* at N=32 shows no localised states, so its basin map is not meaningful | NOTED — screen a\* at N≥48; `plan` now warns when most points are selected |
| 2 | WSL's JAX env has no sklearn, so replicate clustering tests run only in `.venv` | NOTED — the end-to-end test uses one seed per point so it runs in WSL |

## Associated docs
- [[BATCHED_RUNS]] · [[BASIN_MAPPING]] · [[SOLVER_AND_RUNTIME_CHANGELOG]]

## Branches
- [[Branch - Stability - Index]]
