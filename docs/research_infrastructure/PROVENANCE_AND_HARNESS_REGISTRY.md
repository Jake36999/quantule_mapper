---
tags: [record, infra, provenance, registry]
date: 2026-10-04
branch: Branch - Validation - Index
status: complete
---

# Component-hash provenance and the harness registry (Phase D)

This is the build record for Phase D of [[IMPLEMENTATION_PLAN_2026-10]]. Every run stamp now records
the exact content (git blob hash) of every repo module the run imported. Harnesses declare what they
are in a `HARNESS = {...}` manifest. A generated registry ([[HARNESS_REGISTRY]]) and the results index
connect runs to the harness that produced them.

> [!info] Sources
> - `jax_scout/provenance.py`: `git_blob_hash`, `component_hashes`; `stamp()` gains
>   `component_hashes` and `dirty_components`
> - `tools/stepper_staleness.py`: `blob_contains_fix`, `hash_verdict`; `staleness(...,
>   component_hashes=)`
> - `docs/registry/COMPONENT_FIXES.json`: `affects_files`
> - `tools/build_harness_registry.py` (new), producing `docs/registry/harness_registry.json` and
>   [[HARNESS_REGISTRY]]
> - `tools/build_results_index.py`: `harnesses` table, `runs.harness` (the harness id), `produced_by`
>   edges, hash-aware staleness
> - `.github/workflows/harness-manifest.yml` (new)
> - Tests: `tests/test_harness_registry.py` (new, 11), `tests/test_stepper_staleness.py`

## Summary
The [[SESSION_SYNTHESIS_2026-09-17]] §6 problem was that ~126 harnesses existed with no way to ask
whether one was current, what superseded it, which enquiry it served, or which runs it produced. It
is now answerable:
- **8** active harnesses carry manifests.
- **108** scripts that write to `sweep_runs/` are listed as `UNREVIEWED`, with run-id prefixes inferred
  from their source.
- **124 of 204** indexed runs resolve to the harness that produced them.

Provenance moves from "which commit" to "which file contents". This is what makes a dirty-tree run, or
a commit-misleading run, detectable.

## Detail

### Component hashes
- `git_blob_hash` reproduces `git hash-object` in-process. The repo uses `* text=auto` with
  core.autocrlf, so CRLF is normalised to LF for files with no NUL byte; the tests check it against
  `git hash-object`.
- `stamp()` hashes every imported module under the repo. It skips `.venv`, `site-packages` and
  `external research`, and reads HEAD's tree once to list `dirty_components`. There is no subprocess
  per file.

### Staleness by content (upgrade of Phase B)
A fix entry can name `affects_files`. When a run recorded hashes for any of them, staleness is decided
by **content**, which is strongest:
1. Find the commits that *introduced* that blob at that path.
2. Ask whether the fix is an ancestor of any of them.

Content beats the commit stamp. A run stamped on a post-fix commit but carrying a dirty, pre-fix
`kernels.py` comes out STALE; this is tested.

> [!warning] Two errors caught while building this
> 1. `git log --find-object` lists commits that **add or remove** a blob. The fix commit removes the
>    old `kernels.py` blob, so the first version judged pre-fix content "fixed". It now keeps only
>    commits after which the path holds the blob. Caught by `test_hash_verdict_uses_the_fixed_files_content`.
> 2. Inserting the manifest after the module docstring put it **above** `from __future__ import
>    annotations` in three TG harnesses. That is a SyntaxError, and `ast.parse` accepted it. The
>    manifests are now placed after the future imports, and `test_every_manifested_script_still_compiles`
>    uses `compile()`.

### Harness manifest
- It is a literal dict read with `ast.literal_eval`, **never by importing** the script, so a broken
  harness still registers.
- A script counts as a run-writer when a non-docstring string literal mentions `sweep_runs` **and**
  the source contains a write call. Docstring-only mentions (e.g. `provenance.py`) are not counted.
- A tool can opt out with `NOT_A_HARNESS = "reason"`.

### CI
`.github/workflows/harness-manifest.yml` checks **changed** files only, so the legacy UNREVIEWED
harnesses do not fail every build. It checks that the declaration exists and never gates running.

## What this changes

| status | item |
|---|---|
| **Unchanged** | Numerical output of every harness: the manifests are inert dict literals. |
| **Newly possible** | Exact reproduction from `component_hashes`; content-level stale detection; "which harness made this run" by query. |

## Issues raised

| # | issue | status |
|---|---|---|
| 1 | 108 UNREVIEWED harnesses | OPEN — triage gradually; the generator lists them, nothing forces it |
| 2 | `git` history unreadable from WSL inside a Windows-created worktree | NOTED — history tests skip there, and run in the main tree and CI |
| 3 | Runs from before Phase D have no `component_hashes` | NOTED — they fall back to commit ancestry, then date |

## Associated docs
- [[IMPLEMENTATION_PLAN_2026-10]] · [[HARNESS_REGISTRY]] · [[SOLVER_AND_RUNTIME_CHANGELOG]] · [[SESSION_SYNTHESIS_2026-09-17]]

## Branches
- [[Branch - Validation - Index]]
