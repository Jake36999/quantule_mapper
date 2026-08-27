---
tags: [record, meta, infrastructure]
date: 2026-08-25
status: complete
---

# Infrastructure & Orchestration Assessment

Author: Claude, 2026-08-25. Assessment of the orchestration method and four proposed additions:
CI/CD for microservice deployment, expanding Redis/Docker infrastructure, a centralised results
database, and a RAG memory layer over the Obsidian vault. Plus alternatives.

Every recommendation is judged against one test, set by
[[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW|the pressure test]]: **does this advance validation, or does
it just add machinery?** The bottleneck is not throughput. It is that 16 free parameters face one
external comparison.

---

## 0. Two corrections to the premise, both material

> [!danger] There is no Redis or Docker infrastructure in this repository
> A full scan of every `.py`, `.json`, `.yml`, `.toml` and `package.json` outside `docs/transcripts/`
> returns **zero** references to Redis, Celery, RQ, Docker or docker-compose. No `Dockerfile`, no
> `docker-compose.yml`, no CI config of any kind. Redis appears **only in the 2025 transcripts and
> design documents** — it was discussed and specified, never implemented here.
>
> So "expanding the existing Redis and Docker infra" would in fact be **building it from scratch**,
> and should be costed as such.

> [!important] A centralised results database already exists — and was abandoned
> `orchestrator/` is a complete job system: `orchestrator_engine`, `orchestrator_service`,
> `scheduling/queue_manager`, `result_processor`, `job_manifest`, `contracts`, `run_identity`,
> `schema_utils`, `diagnostics/`. It is backed by **`queue_runtime.db`, a 2.8 MB SQLite store holding
> 1,850 result rows and 2 registered workers** — so it ran at real scale.
>
> It is dead. Every one of those 1,850 rows was written on **21–22 March 2026**, and **all 1,850
> point at a `G:\quantule_mapper\` path that no longer exists.** Nothing since has touched it. The
> 182 catalogued runs from June onward write `summary.json` files to `sweep_runs/` instead.
>
> There is also `mcp_server/` — a read-only MCP query layer (`data_access`, `guards`, `write_tools`)
> last touched 2026-07-03, also dormant.

**The right question is therefore not "should we build a results database" but "why did the one we
built stop being used, and does that reason still apply?"** It does, and the reason is instructive —
see §3.

---

## 1. What the orchestration method actually is today

| layer | mechanism | assessment |
|---|---|---|
| Queue | [[RUN_QUEUE]] — a 20-column markdown table, hand-edited | Crude, and **appropriate**. It carries rationale, gates, lane choice and launch notes — things a job queue cannot hold. |
| Dispatch | Hand-launched `wsl.exe … python jax_scout/…` | Fragile (three consecutive failures to the WSL client-exit lifecycle) but low ceremony |
| Compute | One GTX 1080 via WSL/JAX; Colab A100 capsule for heavy runs | Correct for the scale |
| Results | `sweep_runs/<RUN_ID>/summary.json` + CSVs, gitignored | Heterogeneous schemas; works, needs an index |
| Index | [[runs/_INDEX\|Run catalogue]] + [[EXPERIMENT_TRACKER]], regenerated from disk | New, and the right shape |
| Provenance | git commit, preregistered gates, `RUN_COMPLETE.json` sentinels | Good where used; inconsistently used |

**Throughput is not the constraint.** 182 runs in ~10 weeks, three of them in August. The longest run
this session was 5.68 h; the useful one was 0.69 h. One user, one GPU, no concurrency pressure.

---

## 2. The four proposals, assessed

### 2a. CI/CD for microservice deployment — **split verdict**

**Microservice deployment: NO.** Microservices solve problems this project does not have — team-scale
parallel development, independent deploy cadence, multi-tenant scaling, fault isolation across
services. There is one user, one machine, no service, and nothing deployed. Adopting them would add
network boundaries, serialization, container builds and orchestration config to a workload that is
currently `python script.py`. Every hour spent there is an hour not spent on validation.

**CI: YES — but for something completely different, and it is the single best infrastructure
investment available.**

> [!important] Run the physics identities as regression tests on every commit
> The project's most valuable asset is its identity checks — `v = 2Dk` to 0.9999, energy and U(1)
> charge to ~1e-13, the momentum ledger closing at second order, the `off`-arm null being exactly
> zero, well/hill antisymmetry. **All three bugs in the integrity ledger were caught by one of these
> contradicting a known identity.** They are currently run *by hand, occasionally, by whoever
> remembers.*
>
> Automating them turns the project's best habit into a property of the system. A bug in
> `core_saturation_search` — 38 dependent modules — would be caught the same day rather than after it
> had reshaped four verdicts.

Concretely: a `tests/test_identities.py` running small, fast (N=32, T≤4) versions of each identity
check on CPU, wired to GitHub Actions on push. Minutes of runtime, no GPU needed. This is **not**
CI/CD for deployment; it is CI as an instrument-integrity guard, and it directly serves threat **T4**
and **T8**.

### 2b. Expanding Redis / Docker — **NO**

It does not exist to expand (§0), and the case for building it is weak:

- **Redis** is a shared, low-latency, multi-consumer store. There is one consumer, and latency is
  irrelevant when runs take hours. SQLite already outperforms it for this access pattern and needs no
  daemon.
- **Docker** has one genuinely good use here — **pinning the JAX/CUDA/driver stack for
  reproducibility**, which is a real concern given the numpy-ABI breakage already seen in the system
  Python. But that is one Dockerfile for reproducibility, not "infrastructure," and even that is
  partly covered by the recorded `environment_versions.json` and the 7.9e-12 cross-platform
  determinism evidence.

If reproducibility becomes a publication requirement, write the Dockerfile then. Not before.

### 2c. Centralised results database — **YES, but as a build artifact, not a service**

The need is real. Every cross-run analysis this session — the mass-axis gradient fraction, the verdict
polarity distribution, the activity profile — was an ad-hoc Python script re-parsing JSON. That is the
symptom a results index fixes.

But note **why the last one died**: `orchestrator/` was built for the **hunt era** — mass parameter
sweeps, many short jobs, worker pools, a live queue. The project then pivoted to **few, long, bespoke
GPU runs**. A queue-and-worker architecture is the wrong shape for three runs a month, so it fell out
of use and rotted (stale `G:\` paths). *Rebuilding a service would repeat that.*

**Recommendation:** extend `tools/build_run_catalogue.py` to emit `docs/runs/_index.parquet` (and/or a
small SQLite) alongside the notes it already writes. Same single pass over `sweep_runs/`, no daemon,
no service, no state to rot. Regenerable from disk at any time, so it can never disagree with the
runs. Then cross-run analysis is one `pandas.read_parquet` away.

This also fits the established two-layer model: tracked index, untracked raw data.

### 2d. RAG memory over the Obsidian vault — **NO as framed; there is a better version**

The retrieval problem is largely already solved. The vault has structured frontmatter (`run_id`,
`date`, `family`, `sector`, `verdict`, `N`, `L`, `T`, `dt`, `git_commit`), Dataview queries over it,
a citation graph, generated lineage blocks, and full-text search. Adding embeddings would buy fuzzy
semantic recall on top of a corpus that is already densely indexed and highly structured.

And there is a specific hazard:

> [!danger] RAG over your own notes is a confirmation amplifier
> Semantic retrieval surfaces documents that are *similar* to the query — which means it returns your
> own prior framing, in your own vocabulary, ranked by how well it matches what you already think.
> For a research programme whose most under-mitigated threat is **reviewer scarcity (T10)** — every
> reviewer to date being Jake or an AI agent sharing priors — a system optimised to retrieve
> agreement is pointed the wrong way.

**The version worth building inverts it.** Not "find documents related to this claim" but:

> **"Find documents that contradict this claim."**

A contradiction detector, using the structure that already exists: same `family` or `sector`, opposing
verdict polarity, later date, overlapping parameter ranges. That is mostly a *structured* query, not a
semantic one — the frontmatter already carries what it needs. It attacks T10 instead of feeding it,
and it is far cheaper than an embedding pipeline.

The "custom evaluator taking advantage of the Obsidian linking architecture" instinct is right. Point
it at disagreement rather than recall.

---

## 3. Additional proposals

| # | proposal | value | effort | serves |
|---|---|---|---|---|
| **A1** | **Physics-identity CI** (§2a) | **highest** | low | T4, T8 |
| **A2** | **Results index as a build artifact** (§2c) | high | low | analysis throughput |
| **A3** | **Degrees-of-freedom field in every run record** — `params_free`, `params_fixed_before`, `observations_matched` | **high** | low | **T2** — the critical threat. An infrastructure answer to an epistemics problem: it makes parameter freedom impossible to overlook because every result carries the count. |
| **A4** | **Contradiction finder** (§2d) | high | medium | **T10** |
| **A5** | **Decide the orchestrator's fate** — revive deliberately or delete | medium | low | 1,850 stale rows and a dormant queue system are a trap: a future agent will find `orchestrator/` and assume it is live. Whatever the choice, record it. |
| **A6** | **Content-addressed run manifests** — hash of (harness source + config + git sha) as run identity | medium | low | Makes "was this run against the same code?" answerable, which [[DOC_LINEAGE\|the lineage work]] showed is currently unanswerable pre-clean-slate |
| **A7** | **Standard launcher script** wrapping the WSL keep-alive + detached launch + liveness check | medium | low | Three consecutive run failures this session came from this being ad-hoc |
| **A8** | **Consolidate the 11 duplicate energy/force observables** into one audited module | high | medium | T8 — carried over from the pressure test |

---

## 4. Recommendation

**Build three things, all small:**

1. **A1 — physics-identity CI.** Institutionalises the practice that caught all three bugs.
2. **A2 — results index as a build artifact.** Removes the ad-hoc-script tax on every cross-run
   question.
3. **A3 — the degrees-of-freedom field.** The cheapest possible mitigation for the most dangerous
   threat, and it changes how every future result reads.

**Explicitly do not build:** microservices, Redis, a Docker orchestration layer, a results *service*,
or a RAG recall layer. Each is a reasonable engineering instinct aimed at a problem this project does
not have — and the project's scarce resource is not compute or retrieval, it is **validation**.

**Decide (A5):** the orchestrator is either revived on purpose or removed. Leaving a dormant job
system with 1,850 rows pointing at a dead drive is the kind of thing that misleads a future reader —
including a future agent — into thinking it is live.

> [!warning] The general risk in this whole category
> Infrastructure work is legible, satisfying, and produces visible progress — exactly like verification
> work, and with the same failure mode. A better queue does not make the theory more testable. The one
> number that would change this programme's standing is a parameter-free dimensionless prediction, and
> **no item on this page produces one.** A1–A3 are worth doing because they are cheap and protect what
> already works. They are not progress on the science.

---

## What changed as a result

- **Code / model changes:** none yet.
- **Verdicts changed:** none.
- **What was done next, and why:** pending decision on A1–A3 and A5.

## Issues raised

| # | issue | status | resolved by |
|---|---|---|---|
| 1 | No CI; physics identities run by hand and inconsistently (A1) | OPEN | |
| 2 | No cross-run results index; every analysis is an ad-hoc script (A2) | OPEN | |
| 3 | Degrees of freedom not recorded per result (A3, threat T2) | OPEN | |
| 4 | `orchestrator/` + `queue_runtime.db` dormant with 1,850 stale-path rows (A5) | OPEN | |
| 5 | No standard launcher; three run failures from ad-hoc WSL launching (A7) | OPEN | |

---

## Associated docs

- [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]] — the threat register this is judged against
- [[META_ANALYSIS_BRANCH_PROGRESS]] · [[Main branch]] · [[RUN_QUEUE]] · [[DOCUMENTATION_METHODOLOGY]]

## Branches

- [[Main branch]]

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `9517d9f` (2026-08-26) — *Infrastructure assessment: build three small things, not a platform*
**Revised since:** 1 commit(s), most recently `60093e5` (2026-08-27)

**Harness code changed since it was written:** 1 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[ACTION_PLAN_2026-08]], [[SESSION_SYNTHESIS_2026-08]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
