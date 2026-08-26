---
tags: [index, methodology]
---

# Documentation Methodology

How this vault is built, what each kind of note is for, and the rules that keep the graph clean.
Authority for *structure*. [[IRER_MASTER_HYPOTHESIS_CATALOG|The master catalog]] remains the authority
for *scientific status*.

Conventions here derive from Jake's definitions in [[Main branch]], [[Side branch]],
[[Previous experiment]], [[Associated docs]] and [[Next experiment and or doc]].

---

## 1. The one rule that matters

**Generated content lives above the marker. Human content lives below it.**

Every generated note ends with:

```markdown
## Review notes

*(Jake's secondary review — this section is preserved across catalogue rebuilds;
anything above it is regenerated.)*
```

`tools/build_run_catalogue.py` rewrites everything above that line on every run and never touches a
character below it. So:

- **Never** hand-edit above the marker in a generated note — the next rebuild discards it.
- **Always** put your reading, corrections, and cross-links below it — they are permanent.
- If you disagree with a generated claim, say so below the marker. That disagreement is data
  (see §5).

---

## 2. Note kinds

| kind | where | generated? | template |
|---|---|---|---|
| **Run note** | `runs/<RUN_ID>.md` | yes — from `sweep_runs/` | — (built) |
| **Branch index** | `Branch - <Name> - Index.md` | yes — run listing regenerated | [[_templates/branch-index\|branch-index]] |
| **Experiment note** | sector folder | hand-written | [[_templates/experiment-note\|experiment-note]] |
| **Result / doc of record** | sector folder | hand-written | [[_templates/result-note\|result-note]] |
| **Visual review** | beside the run note, or below its marker | hand-written | [[_templates/visual-review\|visual-review]] |
| **Tracker** | [[EXPERIMENT_TRACKER]] | yes, with preserved readings | — (built) |
| **Index / MOC** | [[HOME]], [[runs/_INDEX]] | yes | — (built) |

A **run note** records what a machine did. An **experiment note** records what *we were asking* and
what we concluded. They are different documents and should stay different: runs are cheap and
numerous, experiments are the unit of reasoning.

---

## 3. Branch discipline

Per [[Main branch]] and [[Side branch]]:

- **[[Main branch]]** is the project's overarching goal — simulate a pocket of universe under IRER
  first principles and test whether emergent fields, forces and structures appear, and whether they
  follow real-world dynamics.
- A **side branch** is exploratory, robustness-testing, or scoping work that does not directly
  advance those goals.

**The linking rule: only a branch *index* links back to [[Main branch]].** Individual run and
experiment notes link to their branch index, never straight to the main branch. This is what stops
exploratory work drifting into the main line uncontrolled, and it is what keeps graph view and the
document canvases readable.

Current branches:

| branch index | kind | sector |
|---|---|---|
| [[Branch - Gravity - Index]] | main | TG ladder + Gravity-D mirror |
| [[Branch - Transport - Index]] | main | Phase D: NLS C2, KG C3 |
| [[Branch - Stability - Index]] | main | Phase C attractor + hunts |
| [[Branch - Validation - Index]] | side | parity, diagnostics, reproduction |
| [[Branch - Unsorted - Index]] | side | unclassified — triage these |

Each index carries a chronological tracker of its own notes at the bottom, per [[Side branch]].

**Naming.** Branch material is prefixed so it sorts and greps together:
`Branch - <Branch name> - <Document name>`. The index is always `- Index`.

---

## 4. Backlink categories

Every substantive note ends with the same four sections. Definitions are Jake's:

### Previous experiments
> An experiment which uses similar parameters, or is following the same branch, goals, or objective
> of a previous experiment. — [[Previous experiment]]

In generated run notes this is **auto-derived** as the preceding runs in the same family. Auto-derivation
approximates "same parameters/branch" by family and date; it is often right and sometimes wrong.
**Correct it below the marker.**

### Associated docs
> Defined similarly to a previous experiment, but based also on **context** — the note contains a large
> amount of semantic and descriptive information which could be interpreted differently depending on the
> context in which it is read. It is important to define and track the chronology of notes and their
> contained information. — [[Associated docs]]

Generated run notes auto-populate this with every prose document in `docs/` that names the run id.
That catches citation but not *context* — add context links by hand.

### Next experiment
> Creating these links with double square brackets makes Obsidian create a blank note, which is there
> for us to edit — to plan, record, or review. By creating these blank notes we have a way to track what
> documentation still needs filling out. — [[Next experiment and or doc]]

**A red (unresolved) link is a to-do, not an error.** Do not "clean up" unresolved links: they are the
documentation backlog. [[EXPERIMENT_TRACKER]] counts them.

### Branches
Link to the branch index only (§3).

---

## 5. Images and visual review

**Every figure gets a sentence.** An embedded image with no prose beside it is not documentation.
The pattern, from Jake's draft template:

```markdown
![[runs/_figures/RUN_ID/some_figure.png]]

Description of what this shows, what to look at, and what it means.
```

Three image sources, kept in separate folders so their provenance stays obvious:

| folder | what | regenerable? |
|---|---|---|
| `runs/_plots/<RUN_ID>/` | plots **generated here** from the run's CSVs | yes — delete and rebuild |
| `runs/_figures/<RUN_ID>/` | images **the run itself produced**, copied in | no — copies of run output |
| `runs/_renders/<RUN_ID>/` | `quantule_viz` renders, matched by run id | no — copies |

`sweep_runs/` and `quantule_viz/outputs/` are gitignored and sit outside the vault, so Obsidian cannot
display anything in them. The copies above are the vault-visible half of the two-layer model.

> [!important] The paired-reading protocol
> [[EXPERIMENT_TRACKER]] gives each visual experiment **two** reading blocks: one written by Claude,
> one left blank for Jake. Fill yours in *without* reading mine where you can — a disagreement between
> the two is the most valuable signal in the vault, and has been fruitful before. Resolved
> disagreements get recorded, not overwritten.

---

## 6. Callouts

Use callouts for facts worth spotting at a glance, so full text can stay long without overwhelming.

| callout | use for |
|---|---|
| `> [!abstract] Verdict` | the run's verdict string |
| `> [!warning]` | a caveat that changes how a number should be read |
| `> [!important]` | something that must not be missed |
| `> [!info]` | provenance, derivation, scope |
| `> [!tip]` | current status, what to do next |
| `> [!danger]` | a retraction or a known-bad result |

---

## 7. Frontmatter

Generated run notes carry a Dataview-queryable schema:

```yaml
run_id, date, family, sector, verdict, complete, source,
N, L, T, dt, elapsed_h, seed, git_commit, n_csv, n_plots, tags
```

Hand-written notes should carry at least:

```yaml
---
tags: [experiment, <sector>]
date: YYYY-MM-DD
branch: <branch index name>
status: planned | running | complete | superseded
---
```

`source: derived` marks a run note reconstructed from `RUN_COMPLETE.json` and handoff reports rather
than a `summary.json` — its verdict is indicative and should be confirmed against the source document.

---

## 8. Verdict vocabulary

Reuse the catalog's, never invent a synonym:

`CONFIRMED` · `FALSIFIED` · `NULL` · `INCONCLUSIVE` · `RETRACTED` · `OPEN` · `DESIGN-ONLY` · `PAUSED`

**A verdict in a result note is historical.** Only [[IRER_MASTER_HYPOTHESIS_CATALOG]] says what is
currently live. A note is not wrong for holding a superseded verdict — that is the record.

---

## 9. Standing posture

Every note inherits it: **no matter claim, no gravity claim, no emergent-physics claim.** Mirror-first;
the frozen Phase C dissipative operator stays byte-identical. If a note's prose drifts toward a stronger
claim than its evidence, that is a documentation defect and should be fixed like any other.

---

---

## 11. What changed as a result — the section that makes review possible

> [!danger] The gap this closes (Jake, 2026-08-25)
> A results document records a conclusion but not its consequences. Reviewing one months later, you
> cannot tell which version of the simulation it ran against, what changed because of it, whether an
> issue it raised was ever fixed, or whether something superseded it. The reader has to reconstruct
> the aftermath by hand — exactly the work the document should have saved.

**Every result, plan and record document carries two more sections:**

### What changed as a result
- **Code / model changes** — what was altered in the harness because of this, with references.
  *"Nothing"* is a valid and useful answer.
- **Verdicts changed** — confirmed, retracted, downgraded, superseded.
- **What was done next, and why** — the decision this document caused.
- **If nothing changed** — say so, and say whether that was a decision or a drift.

### Issues raised
A table, one row per issue, each with a status: `OPEN` / `RESOLVED` / `SUPERSEDED` / `WONTFIX`, and
what resolved it. **An issue with no status is the thing that makes a document impossible to review
later.**

### The automated half

`tools/build_doc_lineage.py` appends a generated **Lineage** block to every live working document,
derived from four signals: the citation graph, name succession (`_PLAN → _RESULTS`, `STAGE1 →
STAGE2`, …), what the master catalog says, and git history. [[DOC_LINEAGE]] aggregates them into a
review queue.

It reports **that** later work exists, never **why**. The semantic half above cannot be automated.

> [!warning] Two coverage limits, both real
> **Git is blind before 2026-07-01.** The repository was clean-slated in `909e6e2`; 52 documents
> predate it, including the entire June stability sector. For those, only citations and the catalog
> can say anything.
>
> **The catalog does not reference documents.** It names only a handful explicitly, referring to
> results by *verdict string* and prose instead. The lineage tool matches verdicts as well as names,
> and even so most documents have neither in the catalog. **This is the structural reason a document
> cannot tell you its own status — the link from evidence to status was never made.** Adding a doc
> reference to each catalog row would close it.

## 10. Rebuilding

```bash
.venv/Scripts/python.exe tools/build_run_catalogue.py   # run notes, tracker, branch indexes
.venv/Scripts/python.exe tools/build_doc_lineage.py     # lineage blocks + DOC_LINEAGE.md
```

Rebuilds run notes, branch indexes, the tracker, the index, and the triage note; imports any new
figures and renders. **Safe to run at any time** — everything below the Review-notes marker survives.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `292fc96` (2026-08-25) — *Obsidian vault: run catalogue, experiment tracker, documentation methodology*
**Revised since:** 2 commit(s), most recently `3eb93af` (2026-08-26)

**Harness code changed since it was written:** 3 commit(s) to `jax_scout/`.
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[INFRASTRUCTURE_AND_ORCHESTRATION_ASSESSMENT]], [[Main branch]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
