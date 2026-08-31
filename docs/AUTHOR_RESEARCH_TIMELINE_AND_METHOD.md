# Author Research Timeline and Method

Status: provenance and method note. This document is not scientific evidence for IRER. It records the human research arc behind IRER and Quantule Mapper: why the framework was formalized, why simulations became the falsification route, why AI assistance was used, and why nulls, retractions, and instrument bugs are preserved.

## 1. Purpose

This document records Jake McIntosh's personal and methodological timeline behind IRER and Quantule Mapper.

It should not be read as:

- evidence that IRER is correct;
- proof of originality;
- proof of external physical correspondence;
- an institutional rejection narrative used as validation.

It should be read as:

- conceptual provenance;
- authorship context;
- explanation of the AI-assisted formalization route;
- explanation of why Quantule Mapper was built;
- a record of the project's self-correction method.

The central framing is:

> I started because I wanted to see where my understanding of physics was wrong, so I could assess how well I had learned what I had been studying independently.

## 2. Early Independent Study

### 2012

At age 11, Jake began studying physics and science independently through encyclopaedias and similar material, with an obsessive focus. During this period he formed his first immature "hypothesis."

### 2014

Jake later discovered that at least one early guessed hypothesis was already true and well understood in existing science. This established an important pattern for the later project: independent intuition must be checked against existing knowledge.

## 3. Formal Education Mismatch

### 2016

Jake applied to study maths, physics, and chemistry at college, but was rejected because his high-school grades were not competitive enough. He was instead recommended a BTEC science course oriented toward producing healthcare workers, such as radiologists.

### 2017

After Jake asked his physics teacher whether Schrodinger's cat could be used as an analogy for quantum tunneling and fluctuations, the teacher recognized that the course was not ideal for Jake's intended physics route. It also became clear that the course would not support entry into university physics in the way Jake had planned.

This was later confirmed when the college took him to a lecture and arranged for him to speak to the head of physics for UK universities. Jake left college at the end of that year.

## 4. Return to Formal Learning

### 2024

Jake applied for Access to Higher Education through the Open University route.

Around this period, he began reassessing whether his independently developed way of understanding physics was correct, incomplete, or misleading.

## 5. IRER Formalization

### 2025

Jake worked with AI to document the way he understood physics. The aim was not to ask AI to invent a theory, but to preserve, organize, and formalize Jake's own conceptual structure.

On reflection, part of why Jake's framework blended existing models together was simple: nobody had told him that those models could not be blended in that way. That conceptual blending became IRER.

The first formalized record was the Declaration of Intellectual Provenance. After that, the project immediately focused on making the theory falsifiable and following the evidence trail.

At the start of this period, Jake could not code at all. He learned to use LLM services as support for programming, system design, documentation, and review. This still required a steep learning curve, especially around architecture, workflows, validation, and reproducibility.

## 6. Simulation and Systems Phase

### May 2025

Jake built early 1D simulations and ran roughly 30,000 exploratory runs. Some early promising signs motivated deeper tooling.

### December 2025

The system was upgraded toward a higher-dimensional simulation concept. The validation metrics and orchestrator design could not keep up with the model complexity.

### January 2026

The orchestrator designs were upgraded.

### February 2026

The Hunter / algorithmic search designs were upgraded.

### March 2026

Initial runs began, but the project encountered numerous execution, validation, and systems issues.

### April 2026

Jake built other applications to practice the workflows and techniques needed for larger system design.

### May 2026

Improved workflows and techniques were used to finalize the modern Quantule Mapper system.

### June 2026 onward

Quantule Mapper became a robust scientific workbench and exploration engine. AI support now accelerates development, documentation, and review, but the project no longer relies on LLM inference to assess data.

Instead, the system emphasizes:

- rigid mathematical tests;
- conservation identities;
- validation gates;
- reproducible runs;
- explicit hypothesis verdicts;
- cautious language;
- preservation of nulls, bugs, and retractions.

## 7. Methodological Stance

The project began as an attempt to find where Jake's understanding of physics was wrong. That question is still not fully answered.

The important change is that the project now has a method:

- preserve original articulations;
- preserve transcripts;
- preserve the Declaration of Intellectual Provenance without retroactive editing;
- preserve null results;
- preserve retractions;
- preserve instrument-bug reports;
- compare against existing models;
- use simulations as falsification tools;
- treat visual structure as insufficient;
- require mathematical and numerical validation;
- separate what is supported, falsified, paused, speculative, or unresolved.

Jake expected the work to be quickly identified as an existing idea, or to fail from a major fault early on. Instead, the project became a long-running investigation and systems-building effort.

## 8. Current Status

IRER remains a developing framework. Quantule Mapper has shown that several IRER concepts can be translated into known nonlinear-field mathematics. Some sectors are internally supported, some have been falsified or retired, and gravity-like claims remain paused behind explicit re-entry gates.

The current value of the project lies in:

- its self-correcting numerical method;
- its willingness to preserve failed ideas and bugs;
- its translation of Jake's conceptual framework into testable mathematical forms;
- its shift away from narrative interpretation and toward reproducible validation.

No part of this timeline changes the scientific status of any hypothesis. It records why the work exists and how its method developed.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 6 commit(s), most recently `f2b0527` (2026-08-27)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
