# IRER Archive — Round 1 (May–August 2025)

**Purpose.** A transcript-replacement archive of the IRER theory's original formalisation period.
The standard is *archival sufficiency*: treat the source transcripts as awaiting deletion. Every
document here must carry enough verbatim material and explanation that a new collaborator or
agent could understand, discuss, and critique IRER — Jake's original formulations, the reasoning
sequences, the AI formal translations, Jake's responses, the abandoned branches — without ever
opening `F:\transcripts` or the Declaration v9. (No transcript is deleted or modified; this
archive removes the *dependency*, not the sources.)

This supersedes the earlier Codex packages
(`../provenance_transcript_extraction/`, `../expanded_declaration/`), which are retained as
historical orientation material. Their diagnosis: they pointed at content instead of containing it.

## Structure

| Path | Layer | Content |
|---|---|---|
| `00_CORPUS_COVERAGE_REGISTER.md` | control | Every May–Aug conversation → triage scores, disposition, extraction status. Script-generated; statuses maintained by hand as dossiers land. |
| `00_V9_CITATION_RESOLUTION.md` | control | v9's named sources (`transcript.txt`, `sctriptpt2.txt`, `scriptpt3–5.txt`) located in the surviving per-chat archive by quote matching. |
| `_work_queue.md` | control | Extraction queue ordered by term density. |
| `conversations/2025-05 … 2025-08/` | **A** | One dossier per relevant conversation — the atomic, transcript-replacing unit. Verbatim segments with declared omissions. |
| `families/F1 … F7/` | **B** | Concept-family assemblies built *from* dossiers: original formulations (verbatim, chronological), developmental dialogues, formal translations + author responses, revisions/abandoned branches, applications/status, and (last) a canonical exposition. |
| `aletheia_context/` | **A/B** | The AI-development stream (Aletheia identity work, declaration-of-understanding material) — same dossier standard, kept separate from the physics record. |
| `declaration_v10/` | **C** | The Declaration of Intellectual Provenance successor: chapter files in v9's voice and structure, `DELTA_REPORT_V9_TO_V10.md` (everything recovered that v9 missed), and `build/` for the concatenated single-file deliverable. |
| `CROSSMAP_CONCEPTS_VS_RESULTS.md` | **cross** | Recovered concepts × current computational-stack results → **missed connections** the main workflow didn't make (e.g. gravity "failures" match IRER's *predicted* saturating/non-universal force; stability⟂transport = Dynamic Informational Inertia). Feeds the concept↔implementation register and proposes next tests. Interpretive only — no verdict/posture changes. |

## Concept families

- **F1** AIS/PIF · Informational Indifference · gradients · FMIA / axis of least effort / informational parallels
- **F2** OIW · Informational Resonance · Resonance Density · PAS · collapse/RFD · novelty-from-instability
- **F3** Quantules · Payan states · angular deficits · chirality
- **F4** Manifold topology / angles of the manifold · gradient-derived forces · chorotic boundaries/IEH · gravity model
- **F5** Time as chronology of resolution · observer model · observer-resolution loop · emergent spacetime & c_emergent
- **F6** Prime-harmonic resonance · coupling equations · entropy as resonance fragmentation · homeostasis · IQG equations
- **F7** Higher-order/speculative: sentience/SPTT · recursive field participation · inter-world resonance · societal metaphors · "42"
- **(F8)** Aletheia/AI-development context → `aletheia_context/`

## Extraction rules (Layer A)

1. Quote-first, commentary-second. Jake's theory-bearing turns are **never** paraphrase-only.
2. Every omission is declared: `[lines a–b omitted: <topic>]`. Formal translations, equations,
   and definitions are never trimmed.
3. Spelling/grammar preserved verbatim (theoretically meaningful in places).
4. Segments tagged: `original_theorising` · `concept_revision` · `application_method` ·
   `ai_context` · `provenance` · `project_report`.
5. Silence ≠ acceptance. Jake's uptake/rejection of an AI translation is noted only when visible.
6. Errors and later-superseded claims are preserved and labelled with a pointer to
   `docs/IRER_MASTER_HYPOTHESIS_CATALOG.md` — never silently corrected.
7. No invented connective reasoning. Editorial notes are clearly editorial.
8. Mixed physics/Aletheia conversations are segmented, never dropped.
9. **Quality-gate-locked standard (Jake, 2026-07-16):** the quote-first depth and segmentation of the two approved calibration dossiers (D0517, D0518-PF) is the binding bar. Do **not** compress dossiers to save space.
10. **Addendums and revisions are first-class events and must be labelled explicitly** in segment headers and Notes:
    - `[ADDENDUM: <name/number>]` — the moment content is adopted into an addendum/master file (e.g. "Save both … included in the addendum" → Addendum I–IV assignments).
    - `[REVISION of <concept/entry> — v<n>→v<n+1>]` — a drafting pass that changes existing text (paste → audit → "whats missing?" → revised paste → "what changed?"). Capture WHAT changed (the AI's change-tables where present, else a Notes diff) and which version the wording descends from.
    - `[SUPERSEDED by <where>]` — wording later replaced; keep both, point forward.

## Status board

| Stage | State |
|---|---|
| 0 — Triage register + citation resolution + skeleton | skeleton DONE; `tools/triage.py` written (in session scratchpad), **run pending** (platform tool outage 2026-07-16) |
| 1 — Extraction: May quality gate (3 pivotal dossiers) | 2/3 DONE (D0517 fundamental-forces, D0518 phase-field). #3 (DoA conversation) = **14,080 lines / 160 turns** — reconnaissance + extraction work order at `conversations/2025-05/_WORKORDER_D20250518_081202.md`; awaiting Jake's gate review before the big extraction |
| 1 — Extraction: May tranche | pending |
| 1 — Extraction: June / July / August tranches | pending |
| 2 — Family assemblies F1–F7 | pending (F1 seeds from Codex phase-1B salvage) |
| 3 — Canonical expositions | pending |
| 4 — Declaration v10 + delta report + single-file build | pending |
| 5 — Archival-sufficiency audit (fresh agent, no transcript access) | pending |

**Round 2 (later):** Sept 2025–Jan 2026 (`D:\memory_bank\bank 1` monthlies), `Aleheia'sChat.txt`
bridge record, Jan–Apr 2025 sweep, Category_A–D dedup — the implementation/application era,
which will also feed "how the theory is applied in the current project" documentation.
