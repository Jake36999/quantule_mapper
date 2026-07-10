# Evidence Gaps & Queued Enrichments

Where the package would be more defensible or richer. Most are **mechanical enrichment** (Codex-appropriate, no new
science); a few are **optional tiny reruns** that would materially strengthen a comparison (queued, not run without
approval). Nothing here blocks the current documentation.

## A. Mechanical enrichment (Codex, no new science)
| id | task | why it strengthens the package |
|---|---|---|
| ENR-1 | Fill missing `run_path`/`figures` fields in `09_machine_readable/evidence_manifest.json` from a `sweep_runs/` walk | complete traceability |
| ENR-2 | Generate SHA-256 checksums for canonical run summaries (summary.json, key .npz, key .png) → `checksum_manifest.txt` | tamper-evidence / reproducibility |
| ENR-3 | Verify every manifest/index link points to an existing file; flag broken/stale/duplicate | link integrity |
| ENR-4 | Build small **contact-sheet thumbnails** of existing canonical plots (Phase C case panels; gravity radial/geometry-law; k6 emergence gifs) and promote **only** the contact sheets (not the full trees) to `docs/evidence_package/<sector>/thumbs/` | portable visuals without bloat |
| ENR-5 | Render a couple of **canonical figures directly from existing `.npz`** (no rerun): C2.9 separation-vs-time (track_*.npz), C3 collision phase-diagram grid (collide_*.npz), C3 Q(ω) VK line, gravity radial Ω² cliff | the strongest results currently have data-only, no plot |

## B. Old-vs-corrected overlays (from existing data where possible)
| id | task | note |
|---|---|---|
| ENR-6 | Overlay old (pre-C2.6) vs corrected transport: μ≈0.036 vs v=2Dk, using existing `C24_LOCAL_N96/`, `C23_N96/` vs `C27_REDERIVE/` | the single most persuasive maturity figure; likely from existing summaries |
| ENR-7 | Tabulate the D_eff=D/151 identity: old one-step phase vs corrected, from the C2.6 gate/audit outputs | already-computed numbers; assemble into one table |

## C. Optional tiny reruns (QUEUED_OPTIONAL — approval required)
| id | task | cost | payoff |
|---|---|---|---|
| RUN-1 | Re-run **one** old-parameter case on the corrected substrate to produce a clean side-by-side (if no existing pair overlays cleanly) | ~1 smoke run | makes ENR-6 airtight |
| RUN-2 | **C2 anti-phase collision** (NLS analog of the C3 phase diagram) — does the first-order substrate show the same anti-phase node channel? | ~10–20 min | extends cross-substrate universality to collisions |
| RUN-3 | **C3 captured-remnant long-time fate** (evolve a captured pair long) — stable "Q-ball molecule" vs slow decay | ~15 min | closes an open C3 thread |
| RUN-4 | **C3 continuum-limit velocity fidelity** (bigger box + longer T) — tighten v_frac→1 to an identity | ~20–30 min | upgrades C3 transport from "supported" to "identity" |
| RUN-5 | **stable-overdense-load-on-ρ_vac-background** pilot — the gravity re-entry condition | research sub-project | unblocks the gravity ladder (rung B) |

## Notes
- **Do not promote raw trees** (`quantule_viz/outputs/`, `sweep_runs/`) into git — they are gitignored on purpose.
  Only promote **contact sheets** (ENR-4) and **freshly-rendered canonical figures** (ENR-5) that are small and
  canonical.
- Reruns (Section C) are the only items that produce new data; all are cheap and clearly scoped. RUN-2/3/4 are the
  natural next C3/C2 science; RUN-5 is the gravity re-entry gate.
- The package is complete and defensible **without** any Section C rerun; those are enrichments, not fixes.
