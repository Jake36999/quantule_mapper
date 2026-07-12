# Digitization Provenance Record - EMP-V5-NG

---

- **Target id:** EMP-V5-NG
- **Supports metric:** V5
- **Match-level:** `CLOSE ANALOGUE`

## Source

- **Full citation:** Jason H. V. Nguyen, Paul Dyke, De Luo, Boris A. Malomed, and Randall G. Hulet, "Collisions of matter-wave solitons," *Nature Physics* 10, 918-922 (2014).
- **DOI / stable link:** https://doi.org/10.1038/nphys3135 ; arXiv record https://arxiv.org/abs/1407.5087 ; arXiv PDF https://arxiv.org/pdf/1407.5087 ; arXiv source package https://arxiv.org/e-print/1407.5087
- **Verified to exist by:** Codex / 2026-07-12 - arXiv API metadata, PDF, and source package were downloaded and archived off-git; DOI and journal reference verified from arXiv metadata.
- **License / terms for the figure or data:** arXiv preprint and source package downloaded from arxiv.org for local provenance/review; journal version is Nature Physics content subject to publisher terms. Raw PDF/source/rendered figures are stored off-git; repository contains only a small derived qualitative table and provenance record.
- **Physical system:** Attractive-interaction Bose-Einstein-condensate matter-wave solitons; this is an analogue collision-outcome comparison source, not the IRER/Quantule Mapper system.

## Extraction

- **Figure / table number:** Fig. 2 panels a-c and Fig. 3 from arXiv source package `fig2.pdf` and `fig3.pdf`.
- **Quantity extracted:** Qualitative phase-dependent collision outcome. Rows encode relative phase class (`Delta phi approx 0` or `Delta phi approx pi`) and observed outcome (`COLLAPSE`, `MERGE`, `NO_COLLAPSE`, `PASS_THROUGH`) following `EMP_V5_NG_DATA_SCHEMA.md`. No collision speed was extracted because the V5 branch is a qualitative phase-outcome overlay and the selected panels do not provide a per-row speed value in the schema's required form.
- **Extraction method / tool:** Read arXiv TeX source captions/text; rendered source figure PDFs to PNG using bundled Poppler `pdftoppm.exe` at 220 dpi; visually checked Fig. 2/Fig. 3 rendered PNGs. No WebPlotDigitizer coordinate extraction was used because the V5 schema is categorical rather than axis-value digitization.
- **Operator + date:** Codex / 2026-07-12.
- **Estimated extraction uncertainty:** Phase labels are approximate classes rather than measured phase values: `0.000000` represents `Delta phi approx 0`, and `3.141593` represents `Delta phi approx pi`; practical class uncertainty is approximately +/-0.4 rad following the comparator's phase-class band. Outcome uncertainty is low for Fig. 2a/Fig. 2b/Fig. 3 because captions/text explicitly state collapse, merger, and pass-through; Fig. 2c is recorded as `NO_COLLAPSE` rather than `PASS_THROUGH` because it shows robust survival after many oscillations but does not tag soliton identities in the panel.
- **Calibration checks performed:** Confirmed CSV header exactly matches `EMP_V5_NG_DATA_SCHEMA.md`; confirmed allowed `observed_outcome` values; confirmed phase classes against the schema's in-phase/anti-phase definitions; checked rendered Fig. 2 and Fig. 3 against TeX captions and source text.

## Mapping to Quantule Mapper

- **Which QM observable it is compared against:** V5 C3 collision phase-by-outcome grid, reduced by the empirical comparator to survived-as-two vs not-transmitted by phase class.
- **Unit / scaling conversions applied (and their justification):** No numeric unit conversion. The only conversion is categorical: `Delta phi approx 0` -> `relative_phase_rad=0.000000`, `in_phase`; `Delta phi approx pi` -> `relative_phase_rad=3.141593`, `anti_phase`.
- **Known caveats in the mapping:** Different physical system, dimensionality, trapping, attractive BEC collapse mechanism, atom-number threshold effects, shot-to-shot variation, and qualitative categorical extraction. `COLLAPSE` is grouped by the comparator as not-transmitted, but its mechanism differs from C3 capture/binding. This dataset is a CLOSE ANALOGUE context overlay only and does not establish physical correspondence.

## Governance attestation

- [x] Stored raw source off-git (E:), only small derived table committed here.
- [x] Match-level left at `CLOSE ANALOGUE`.
- [x] Not referenced by `validation_config.yaml`, the `external_data.py` manifest, any Hunter config, or the
      production validation pipeline.
- [x] No verdict, gate, or IRER-framing change results from this dataset.
- [x] Comparison output marked `PROVISIONAL_UNTIL_CLAUDE_REVIEW`.
