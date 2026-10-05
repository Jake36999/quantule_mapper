# Digitization Provenance Record - EMP-V1-MM

---

- **Target id:** EMP-V1-MM
- **Supports metric:** V1
- **Match-level:** `CLOSE ANALOGUE`

## Source

- **Full citation:** F. M. Mitschke and L. F. Mollenauer, "Experimental observation of interaction forces between solitons in optical fibers," *Optics Letters* 12(5), 355-357 (1987).
- **DOI / stable link:** https://doi.org/10.1364/OL.12.000355 ; publisher landing page archived from https://opg.optica.org/ol/viewmedia.cfm?uri=ol-12-5-355&html=true ; figure-source page identified as https://www.researchgate.net/figure/Pulse-separation-at-fiber-output-sout-as-a-function-of-the-relative-phase-ph-of-the-pulse_fig3_26797710
- **Verified to exist by:** Codex / 2026-07-11 - DOI and publisher metadata verified from Optica HTML; figure caption and rendered figure verified from indexed ResearchGate figure page.
- **License / terms for the figure or data:** Publisher article is Optica Publishing Group content; ResearchGate figure page states "Content is subject to copyright. Terms and conditions apply." Raw/access-attempt artifacts are stored off-git for provenance only. Derived digitized points are recorded as a small measurement table for review.
- **Physical system:** Optical-fiber temporal solitons; this is an analogue comparison source, not the IRER/Quantule Mapper system.

## Extraction

- **Figure / table number:** Fig. 3, captioned "Pulse separation at fiber output sigma_out as a function of the relative phase phi of the pulse pairs for a close (filled circles) and a wide (open circles) initial spacing sigma_in."
- **Quantity extracted:** x-axis relative phase phi in radians from -pi to +pi; y-axis output pulse separation sigma_out/tau; two initial spacings sigma_in/tau = 1.53 and sigma_in/tau = 3.82. CSV uses q_half_separation = (sigma_in/tau)/2 and interaction_measure = (sigma_out/tau) - (sigma_in/tau), a signed separation-change proxy where positive means output separation grew and negative means output separation shrank.
- **Extraction method / tool:** Manual point extraction from the rendered Fig. 3 image shown by the direct ResearchGate figure asset through Codex web rendering; axis calibration and row calculation performed by visual inspection and spreadsheet-style arithmetic. HTTP and headless-browser local downloads of the same asset were blocked by host security, so the local off-git folder records access logs, screenshots of blocked attempts, publisher metadata, and checksums for all locally acquired artifacts.
- **Operator + date:** Codex / 2026-07-11.
- **Estimated extraction uncertainty:** relative_phase approximately +/-0.08 rad for most points and up to +/-0.12 rad near densely clustered points; sigma_out/tau approximately +/-0.08 for open-circle points and +/-0.12 for filled-circle points; interaction_measure inherits the same vertical uncertainty because sigma_in/tau is taken from the figure labels. The close-spacing filled-circle series includes a known paper-level caveat where pulse merger/self-frequency-shift effects distort the simple attraction branch.
- **Calibration checks performed:** x-axis endpoints calibrated to -pi, 0, +pi tick labels; y-axis calibrated to 0 through 7 tick labels; initial spacings read from figure labels sigma_in/tau = 1.53 and 3.82; q_half_separation values checked as 0.765 and 1.910; sign convention checked against the schema as output-separation growth >0 and shrinkage <0.

## Mapping to Quantule Mapper

- **Which QM observable it is compared against:** V1 q_ddot(q, Delta phi) phase-force sign/decay family. The extracted interaction_measure is a signed monotone proxy based on output separation change, not a physical force or acceleration.
- **Unit / scaling conversions applied (and their justification):** Phase is kept in radians. Initial separations are normalized by the reported pulse width tau and converted to half-separation q = sigma_in/(2 tau) to match the V1 comparator convention. interaction_measure is sigma_out/tau - sigma_in/tau and remains an arbitrary signed proxy; no force-unit conversion is applied.
- **Known caveats in the mapping:** Optical-fiber experiment, finite propagation length, Raman/self-frequency-shift effects, close-pair merger effects, figure-digitization uncertainty, and output-separation proxy all limit the comparison. This is a CLOSE ANALOGUE context overlay only and does not establish physical correspondence.

## Governance attestation

- [x] Stored raw source off-git (E:), only small derived table committed here.
- [x] Match-level left at `CLOSE ANALOGUE`.
- [x] Not referenced by `validation_config.yaml`, the `external_data.py` manifest, any Hunter config, or the
      production validation pipeline.
- [x] No verdict, gate, or IRER-framing change results from this dataset.
- [x] Comparison output marked `PROVISIONAL_UNTIL_CLAUDE_REVIEW`.
