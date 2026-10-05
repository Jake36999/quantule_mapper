# Digitization Provenance Record — TEMPLATE

Copy this file to `provenance_<EMP-id>.md` and complete **every** field before any comparison is run against the
dataset. Incomplete provenance → the dataset is not usable in this branch.

---

- **Target id:** (e.g. EMP-V1-MM)
- **Supports metric:** (V1 / V5)
- **Match-level:** `CLOSE ANALOGUE`  *(fixed ceiling — do not raise)*

## Source

- **Full citation:**
- **DOI / stable link:**
- **Verified to exist by:** (name / date)  — *not just cited; confirmed accessible*
- **License / terms for the figure or data:**
- **Physical system:** (e.g. optical fiber / BEC bright solitons)  — note it is an *analogue*, not the IRER system

## Extraction

- **Figure / table number:**
- **Quantity extracted:** (axes, units)
- **Extraction method / tool:** (e.g. WebPlotDigitizer version)
- **Operator + date:**
- **Estimated extraction uncertainty:** (per-point or overall, with basis)
- **Calibration checks performed:** (axis endpoints, known reference points)

## Mapping to Quantule Mapper

- **Which QM observable it is compared against:** (V1 q̈(q,Δφ) / V5 outcome grid)
- **Unit / scaling conversions applied (and their justification):**
- **Known caveats in the mapping:** (dimensionality, integrability, parameter regime mismatch)

## Governance attestation

- [ ] Stored raw source off-git (E:), only small derived table committed here.
- [ ] Match-level left at `CLOSE ANALOGUE`.
- [ ] Not referenced by `validation_config.yaml`, the `external_data.py` manifest, any Hunter config, or the
      production validation pipeline.
- [ ] No verdict, gate, or IRER-framing change results from this dataset.
- [ ] Comparison output marked `PROVISIONAL_UNTIL_CLAUDE_REVIEW`.
