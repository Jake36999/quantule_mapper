# G1 Temporal Clock Calibration Results

Status: `TEMPORAL_CLOCK_CALIBRATION_FAILED`.

Primary run:

```text
sweep_runs/GRAVITY_MATURITY_G1_CLOCK_20260714_100527/
```

## Evidence

The run executed the bounded G1 matrix with:

- CPU SciPy sparse generalized eigenproblem;
- JAX GPU local KG time-domain evolution;
- `backend=gpu`, `device=cuda:0`, x64 enabled;
- 24 eigenfrequency rows;
- 24 time-domain rows;
- 24 clock-family comparison rows;
- 9 numerical-refinement rows;
- 5 falsification checks.

The near-source trapped eigenmode did close the eigen/time contract:

```text
core_N64_near_m12_w15_a1 relative frequency error = 6.489e-07
```

The flat and far contracts did not close under the preregistered local-clock setup:

```text
core_N64_flat_m12_w15_a1 relative frequency error = 6.796e-04
core_N64_far_m12_w15_a1  relative frequency error = 1.063e-02
```

The objective temporal differential was detected:

```text
near omega_time = 6.436298238072359
far omega_time  = 6.46626416056168
flat omega_time = 12.03422898751402
```

Measured amplitude invariance passed:

```text
amplitude spread = 1.7763568394002505e-15
```

The two tested clock constructions did not agree in shift direction:

```text
trapped eigenmode far-minus-near = 0.029965922489321173
high-mass compact packet far-minus-near = -0.01583549954732888
```

Localization telemetry explains the main failure boundary. The nominal far eigenmode drifted toward the low-lapse source region rather than remaining at the intended far clock position:

```text
core_N64_far_m12_w15_a1 mode localization error = 4.093792797333923
packet_N64_far_m20_w1_a1 mode localization error = 4.127097138823318
```

## Inference

The supplied objective temporal field can produce measurable time-domain frequency differences, but the current local-clock construction is not yet a calibrated local clock. The sparse eigenproblem and GPU evolution agree well only when the mode remains localized.

## Speculation

A stronger or differently designed probe-independent localization mechanism may be required before testing whether `omega_local ~= m N_t(R)` or another derived local clock law emerges.

## Falsified Interpretations

- G1 does not support `OBJECTIVE_LOCAL_CLOCK_LIMIT_SUPPORTED`.
- G1 does not support `TEMPORAL_CLOCK_MECHANISM_AGREEMENT_SUPPORTED`.
- G1 does not close the full `TEMPORAL_KG_EIGENFREQUENCY_CONTRACT_CLOSED` gate.

## Maturity Status

```text
CHARACTERIZED_NEGATIVE
```

## Next Smallest Experiment

Redesign the localized clock calibration instrument so the far clock mode remains localized at the intended position, then rerun only the flat/near/far eigen-time contract before expanding the matrix.
