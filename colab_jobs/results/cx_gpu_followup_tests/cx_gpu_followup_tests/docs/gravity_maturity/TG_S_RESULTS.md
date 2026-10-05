# TG-S Source Semantics Characterization Results

Timestamp: 2026-07-14.
Run directory: `/content/qm_job/results/cx_tg_rate_source_semantics_bridge/tg_s_semantics`.
Status: `TG_NODE_STATE_LOAD_SOURCE_SUPPORTED`.

## Summary

This audit characterizes source instruments only. It does not rerun temporal/geometric feedback and does not relabel `TG_SOURCE_SEMANTICS_FAILED_ON_KG_NODE`.
Baseline node gate: `True`.

## Source Classifications

| source | classification | passes | label | reason |
| --- | --- | --- | --- | --- |
| S_state | NODE_STATE_LOAD | True | TG_NODE_STATE_LOAD_SOURCE_SUPPORTED | stable nonzero stationary load; global phase and translation invariant |
| L_lock | DENSITY_OR_BREATHING_ACTIVITY | False |  | does not cleanly separate locking from null/breathing/unlocking controls |
| R_relax | PHASE_TENSION_RELAXATION | True | TG_PHASE_TENSION_RELAXATION_SOURCE_SUPPORTED | detects mismatch relaxation, including scrambled relaxation, and is quiet on the stationary node |
| P_threshold | DENSITY_OR_BREATHING_ACTIVITY | False |  | comparison-only threshold source; cannot promote feedback by itself |

## Feedback Rerun Rule

A future TG-B1 rerun must choose exactly one supported source hypothesis and state it explicitly.