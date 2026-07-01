# ADAPTIVE_HUNT_CALIBRATION_001

**Status: objective-calibration run, NOT a final scientific result.**

The first adaptive corrected-physics mutual-support hunt. Stopped early by operator after
2.24 h / 27 generations / **1296 configs** (of an authorised 4.5 h). Its value is diagnostic:
it exposed a bad scoring incentive before the full timebox was spent.

## Verdict
- 1296 configs searched (coherent + random-phase multiseed; broadband-noise scan alongside).
- **No validated mutual-support candidate.**
- Apparent positives (mid-run: iso_surv≈0.12, abl_sens≈1.75) **failed full-fidelity validation**
  (`validate_candidates.py`, N=48³/800 vs scout N=40³/500): ablation was finite there and merely
  removed the one ablated node (rest unchanged) — no disruption; 2/3 sat in runaway geometry
  (curv 116 / 298, D≈4–5).
- **High ablation sensitivity was caused by scout-resolution instability / ablation
  non-finiteness** — at N=40³/500 the ablated run went non-finite, adding +1 to the metric.
- **No broader model-class negative yet** — the run was stopped early *and* the objective was flawed.

> *"First adaptive hunt produced apparent support candidates, but full-fidelity validation showed
> they were instability artifacts. The objective must be corrected before the full timebox can be
> interpreted."*

## Root cause (the loophole)
`ablation_sensitivity` rewarded ablation-induced **instability**: a `+1` term for the ablated run
going non-finite, plus node/energy "disruption" that at scout resolution was numerical blow-up, not
bounded reorganisation. The adaptive search correctly found and exploited this loophole — evidence
the search machinery works; the fitness did not.

## Corrective actions (next run = bounded-support objective)
1. `bounded_ablation_sensitivity`: **0 unless BOTH intact and ablated runs are valid bounded
   trajectories** (finite, amp-bounded, non-runaway geometry, multi-node intact). No reward for
   ablation→NaN/blow-up/zero-node-collapse/saturation/curvature-runaway.
2. Reward only **bounded disruption of the remaining cluster** (surviving nodes weaken / coherence
   changes / energy-retention changes / spacing reorganises) while staying finite — NOT
   `6 nodes → 5 unchanged` (that is independent coexistence → score ≈ 0).
3. Support gate requires **all three legs**: intact survives ∧ isolated weaker than cluster ∧
   bounded ablation disrupts the remainder. Low iso-survival alone, or instability-driven
   ablation alone, do not qualify.
4. **No warm-start from contaminated elites** — restart from fresh LHS (the previous elites were
   selected partly by the flawed objective).
5. Log BOTH raw and gated ablation metrics; calibrate before the full timebox.

## Scientific interpretation
*"No genuine mutual-support candidate has been found yet. The first adaptive hunt found an
instability exploit in the objective. The broader corrected-model question remains open until a
clean bounded-support adaptive hunt is completed."*
