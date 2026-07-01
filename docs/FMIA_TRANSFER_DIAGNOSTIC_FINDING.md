# FMIA Transfer / Interaction-Rate Diagnostic — Finding 001

**Date:** 2026-06-20
**Contract:** `IRER-FMIA-TRANSFER-DIAG-v2` (`jax_scout/transfer_diag.py`)
**Status:** SCOUT-level re-analysis. Not IRER evidence. Not promotion/rejection. CuPy/ASTE
validates any finalist.
**Source population:** `sweep_runs/ADAPTIVE_HUNT_20260620_082624` (the clean doubly-gated
1.5 h hunt: 864 evals, 614 bounded stable-node configs). Re-analyzed the 12 most-collective
(lowest `iso_surv`).
**Run:** `sweep_runs/TRANSFER_DIAG_20260620_111531`.

---

## 1. Reframe of the prior ablation result (precise scope)

The destructive-ablation panel ("remove one node, do the others destabilise?") returned
`support_legs = 0 / 864`. The correct statement is:

> **Destructive ablation-dependency was not observed.**

This is **not** "node support is falsified." What the ablation work actually ruled out:

- destructive, ablation-critical mutual support (removing a node does not collapse the rest);
- A-field rescue topologies producing ablation-sensitive dependency;
- instability-driven false positives (the energy-runaway and non-finite-ablation loopholes,
  both gated out).

Per IRER theory (Declaration of Intellectual Provenance v9), destructive ablation is the
**wrong test**. Once stable nodes are set, support manifests as the **rate of interaction /
transfer of information** between them — Concept 21 *Fields of Minimal Informational Action*
(FMIA): systems route along *Informational Parallels*, "coherent channels … characterized by
low resistance or minimal informational dissipation/tension"; "Gradient-Derived Informational
Forces propagate along FMIA Informational Parallels"; overlapping trajectories form
*Interference Channels*. Concept 22 ties the routing metric to a "resonance-weighted metric
tensor g_ij(RD,PAS) governing geodesics in Informational Parallels" — our Ω²/Δ_g conformal
metric. The author's own proposed empirical test (provenance L2501): represent ρ-peaks as
nodes and their interactions as edges; measure preferred pathways of information/stress flow.

This diagnostic operationalizes exactly that.

---

## 2. What the diagnostic measures

Time-resolved node tracking over a captured trajectory (N=48, 800 steps, 41 snapshots),
then a **node-interaction graph** whose edges are measured transfer channels:

| Channel | Theory anchor | Metric |
|---|---|---|
| Energy transfer | energy transference along least-resistance paths | residual lagged anti-correlation of E_i(t), E_j(t) |
| Phase / current coupling | Gradient-Derived Forces along Parallels | coupling of phase-fluctuation residuals; action-rate (dφ/dt) coupling |
| Geometric corridor | FMIA Informational Parallel = low-resistance channel | Ω² density-bridge conductance between node centroids |
| Information flux | "how information/stress moves through the substrate" | J_info = ρ∇φ flux through inter-node bridge, projected on node axis |
| Interference lattice | overlapping FMIA Interference Channels | shared local power-spectrum overlap |

## 3. Critical methodology — null-referencing (why v2 ≠ v1)

The **raw** metrics are dominated by artifacts that have nothing to do with coupling:

- `raw_phase_lock` ≈ 0.61–0.97, `raw_E_xcorr` ≈ 0.82–1.00, `raw_interference` ≈ 0.82–0.91.

Two identical-frequency oscillators in the same potential trivially hold a constant phase
offset and co-settle in energy → "phase lock ~1", "energy xcorr ~1" with **zero** real
coupling. This is the same class of undiscriminating-metric trap as the pinned
`shear_fraction` and the energy-runaway loophole. v2 measures coupling as **excess over a
null**:

1. **Frequency-ramp detrend** — remove each node's own linear phase ramp (kills the
   identical-frequency triviality).
2. **Global-mode regression** — regress the global breathing mode E_tot(t) out of each node
   energy, and the global phase-wobble mode out of each node phase (a *shared global driver*
   is collective, **not** pairwise transfer — this is the collective-vs-transfer cut the
   objective requires).
3. **Circular-shift surrogate** — 24 random time-shift surrogates per pair give the chance
   coupling baseline; report signal − surrogate-mean.
4. **Random-window baseline** — interference overlap minus node-vs-random-location overlap.
5. **Cross-sim independence floor** (`transfer_null_control.py`) — feed the pipeline node
   series from *different, independent* simulations (true coupling = 0) to measure the
   metric's intrinsic floor.

Effect of null-referencing (per-pair examples):

- energy exchange 0.35 → **0.05**;  interference 0.87 → **0.07**.  Both collapse to ~null.
- phase coupling survives the surrogate (0.26–0.78) → required the independence-floor control.

## 4. The independence-floor control (decisive)

`transfer_null_control.py`, phase-coupling excess:

| | mean | median | p95 | max |
|---|---|---|---|---|
| WITHIN candidate (in-situ, real pairs) | 0.590 | 0.660 | — | 0.788 |
| CROSS candidate (independence floor, coupling=0) | 0.387 | 0.430 | **0.730** | 0.758 |

Separation within−cross = 0.203 (Cohen d ≈ 0.92), but the **floor is high**: the metric
credits ~0.43 of "coupling" to genuinely independent series, and only **28 %** of in-situ
pairs exceed the floor's p95. Classification thresholds were set to this control-derived
floor (`THR_PHASECOUP = 0.73`) so a transfer class fires only above guaranteed-zero.

## 5. Re-classification of the 12 most-collective candidates

| class | n |
|---|---|
| `collective_density_threshold` | 11 |
| `phase_current_transfer_candidate` | 1 |

- **`omega_corridor_conductance ≈ 0` across the whole population** — every node sits in a
  near-null density void; **there is no FMIA density bridge / Informational Parallel**
  connecting the nodes in this population. Energy/interference channels are at null.
- **One marginal thread:** `gen15` (iso_surv 0.38; D=4.96, η=0.079, ρ_vac=0.10, ω0=2.0,
  a_coupling=0.16, s=0.739, f=−0.294, a=0.297) clears the independence floor
  (phase_coupling 0.784 > floor-p95 0.73; action-rate 0.629). Marginal, single config — a
  thread to probe with a sharper estimator, **not** a result.

## 6. Honest interpretation

- The bounded stable-node population is best described as **collective density-threshold
  condensates**: the cluster matters (isolated seed weaker), but there is **no clean,
  separable inter-node transfer above null** in the energy, interference, or geometric-corridor
  channels. The phase channel shows a weak, mostly-non-separable signal.
- This does **not** falsify FMIA transfer. Two caveats cut against any negative claim:
  1. the diagnostic floors are high — the estimators are not yet sharp enough to detect weak
     transfer;
  2. this population conspicuously **lacks density bridges** (conductance ≈ 0). FMIA transfer
     *requires* a low-resistance corridor to route through. The current hunt objective
     (bounded multi-node + collective + ablation) does **not** select for that structural
     precondition — so absence of transfer here may reflect what was searched for, not the
     theory.

## 7. Recommended objective update (for decision)

Add the **structural precondition for transfer** to the fitness, then re-hunt:

- reward configs that form an **inter-node density / low-Ω² corridor** (conductance > 0), and
- reward **above-independence-floor** phase/current/energy coupling measured *along* that
  corridor (node-interaction-graph edges), not just node count + collectiveness.

Open sub-decisions: (a) sharpen the phase-coupling estimator to lower its independence floor
before trusting it; (b) probe the single `gen15` above-floor candidate at higher fidelity;
(c) search specifically for bridge-forming configs (the current population has none).

---

## 8. gen15 deep-dive (calibration thread) — 2026-06-20, `transfer_deepdive.py`

Probed the single above-floor candidate (`GEN15_DEEPDIVE_20260620_113052`) for robustness,
fidelity, and causal routing. **Result: near-negative / marginal — not a positive control.**

- **Robustness (fragile):** above the 0.73 floor only for the baseline seed + 1 of 3 altered
  seeds (others 0.58, 0.65) and for small `D`/`ω0` perturbations; small `s`/`f`/`a_coupling`
  perturbations collapse it to ~0.3. 4/12 runs above floor, but only 1/3 *altered seeds*.
- **Fidelity (decisive failure):** at 1600 steps (vs 800) phase coupling collapses
  **0.784 → 0.284**, nodes 4→3, conductance → 0. The coupling is a finite-window effect that
  does **not** persist under longer evolution.
- **Corridor perturbation (inconclusive):** other-node response to a node-kick is tiny (~1 %)
  and the lag-to-peak is pinned at the window boundary (identical 600 for node- and void-kick)
  = unresolved chaotic divergence, not a resolved propagation delay. Ratio 1.85 reflects
  energy/sensitivity near the node, not structured routing.

**Parametric clue:** the (transient) coupling lives on a narrow ridge — high D (~5), high
ω0 (2.0); governed by the nonlinearity terms `s`/`f`/`a` (which kill it when nudged) — and has
near-zero Ω² conductance even when "above floor." **Use gen15 as a marginal/negative control
for the bridge diagnostic, not a seed to chase.** This reinforces the structural reading: the
missing ingredient is an Ω² corridor / density bridge, which neither this population nor gen15
possesses.

---

## 9. Bridge-hunting objective (`fmia_transfer_score`) — built + calibrated 2026-06-20

`jax_scout/bridge_hunt.py`. Objective updated from node-survival to the FMIA structural
precondition + null-referenced transfer. **Staged** (cheap → expensive):

1. **Stage 1** (batched `sweep_probe`): hard gates finite/amp/energy-band[0.1,5]/curvature/
   multinode → reject unstable/energy/geometry runaway (instability cannot game the score).
2. **Stage 2** (final field only — cheap snapshot metrics): density-bridge conductance, Ω²
   corridor, J_info bridge flux, interference overlap between node centroids.
3. **Stage 3** (trajectory re-run — only for bridge-formers `max_cond>0.05`): full
   null-referenced temporal transfer (phase-coupling-above-floor / energy-exchange / action-rate).

`fmia_transfer_score = node_stability + energy_clean(er) · ( mean_cond + ½·max_cond +
½·interf_excess + 2·phase_excess_above_floor + 1.5·energy_exchange + ½·action_rate +
0.3·J_flux )`. `energy_clean` peaks at er=1 (energy-conservative stable collapse).
Reclassifies into {candidate_transfer_seed, bridge_no_transfer, marginal_phase_thread,
collective_density_threshold, no_corridor_stable_nodes} + reject classes.

**Two calibration findings (both fixed before any launch):**
- *Instability gaming* — rejected by the Stage-1 hard gates (44/80 LHS configs rejected).
- *Space-filling / trivial-J_flux gaming* — first calibration flagged 15/36 "transfer seeds",
  but they were inflated by `max_cond` (∝ node count; corr 0.58) and by `J_flux>0.12` (trivially
  true, not null-referenced). FIX: `candidate_transfer_seed` now requires a NULL-REFERENCED
  channel (phase-above-0.73-floor OR energy-exchange-excess); score uses `mean_cond` (naturally
  low for space-filling) with node-count moderation; energy-conservation factor discounts
  dissipating/growing configs. After the fix: 7/36 seeds, all null-referenced, top configs
  energy-conserved (er≈1.4–1.6).

**Key positive:** bridge-forming, transfer-capable configs DO exist in the raw parameter space
(e.g. a bounded er=1.51, 7-node config with `max_cond`=0.62) — the prior hunt selected against
them. Calibration runs: `sweep_runs/BRIDGE_CALIB_2026062*`. Governance gate still 17/17.
Next: launch the timeboxed bridge hunt; validate `candidate_transfer_seed` finalists at higher
fidelity + corridor-perturbation + CuPy (scout findings are not IRER evidence).

---

## 10. Bridge-hunt runs + runner-stability engineering (2026-06-20)

**Interim science (strongly positive).** A partial 3 h run (`BRIDGE_HUNT_20260620_161613`)
reached gen 10 / 480 evals before an OOM (below). The objective behaved exactly as intended:
- `best_fmia` climbed monotonically 1.94 → 2.30 → 2.58 → 2.67 → 2.74 across the 10 gens;
- `candidate_transfer_seed` accumulated to 73 (the evolutionary search concentrating on
  bridge-formers — the old population produced **zero**);
- top profile = bounded, **energy-conserved (er≈1.0)**, **phase coupling 0.76 > the 0.73
  independence floor**, with a real density bridge — the cleanest FMIA-transfer profile to date;
- `bridge_no_transfer` is the dominant accepted class — structure (bridges) forms faster than
  *confirmed null-referenced transfer*, which is the honest expected ordering.

This confirms the structural hypothesis: the missing ingredient was the OBJECTIVE, not the
model. Bridges + above-floor transfer are findable when explicitly rewarded.

**Runner stability — three failures, diagnosed and fixed (the science was never the problem):**
1. *gen-3 OOM* (first 3 h launch, `BRIDGE_HUNT_20260620_121115`): died at ~12 min. Cause:
   66 % of accepted configs triggered the Stage-3 trajectory capture; the elite-bred gen-3
   population was enriched for bridge-formers → too many GPU captures at once.
   FIX: cap Stage 3 to the global top-`STAGE3_TOPK` (=10) bridge-formers per generation
   (the structural Stage-2 pass is CPU/numpy and runs for all); N_SNAP 40→30.
2. *gen-11 OOM* (second launch, `BRIDGE_HUNT_20260620_161613`): died at ~56 min after 10 clean
   gens. Cause: **cumulative** GPU-memory growth — `XLA_PYTHON_CLIENT_PREALLOCATE=false` grows
   the allocation on every capture and never releases it. The card is an 8 GB GTX 1080 **shared
   with the Windows display and fragmented under WSL2**, so the real ceiling is ~2-3 GB
   *contiguous*, not 8 GB; ~100 captures over 10 gens hit it.
3. *allocator dead-ends:* `PREALLOCATE=true` (the obvious fix — one reused pool) FAILED the
   opposite way: it needs a single large *contiguous* block up front and got RESOURCE_EXHAUSTED
   at startup even with ~6 GB "free" (fragmentation + display sharing). RESOLUTION:
   `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (+ `PREALLOCATE=false`) — cudaMalloc/cudaFree on
   demand per op, no big contiguous grab AND releases memory back, so neither failure mode
   occurs. Slower per op, but the working set is ~0.3 GB. Validated by an accelerated stress
   test (`--stage3-topk 30`, ~120 cumulative captures > the gen-11 death point) that exited
   cleanly. Also: `gc.collect()` each generation.
4. *SyntaxError* (stress-test harness): misplaced `global` declaration; trivial, fixed.

   NOTE on measurement: free-VRAM readings are only meaningful at TRUE IDLE — a hung/failed
   process holds memory and contaminates the figure (a "2.3 GB free" reading was entirely a
   stuck process; true idle was 6.3 GB free). Do not size pools from a reading taken while
   anything is running.

Also added: per-config `try/except` in `evaluate` (one bad config can't kill a generation),
a top-level generation guard that writes `CRASH.txt` and stops cleanly (no more silent deaths),
and a `--stage3-topk` override to stress-test the GPU-memory fix at 3× load.

**Decisions recorded:** the silent gen-11 death sat undetected for ~3.5 h because the run was
trusted to send a completion notification, which an OOM/abort does not. Going forward, liveness
is verified via `status.json` mtime, and the runner is engineered to always exit cleanly
(timebox or caught crash) so completion always notifies.

**Status:** GPU-memory fix under accelerated-stress validation; full 3 h hardened run to be
relaunched once it passes the gen-11 point. `candidate_transfer_seed` finalists (bounded,
energy-conserved er≈1, phase>floor or energy-exchange, real bridge) then go to higher-fidelity
+ corridor-perturbation + CuPy validation — scout findings remain not-IRER-evidence on their own.

---

## 11. Full bridge-hunt COMPLETED (`BRIDGE_HUNT_20260620_180938`) — 2026-06-20

34 generations, 1632 evals, 3.09 h, **clean exit** (platform-allocator fix held throughout).
`best_fmia` converged early (2.74 by gen 8, plateau 2.749). Instability rejected throughout
(112 energy_runaway + 127 unstable + 54 geometry = 293). Accepted 1339.

**Transfer-class outcome:** `bridge_no_transfer` 929, **`candidate_transfer_seed` 269**,
`no_corridor_stable_nodes` 141. The search produced a large bridge-forming population — vs the
original objective's population which had **no Ω² corridors at all**.

**Ranked by null-referenced evidence (NOT the Jflux-inflated fmia, which let high-J_flux configs
top the raw ranking — J_flux is not null-referenced):**
- 45 / 269 seeds clear the phase-coupling independence floor (>0.73);
- 155 show energy-exchange excess > 0.25;
- **20 satisfy BOTH independent null-referenced channels** (phase > 0.73 AND exchange > 0.20);
- **10 multi-channel finalists** also have a strong bridge (maxCond > 0.2) and are energy-conserved
  (er∈[0.7,1.5], curv < 0.5). Saved to `finalists.json`.

Top finalists are **7-node, bounded (curv≈0.03), energy-conserved (er 1.14–1.50)**, with
**strong density bridges (maxCond 0.6–0.73)**, phase coupling 0.73–0.785 (above floor), and
energy-exchange 0.25–0.32 — the FMIA Informational-Parallel profile the prior objective never
produced.

**Parametric signature of the bridge-transfer basin** (consistent across finalists): high
diffusion D≈5 (builds inter-node density bridges), slight anti-damping gain η≈−0.2 (sustains the
standing structure), minimal geometric coupling a_coupling=0.1 (the bound floor), negative
quintic/cubic (s, a < 0). Physically coherent: high D is exactly what was missing — it forms the
inter-node corridors the node-survival objective suppressed.

**Honest caveats:** (1) SCOUT-level (JAX, N=48) — NOT IRER evidence; CuPy/ASTE validates. (2)
phase_coup max 0.808 is only modestly above the 0.73 floor — the strongest evidence is the
AGREEMENT of two independent null-referenced channels on the same 20 configs, not any single
metric's margin. (3) J_flux (~2) is large but not null-referenced; excluded from primary evidence.

**Next (validation tier):** take the 10 multi-channel finalists (esp. gen18 maxCond=0.73 / gen17
maxCond=0.64+exch=0.315 / gen34) to: higher fidelity + longer trajectory; corridor-perturbation
causal test (kick one node, measure bounded delayed response routed along the bridge); seed/
perturbation robustness (as in the gen15 deep-dive); then CuPy/ASTE FP64 confirmation. Only a
finalist that survives ALL of these is an IRER-transfer candidate.

---

## 12. Validation tier on top-3 finalists (`VALIDATION_20260620_213438`) — 2026-06-20

Top 3 frozen (`frozen_finalists.json`): gen18 `8e638162144d`, gen14 `a1d411f36fdd`,
gen34 `675663b645da` (all 7-node, scout pcoup 0.76-0.78, exch 0.25-0.30, bridge 0.68-0.73).

**Step 2 — high-fidelity JAX (N=48/1600, +2 altered seeds, 6 param perturbations, N=64 grid):
the TRANSFER signal does NOT survive longer trajectory.** All 3 baselines: phase coupling
collapses 0.78 → **0.27-0.39 (below the 0.73 floor)** at 1600 steps; energy-exchange drops below
its scout value; node count coarsens 7 → 4. (Phase > floor reappears for *one* altered seed each
— seed-fragile, not robust.) This is the same finite-window fragility that sank gen15. Per the
Step-2 failure conditions ("phase coupling falls below floor", "energy exchange not above null"),
all three **FAIL high-fidelity → verdict NONREPRODUCING_REJECT** for the transfer claim.

**What IS robust:** the density bridges / Ω² corridors persist at 1600 steps, at N=64, across
altered seeds, and under every parameter perturbation (maxCond 0.26-1.14, meanCond 0.15-0.53).
The *structural* precondition is real and reproducible; only the (correlational, null-referenced)
transfer metrics along it are a finite-window transient.

**Step 3 — corridor perturbation:** the first pass looked strongly positive (bridge-kick →
10-14× the void-kick other-node response, resolved lag, recovery) BUT the kicks were not
energy-matched (a multiplicative kick in the dense bridge injects more energy than in a void), so
the ratio is confounded by injection. An **energy-matched re-test** (`corridor_retest.py`,
closed-form equal-ΔE kicks) is required to decide whether the causal routing is real or an
injection artifact. [result appended below]

**Honest verdict (JAX tier):** the bridge objective robustly produces the STRUCTURE (Ω² corridors
/ Informational Parallels), but the null-referenced TRANSFER signal is a narrow finite-window
(~800-step) transient that does not survive to 1600 steps — answering the "robust physics vs
scout artifact vs transient" question as: structure = robust; correlational transfer = transient.
No finalist is promoted to CuPy on these metrics. The open question is the energy-matched causal
routing test.

### Corridor-perturbation causal tests (3 variants; `corridor_retest.py`, `corridor_phasetest.py`)
- **Un-matched amplitude kick:** bridge/void = 10-14× — but CONFOUNDED (dense bridge injects more
  energy than a sparse void).
- **Energy-matched amplitude kick** (`corridor_retest_energymatched.json`): bridge/void = 1.17 /
  0.45 / 2.19 — the 10-14× was injection, not routing. (And the void control breaks: matching
  energy in an empty void needs eps 1.0-1.8 → a giant new blob.)
- **Energy-conserving phase kick** (`corridor_phasetest.json`, the clean probe): perturbing the
  STRUCTURE (node OR bridge) propagates to the other nodes 3-9× more than perturbing the void
  (node/void 3.1-7.4, bridge/void 2.7-9.1) — so the multi-node configuration IS genuinely COUPLED
  (not independent condensates). BUT node ≈ bridge, so the bridge is **not a SELECTIVE routing
  corridor**; and the response is boundary-pinned (still growing at +1200 steps) → dynamics
  unresolved in-window. `phase_routing_positive = 0/3`.

### FINAL validation verdict (JAX tier) — top 3 finalists
**NOT VALIDATED_FMIA_TRANSFER_CANDIDATE.** The specific FMIA claim (information routed
*selectively* along Ω² corridors) is not supported: correlational transfer metrics are a
finite-window transient (fail at 1600 steps), and the bridge is not a selective routing corridor
(node ≈ bridge under the clean phase kick). What IS robustly established and is a real advance over
the original independent-condensate result: the bridge objective produces **bounded multi-node
structures with persistent Ω² density bridges that are genuinely COUPLED** (structure perturbations
propagate 3-9× more than void perturbations). So: `coupled-structure = robust`,
`selective-FMIA-transfer = not validated`. Nothing escalates to CuPy. NOT a global falsification of
IRER (scout-only, N=48, limited 1200-step perturbation window, density+phase kicks only) — but the
current finalists are coupled-structure candidates, not transfer candidates.

**Caveats / what could change this:** (1) the perturbation window (1200 steps) is too short to
resolve the response peak/recovery — a longer window might reveal slower selective routing; (2)
only density + single-θ phase kicks were tried; (3) the coupling that DOES exist (3-9× over void)
is real and uncharacterised — worth studying on its own terms (it is NOT the independent-condensate
regime). Next, if pursued: longer-window corridor dynamics, and/or characterise the inter-node
coupling channel directly rather than as bridge-selective routing.

---

## 13. Coupling characterization (Stages 1-2) + A-field audit — 2026-06-20

**Stage 1 (longer-window phase kick, 4000 steps; `corridor_longwindow.py`):** node/void responses
RESOLVE (peak ~1800-2400 steps, partial recovery); bridge kick is still boundary-pinned and only
1.1-1.3× node → not a faster/selective channel. **The no-corridor negative control responds ≈ 0
(30-100× smaller than bridged finalists)** → the propagation is real and bridge-associated, NOT a
trivial "matter vs void" effect. Verdict: COUPLED_STRUCTURE_NONSELECTIVE.

**Stage 2 (response-matrix characterization, 2800 steps; `coupling_characterize.py`):** the
coupling is **GLOBAL / COLLECTIVE, not pairwise** —
- `global_mode_fraction ≈ 0.89` (gen18 0.886, gen34 0.893); pairwise_fraction ≈ 0.11;
- row_pattern_similarity ≈ 0.57 (kicking any node → similar response pattern);
- node_bridge_selectivity 1.30-1.41 (bridge not special);
- amplitude_scaling_exponent ≈ 1.06-1.08 (LINEAR response);
- response_time_to_peak ~2600 steps (slow; no in-window decay);
- phase_current_response_gain 7.0-8.7 and omega_response_gain 0.72-0.80 (perturbations strongly
  excite J_info and Ω², but collectively);
- negative control structure_response_gain ≈ 0.0000 (confirms bridge-association).

**Mechanistic conclusion:** the γ_A=0 finalists are a **holistic, globally-coupled web**, not
directed FMIA wires — exactly what coupling to a SCALAR (density ρ, directionless) predicts. The
quantitative signature is global_mode_fraction ≈ 0.89.

**A-field audit (`afield_prototype.py`):** the existing causal A-field is **scalar, density-sourced**
(`d²A/dt² = -c²k²A + ρ_k`) with **isotropic scalar coupling** (`ρ_vac_eff = ρ_vac+γ_A·A` or
`Ω²_eff = Ω²·exp(γ_A·A)`). It does NOT couple to the current J_info = ρ∇φ, so it cannot create
direction. Production `unified_omega.py` is likewise pure-scalar (grep: no affect/current terms).
Testing it = `A_SCALAR_RESCUE_TEST` (`afield_rescue_test.py`, γ_A sweep on gen18/14/34), NOT a
current-coupled test. [scalar rescue result appended below]

**The theory-required mechanism** (current/"rate of interaction" → directed wires) is specified in
`docs/AFIELD_CURRENT_COUPLED_RFC.md`: a vector potential A sourced by the transverse part of
J_info, coupling to ψ by minimal coupling (∇→∇−iγ_A A), contract-separated, default-off. To be
designed-then-implemented (not hacked); success = global_mode_fraction drops, selectivity rises,
phase coupling persists to 1600, routing becomes resolved+bridge-selective, all bounded.

**Scalar rescue test result (`afield_scalar_rescue.json`):** 0/18. Scalar A has **negligible
effect** — γ_A=0 ≈ γ_A=0.2 on every metric (gen18 pcoup 0.342→0.335, bridge 0.205→0.204), phase
coupling stays collapsed (~0.25-0.36 < floor). Confirms the audit: scalar density-sourced A cannot
rescue or add direction.

### Current-coupled A-field — IMPLEMENTED + tested (2026-06-20)
`physics.n_op/step` gained optional `a_vec` (minimal coupling `−2i a·∇ψ − |a|²ψ`, a=γ_A·A);
`jax_scout/afield_current_coupled.py` evolves the vector A from transverse J_info. **γ_A=0
reproduces baseline bit-for-bit (rel_L2 = 0.00e+00)** — correct + CuPy↔JAX equivalence intact.

Sweep on gen18/14/34 (γ_A {0,.02,.05,.1,.2,.5}, 1600 steps; `afield_current_coupled.json`):
- **Real γ_A-dependent effect** (unlike scalar's flat response): pcoup MOVES with γ_A — rises
  gen18 (0.342→0.402 by γ_A=0.2), falls gen34 (0.249→0.197), non-monotonic gen14. So current
  coupling genuinely engages the dynamics.
- **But NO clean rescue:** no finalist clears the 0.73 floor at any bounded γ_A (best gen14 0.642
  at γ_A=0.5). The γ_A=0.5 "gains" are DISTORTION not wires — exch→0, bridge conductance jumps to
  ~1.0, A-energy balloons (×30), energy grows (er~3).
- **Confound:** the finalists are NOT energy-conserved at 1600 steps (er 2.2-3.2; the er≈1.5 was
  an 800-step value), so the structure drifts regardless of A.

**Verdict (v1, JAX scout):** current coupling is mechanistically active (real effect, unlike
scalar) but the minimal-coupling form at this κ/c_A does NOT rescue long-lived transfer above the
floor. Per RFC classes: gen18/gen14 = A_CURRENT_PARTIAL, gen34 = A_CURRENT_NO_EFFECT. NOT a
falsification of the current-coupled idea: κ (source strength) and c_A (speed) are untuned, the
global_mode_fraction (the direct web→wires metric) was not yet measured under A-on, and the
1600-step energy drift muddies the test. Open follow-ups: global_mode under A-on; κ/c_A tuning;
whether the γ_A=0.5 bridge→1.0 regime is routing or space-filling distortion.

### Stage 1 tuning + Stage 2 A-coupled hunt (2026-06-20/21)
**Stage 1** (`afield_current_tune.py`, γ_A×κ×c_A on gen18): no clean regime — EVERY combo flagged
distortion because er≈3.16 *including γ_A=0 baseline*. The γ_A=0 finalists are energy-unstable by
1600 = the WRONG SUBSTRATE. κ≥4 + slow c_A → non-finite (vector-wave runaway). Recorded
`CURRENT_A_V1_CALIBRATION_NO_RESCUE_ON_GAMMA0_FINALISTS` (not a theory failure).

**Stage 2** (`afield_bridge_hunt.py`, 11-D search base+γ_A+κ+c_A, A-coupled, 1600-step eval, 30
gen / 720 evals / 3.04h; `AF_BRIDGE_HUNT_20260621_060714`): gates work (458 distortion-rejects,
mostly energy_unbounded). **Mixed result, reported honestly:**
- Found 6 `A_PERSISTENT_TRANSFER_CANDIDATE` (energy-bounded@1600 + pcoup>floor) — but they are
  MARGINAL (pcoup 0.733-0.759, right at the 0.73 floor, within noise) and BRIDGE-LESS
  (0.005-0.133). Not directed wires.
- Strong-bridge + strong-A-localization configs (bridge 0.68, A_loc up to 12) have pcoup BELOW
  floor (0.59). The two properties (bridge+A-channel vs persistent-coupling) did NOT co-occur.
- **Novel positive:** the current-coupled A STRONGLY localizes on the bridge corridor (A_loc up to
  12, vs scalar A which saturated the web) — qualitative wire-forming, as predicted.
- The decisive test (does A-localization actually drop global_mode_fraction = web→wires?) is run
  at the validation tier (`afield_validate.py`) on the best wire-candidates. [result appended]

**Web→wires validation (`afield_validation.json`) — MARGINAL, NOT ROBUST (1/3):**
- gen18 (γ_A=0.23): global_mode_fraction 0.90 → **0.73** under A-on (drop 0.17) = the predicted
  web→wires shift — but still predominantly global (pairwise only 0.27) and absolute response tiny.
- gen20: 0.61 → 0.65 (no shift). gen29: 0.61 → **1.00** (ANTI-shift — A-on drove a near-perfect
  global mode, response ×800). So current coupling can make the web MORE holistic, not less.

**FINAL verdict on the current-coupled A-field (JAX scout):** the mechanism is genuinely ACTIVE
(real γ_A effect, strong bridge A-localization A_loc up to 12 — both unlike scalar A), but the
v1 minimal-coupling form did NOT robustly produce directed FMIA wires under the tested parameter
ranges: phase coupling persisted only marginally and bridge-less; bridge + A-localization +
persistence never co-occurred; and the web→wires structural shift appeared in only 1 of 3
candidates (and reversed in another). Recorded as: *"Minimal current-coupled A-field v1 engages
the dynamics but did not yet produce stable directed FMIA wires under tested parameter ranges."*
Per the standing rule this is NOT a theory falsification — it suggests the formalism needs
refinement (how the rate-of-interaction field should couple to phase-current / Payan-state
alignment / manifold directionality), and remains JAX-scout-only (no CuPy current-coupled term;
nothing promotes).
