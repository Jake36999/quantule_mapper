# IRER Simulation Guide — 2025-05-19

Source: `F:\transcripts\2025_05_May\20250519_105705_IRER Simulation Guide.txt` | sha256/16: `ab3df1acb5841797` | total lines: 1123 (the triage register records ~1,811; sha256 matches byte-for-byte — the triage script counts lone-CR terminators, ~688 here, as extra line breaks) | **Disposition: fully preserved** (Jake's turns verbatim and complete; the two methodology documents — the pasted v7 User Guide and the AI's IRER Supplementary Guide — quoted in full; every named artifact in the v7 code preserved verbatim (Rho1DSimulator, run_fft_analysis, analyze_splash_propagation_stats, the Small-Win/Minimal-Validation `__main__`) or, for analysis helpers whose algorithm matches earlier dossiers, as signature + behaviour with declared delta markers; the v8 `analysis_extensions_v8.py` reproduced verbatim)
Families: **F6** (prime-harmonic resonance · `k_peak ≈ ln 2` targeting · spectral matching), **F2** (Resonance Density ρ · collapse/RFD · splash propagation / R90), **F4** (chorotic boundary dynamics · entropic-load regulation) | Streams: physics (application/method), provenance, project_report

**Why this conversation matters:** (1) This is the **post-Declaration (May 19) methodology consolidation** of the whole simulation programme — the era's clearest statement of *why* the 1-D ρ code is run and *how its outputs map onto IRER hypotheses*. Jake pastes the "**1D Rho Simulation & Analysis Script (v7)** User Guide" and asks the AI to write a **supplementary guide directing the code toward IRER goals**; the AI's reply is a research-agenda map (IRER theme → concrete 1-D observable → configuration) that is the single best surviving methodology document of this stream. (2) It carries the **v7 simulator rewrite** — a substantive [REVISION of the May-16 `Rho1DSimulator`]: the earlier random-source term is replaced by **logistic growth** `k·ρ·(1−ρ/ρ_max)`, RNG moves to `np.random.default_rng(global_seed)`, and a full **SHA-256 fingerprinting / run-manifest** reproducibility layer is added. (3) It formalises the **"Small Win" experiment**: a **D-sweep to find the diffusion coefficient whose dominant spatial radian wave-number `k_peak` is closest to `ln(2) ≈ 0.693`** — the operational form of the log-prime resonance hypothesis (targeting the log of the first prime), plus the **R90** splash-propagation radius and the **Influence Hotspot** collapse-pair metric, each with explicit PASS/FAIL criteria. (4) It records the **v8 forward plan being partly implemented** (`analysis_extensions_v8.py`: a unified `select_fft_peak` and an `AnalysisRegistry` plug-in system). **Era/authorship notes:** the guide uses **PAS = Phase Alignment Score** ("Collapse Duality & PAS thresholds") — the mid-May reading, *not* v9 Appendix A's "Potential Actualization State"; the AI's guide cites uploaded reference files via `citeturn0file1` (IRER development memos) and `citeturn0file3` ("v3 alpha pipeline review"), and the v7 code's own comments ("as in v6", "from user's v1IRERsimcode.txt") mark an **author-steering-AI-code / AI→AI lineage** — Jake iterating an AI-authored codebase across versions, not dictating physics for scratch implementation. The `k_peak ≈ ln 2` target is the log-prime resonance apparatus **later NULLED as H1 (0/60)** in `docs/IRER_MASTER_HYPOTHESIS_CATALOG.md`; recorded here as methodology from the START of that arc, not an endorsed result. **No simulations are executed in-transcript** — the file is guide + code, so all numbers here (`ln 2`, `D_sweep` range, `k_err<0.05`, R90 target) are **design targets and criteria, not run outputs**; there are **no FFT peaks / SSE / α results**.

---

## Segment 1 — lines 5–287 — `application_method` / `project_report` (Jake pastes the v7 User Guide and asks for an IRER-directed supplement)

> ### User:
> hey, could you create a supplimenary guide to this guide that helps direct towards IRER goals: User Guide: 1D Rho Simulation & Analysis Script (v7)
> This guide provides instructions on how to set up, run, and interpret the outputs from the 1D Rho Simulation and Analysis Python script. The script is designed for exploring a reaction-diffusion system with collapse events, a "splash" mechanism, and various analysis tools.

The pasted v7 User Guide is preserved in full below (Jake's turn; the two most boilerplate sub-blocks are declared-omitted). The theory/method-bearing structure:

> 1. Setup and Prerequisites
> 1.1. Environment
> This script is designed to run in a Python environment, ideally within a Jupyter Notebook or Google Colab for interactive use and visualization.
>
> 1.2. Required Libraries
> Ensure you have the following Python libraries installed. You can typically install them using pip:
> numpy · pandas · matplotlib · scipy (for FFT and peak finding in analysis) · pywavelets (for CWT analysis) · tqdm (for progress bars)
> It's highly recommended to use a requirements.txt file to manage these dependencies for reproducible environments, especially for archival runs. […]
>
> *[lines 30–52 omitted: `requirements.txt` version-pin template and Google-Colab Drive-mount boilerplate — non-theory-bearing setup; the `gdrive_base_folder_path` variable and its timestamped-master-directory behaviour are re-stated in the code (Segment 3).]*
>
> 2. Understanding the Script Structure
> The Python script is organized into several main sections:
> SECTION 1: IMPORTS AND INITIAL SETUP: Loads necessary libraries and attempts to configure BLAS/MKL thread settings for reproducibility.
> SECTION 1.5: UTILITY FUNCTIONS FOR FILE FINGERPRINTING: Contains helper functions to generate SHA256 checksums for output files and run manifests, aiding in data integrity verification.
> SECTION 2: UTILITY FUNCTIONS (apply_splash): Defines the apply_splash function, which models the redistribution of rho after a collapse.
> SECTION 3: SIMULATION OUTPUT PLOTTING: Contains plot_simulation_output for generating basic visualizations like rho evolution heatmaps and final distributions.
> SECTION 4: Rho1DSimulator CLASS: The core of the simulation. It handles the initialization, stepping through time (including diffusion, growth, dissipation, collapse, and splash), and saving primary simulation data.
> SECTION 5: ANALYSIS FUNCTIONS: A suite of functions to analyze simulation outputs (e.g., analyze_spatial_autocorrelation, analyze_collapse_neighborhood, run_fft_analysis).
> SECTION 5.1: ANALYSIS FUNCTIONS FOR "SMALL WIN" EXPERIMENT: Specific analysis helpers like analyze_splash_propagation_stats.
> SECTION 6: MAIN EXECUTION BLOCK (if __name__ == "__main__":): This is where you configure and launch simulation runs. It defines:
> A unique RUN_UID and GLOBAL_SEED for the entire script execution to ensure reproducibility.
> The master_output_directory.
> base_params_general_runs for general-purpose simulations.
> The "Small Win" experiment setup, including the D-sweep and the "Minimal Validation Grid" parameter sweep.
>
> 3. Running Simulations
> All simulation runs are configured and initiated from the SECTION 6: MAIN EXECUTION BLOCK.
> 3.1. Key Global Settings at the Start of __main__
> GLOBAL_SEED: An integer used to seed all random number generation within the script for a given execution. By default, it's set to a fixed value (e.g., 42) for consistent debugging or can be derived from RUN_UID for unique-yet-reproducible full script runs. Changing this seed will result in different outcomes if any stochastic processes are active (e.g., initial_factor > 0 with non-custom IC, or stochastic_influx_magnitude > 0).
> VERBOSE_FINGERPRINT: Set to True or False to control whether messages about SHA256 fingerprint generation are printed.
> gdrive_base_folder_path: Set this to your target Google Drive path (see section 1.3).
>
> *[lines 92–139 preserved as method: "3.2 Running General Purpose Simulations" gives the copy-`base_params_general_runs` → modify → filter `SIMULATOR_CORE_PARAMS` → instantiate `Rho1DSimulator(**sim_init_args)` → `sim.run(verbose, plot_output, run_output_dir, verbose_fingerprint)` → dump `run_full_parameters.json` + `save_fingerprint` → call analysis functions recipe. All parameter/API details recur verbatim in the code (Segment 3).]*
>
> 3.3. Running the "Small Win" Experiment
> This experiment is pre-configured in Section 6. It first performs a D-sweep to find a diffusion coefficient D that results in a dominant spatial radian wave-number (k_peak) close to ln(2). Then, it uses this "best D" to run detailed simulations (splash ON and OFF) and a "Minimal Validation Grid" to test specific hypotheses about R90 and collapse influence.
>
> Primary Configuration Variables (at the start of the "Small Win" block):
> SW_GRID_SIZE_N: Grid size for these experiments (e.g., 128).
> SW_SOURCE_K: Growth rate k_s.
> D_sweep_values: NumPy array defining the range of Diffusion coefficients to test (e.g., np.round(np.logspace(np.log10(0.03), np.log10(0.5), num=10), 3)).
> SW_TIMESTEPS: Number of timesteps for these runs.
> T_START_FFT_ANALYSIS: Timestep from which to start averaging rho for FFT analysis to avoid initial transients.
> SW_K_TARGET_FOR_WINDOW: The target k_r (e.g., np.log(2)).
> SW_K_WINDOW_HALFWIDTH: The search window around the target k_r for the "smarter peak picking" in the FFT. Set SW_K_TARGET_FOR_WINDOW = None to disable windowed search and use the global maximum power peak.
>
> Execution: Simply run the entire script cell. The "Small Win" experiment will execute automatically.
>
> Minimal Validation Grid: After the D-sweep and "best D" determination, this section runs a predefined matrix of simulations varying splash_fraction, refractory_period, and dissipation_c to try and meet the R90 and Influence Hotspot targets. You can modify the validation_grid_params list to test different combinations.
>
> 4. Interpreting Outputs
> For each simulation run (whether general purpose, a D-sweep point, or a validation grid point), a dedicated subdirectory is created. […]
> 4.1. Key Files in Each Run's Output Directory:
> simulation_core_parameters.json … simulation_summary.json (total_collapses, avg_rho_final, max_rho_overall …) … run_full_parameters.json … rho_history.npy (full spatio-temporal evolution of ρ) … collapse_events.csv … collapse_details_log.csv … sim_plot_....png … *.sha256sum files … run_manifest.sha256sum … analysis_outputs/ or analysis_k_lambda/ subdirectories (spatial_autocorr_metrics...json · avg_collapse_neighborhood_data...json · collapse_influence_counts...npy · spatial_fft_slice..._data.csv [frequency and power for FFT] · splash_propagation_metrics...json [contains R90]).
>
> 4.2. Specific Outputs for "Small Win" Experiment:
> In the SmallWin_N..._UID.../ directory:
> k_peak_vs_D_sweep.csv: Shows D, calculated k_peak, and λ_peak for each point in the D-sweep.
> k_peak_vs_D_plot.png: Visualizes k_peak vs. D against your target k_r.
> Subdirectories for each D-sweep run (e.g., DSweep_D0.030/).
> Subdirectories for the "Best D" runs (e.g., BestD0.030_SplashON_Detailed/).
> In the MinimalValidationGrid_D..._UID.../ directory:
> minimal_validation_grid_results.csv: Summarizes k_peak, R90, and influence hotspot metrics for each parameter combination tested.
> Console Output: The script prints a summary of the "Small Win Criteria Check" indicating PASS/FAIL for k_peak error, R90, and influence hotspot for the "best D" runs. […]
>
> 5. Customization and Further Experiments
> Parameters: The primary way to customize is by modifying the parameter dictionaries in __main__ (e.g., base_params_general_runs, exp_base_params_sw, or creating new ones).
> Initial Conditions: For random ICs: Set initial_factor to a value > 0 (e.g., 0.1) … For deterministic zero IC: Set initial_factor = 0.0. For custom ICs: Prepare a NumPy array of the desired initial ρ values and pass it to sim.run(custom_initial_rho=my_ic_array).
> Analysis: You can choose which analysis functions to call after each simulation run in __main__ […] (e.g., history, events, the full parameter dictionary for the run, and global_seed_for_sampling=GLOBAL_SEED for functions like analyze_collapse_neighborhood).
>
> 6. Ensuring Reproducibility
> GLOBAL_SEED: This is the cornerstone. A simulation run with the same GLOBAL_SEED and identical parameters should produce identical numerical results and output files.
> RUN_UID: Helps distinguish outputs from different overall script executions.
> SHA256 Fingerprints & Manifests: Use these to verify that output files have not changed and that all expected files are present for a given run. […]
> Environment: For the highest level of reproducibility […] Use a fixed Python environment (e.g., via conda environments and a requirements.txt or environment.yml file to lock library versions). Be aware that underlying math libraries (BLAS/MKL) can sometimes introduce tiny floating-point variations across different CPUs or library builds, though the script attempts to mitigate this by setting thread counts to 1. Containerization (e.g., Docker) is the most robust solution for cross-system bit-for-bit reproducibility.
> […]

**Notes:** This is the **v7 methodology as Jake framed it**, and it is theory-bearing in three ways. (a) It states the **"Small Win" objective in plain terms**: find the diffusion `D` whose dominant spatial radian wave-number `k_peak` sits closest to **`ln(2)`** — i.e. tune the reaction–diffusion length-scale until the ρ-field's characteristic mode matches the log of the first prime. This is the log-prime resonance hypothesis (H1) recast as a *length-scale-matching* experiment, and it introduces the **windowed "smarter peak picking"** (search a band `k_r ± halfwidth` rather than taking the global FFT maximum) — a methodological choice that materially shapes what counts as a "match" (relevant when reading H1's later NULL). (b) It elevates **reproducibility to a first-class concern** (GLOBAL_SEED, RUN_UID, SHA-256 fingerprints, run manifests, thread-pinning, Docker) — the archival-integrity posture that persists into the current project's evidence packages. (c) It names the **R90** (90th-percentile splash-propagation radius) and **Influence Hotspot** observables as the splash-mechanism's success metrics. The guide is itself an AI-authored artifact Jake is pasting for further AI work (its "v7", "as in v6" lineage) — author-steering-AI-code.

---

## Segment 2 — lines 289–391 — `application_method` / `project_report` (the AI's IRER Supplementary Guide — methodology centerpiece) [ADDENDUM: feeds unified-IRER-paper Sections 6/10]

The AI's reply is the era's key methodology-consolidation document: it maps IRER themes onto concrete 1-D observables and prescribes experiment templates, integrity practices, and interpretation rules. Preserved in full (verbatim, including the AI's `citeturn0fileN` reference tokens).

> ### Supplementary Guide  
> **Linking the 1-D Rho Simulation & Analysis Script (v7) to the broader IRER research agenda**
>
> #### 0  Purpose of this add-on  
> The v7 user-guide teaches you how to run the code; the notes below explain **why** to run it, **which runs matter most for IRER**, and **how to tag, archive and interpret results** so they feed straight back into outstanding theory questions highlighted in the latest IRER development memos citeturn0file1.
>
> #### 1  IRER research questions the 1-D code can probe
>
> | IRER theme | Concrete observable in the 1-D model | Where to measure / how to configure |
> |------------|--------------------------------------|--------------------------------------|
> | **Collapse Duality & PAS thresholds** | Frequency-distribution of `collapse_events`; slope-change of `rho` just before/after threshold crossing | Vary `rd_threshold`, `collapse_reset_value`; sample `rho_history` slices at T-10…T+10 timesteps for 1000 random collapses |
> | **Resonance picks / prime-indexed modes** | Dominant spatial FFT peaks versus ln(prime) target list | Use `run_fft_analysis`, but swap in **robust peak selector** recommended in the *v3 alpha pipeline review* citeturn0file3; log best-fit α scale |
> | **Entropic-load regulation & splash propagation** | (a) Δρ in neighbourhood plot, (b) cascade size‐distribution from influence heat‐map | Enable/disable `splash_enabled`; sweep `splash_radius`, `splash_fraction` |
> | **Chorotic boundary dynamics** | Emergence/decay rate of coherent ρ domains (autocorr dip/peak lag evolution) | Track `analyze_spatial_autocorrelation` at 5–10 equally spaced times |
> | **Law-emergence / recursive survivability** | Run-to-run stability of summary metrics when only stochastic seed changes | Keep all params fixed, vary `GLOBAL_SEED`; compare metric variance vs. parameter-change variance |
>
> #### 2  Experiment design templates
>
> 1. **Threshold-ladder test**  
>    *Aim*: map critical PAS values.  
>    *Grid*: `rd_threshold` in {0.55, 0.60 … 0.80}, `collapse_reset_value` in {0.05, 0.10}.  
>    *Outputs*: collapse count, mean inter-collapse interval, power-law exponent of cascade sizes.
>
> 2. **Prime-resonance calibration sweep** (replacement for the "DSweep" in Small-Win)  
>    *Grid*: keep D constant at candidate "best-D", vary `grid_size` (64–512) to shift FFT resolution, then brute-select top-5 FFT peaks with **prominence filter**, not simple max.  
>    *Goal*: minimise SSE to {ln 2, ln 3, ln 5}. Record α* and SSE to `prime_fit.csv`.
>
> 3. **Splash-influence decay mapping**  
>    *Grid*: splash_radius = 1-6; splash_fraction = 0.05-0.4.  
>    For each run export the `collapse_influence_counts` matrix and fit an exponential/ stretched-exp to counts versus dx. Store τdx and τdt in a summary table.
>
> 4. **Seed-diversity reproducibility panel**  
>    Fix "nominal" parameter set. Run N = 30 seeds. Store distribution of all headline metrics. Compare coefficient of variation (CV) to same-size sample across parameter variants. High CV ≈ weak law-selection; low CV strengthens IRER's "deterministic resonance" claim.
>
> #### 3  Metadata & integrity checklist (add these to SECTION 6)
>
> | Item | Minimal patch |
> |------|---------------|
> | **RUN_UID** | add suffix `_IRERexp[shortcode]` (e.g. `_THRladder`) |
> | **Theory tags** | new list `irer_tags = ["PAS", "PrimeFFT"]` etc.; save to `run_full_parameters.json` |
> | **Metric manifest** | after each analysis block append a single row to `/metrics_summary.csv` (one file per master_output_directory) |
> | **Hash of code state** | call `save_fingerprint(__file__)` right after imports so manifest proves code identical across seeds |
>
> #### 4  Interpreting results against the open-questions list
>
> * **Prime-match improves after splash?**  
>   Unexpected → revisit assumption that non-local redistribution merely adds "noise". May indicate multi-scale resonance coupling (Advanced concept 7.7) citeturn0file1.
>
> * **Autocorr dip stabilises at lag ≈ grid/16 across seeds**  
>   → candidate emergent length-scale; compare to λ* predicted by `k_peak ≈ ln 2` criterion.
>
> * **Cascade exponent ≤ −1.5 only when refractory ON**  
>   Supports idea that local de-excitement time acts like "informational viscosity" (parallels VT/TFT). Flag for Section 4.2 modelling.
>
> Capture these observations directly in a notebook or markdown log; each bullet should reference run UID + metric values so later synthesis scripts can auto-pull evidence.
>
> #### 5  Recommended next code upgrades (for v8)
>
> 1. **Unified peak-selector utility**: guarantees identical logic for parameter-sweep summary and single-run deep dives (fixes discrepancy in pdf report) citeturn0file3.  
> 2. **`AnalysisRegistry` class**: plug-in style, so future IRER-specific analyses (e.g., χ² fit to prime ladder, entropy-flux estimator) live beside generic ones.  
> 3. **YAML experiment manifest**: readable spec of sweeps; script autogenerates runs + captures provenance.
>
> #### 6  Where each student / collaborator can contribute
>
> | Track | Skills needed | Example task |
> |-------|---------------|--------------|
> | **Code infra** | Python packaging, CI | Implement `AnalysisRegistry`; add pytest with seed-fixtures |
> | **Data science** | NumPy/Pandas | Build dashboard to visualise CV across seeds vs. params |
> | **Theory** | Applied maths, complex systems | Derive expected cascade exponent from IRER gradient-drive toy model |
> | **Documentation** | Scientific writing | Convert this supplement + v7 guide into Sphinx site with cross-links to IRER white-paper sections |
> | **Repro-ops** | Docker/Conda | Freeze env.yml + publish container image for archival runs |
>
> #### 7  Citing & archiving results back into the master IRER doc set
>
> *For each figure or CSV you intend to quote in Section 6 or 10 of the unified IRER paper*:
> 1. Copy file under `/evidence/<section-id>/<RUN_UID>/`.  
> 2. Add SHA256 to the per-section manifest.  
> 3. In the LaTeX source cite as `\cite{RUN_UID_fft}` etc.—keeps provenance chain unbroken.
>
> ### Final tips
> *Start small, log everything, and always map a measurement back to an explicit IRER hypothesis or open question.*  
> Doing so turns raw simulation play into incremental theory-validation work—exactly what Sections 10.1–10.4 of the master framework call for citeturn0file1.

**Notes:** The methodology centerpiece, and the richest single mapping of **IRER concept → measurable 1-D observable** in the code era:
- **"Collapse Duality & PAS thresholds"** → collapse-event frequency distribution + pre/post-threshold ρ slope. Here **PAS = Phase Alignment Score** (mid-May reading); `rd_threshold` is treated as the tunable "critical PAS value". This is the terminology fork flagged in `D20250517_135856` (v9 Appendix A later uses PAS = *Potential Actualization State*).
- **"Resonance picks / prime-indexed modes"** → dominant spatial FFT peaks vs the `ln(prime)` list, with an explicit prescription to **minimise SSE to {ln 2, ln 3, ln 5}** and **record α\* and SSE to `prime_fit.csv`** (Template 2). This is the α/SSE calibration target stated as a *plan* — no such run exists in-transcript, and the whole prime-mode programme is what H1 (0/60 NULL) later falsified. Recorded as the START of that arc.
- **"Entropic-load regulation & splash propagation"** (F2/F4) → neighbourhood Δρ + cascade size-distribution; the splash mechanism as the code embodiment of "entropic-load redistribution".
- **"Chorotic boundary dynamics"** (F4) → autocorr dip/peak-lag evolution as the emergence/decay rate of coherent ρ domains ("chorotic boundary" is the F4-family term for domain edges).
- **"Law-emergence / recursive survivability"** → seed-diversity CV as a test of IRER's **"deterministic resonance"** claim (low run-to-run variance ⇒ law-selection). This is the era's operational definition of testing determinism.
- Cross-refs: **"Advanced concept 7.7"** points back to the 90+-item concept census in `D20250517_135856` (multi-scale resonance coupling); **"VT/TFT"** and "informational viscosity" are F4/F6 constructs; the `citeturn0file1` / `citeturn0file3` tokens are the AI's citations to two uploaded reference files then in context — *file1* = "latest IRER development memos" (the master-doc lineage), *file3* = "v3 alpha pipeline review" (an earlier AI review of the FFT peak-picker) — provenance breadcrumbs for the May-19 working set. Section 7's "archive results back into Sections 6/10 of the unified IRER paper" is an explicit **[ADDENDUM]**-style workflow feeding simulation evidence into the Declaration-lineage doc set (the Appendix-D-bound evidence practice). No numeric results — all targets (`ln 2`, `ln 3`, `ln 5`, thresholds, CV) are design values.

---

## Segment 3 — lines 393–1087 — `application_method` (Jake pastes the full v7 notebook code) [REVISION of `Rho1DSimulator` — May-16 random-source → v7 logistic-growth]

> ### User:
> # %% [markdown]
> # # 1D Rho Simulation and Analysis Framework
> #
> # This notebook implements a 1D reaction-diffusion model with collapse events,
> # a "splash" mechanism, logistic growth, and stochastic influx.
> # […]
> # Key Enhancements in this Version (v7 - Minimal Validation Grid):
> # - Centralized random number generation using a global seed.
> # - Vectorized collapse detection loop.
> # - SHA-256 fingerprinting for key JSON and CSV output files and run manifests.
> # - tqdm progress bar.
> # - Implements the "Minimal Validation Grid" experiment to tune R90 and Influence Hotspot
> #   metrics while maintaining the k_peak target.
> #
> # This is the code the original doc is talking about

Jake pastes the complete v7 notebook (SECTIONS 1–6). The named artifacts, with their fate:

**`apply_splash` (v7) — verbatim** (a leaner signature than the May-16 version; returns `np.clip(rho_splashed, 0, None)`):

```python
def apply_splash(rho, events_details, radius, fraction, kernel, sigma):
    rho_splashed = rho.copy(); size = len(rho)
    for t, collapsed_idx, collapsed_amount_val in events_details:
        amount_to_splash = collapsed_amount_val * fraction
        if amount_to_splash > 0 and radius > 0:
            indices_to_splash = []; weights = []
            for offset in range(-radius, radius + 1):
                if offset == 0: continue
                neighbor_idx = (collapsed_idx + offset + size) % size 
                indices_to_splash.append(neighbor_idx)
                if kernel == 'gaussian' and sigma > 0: weights.append(np.exp(-(offset**2) / (2 * sigma**2)))
                else: weights.append(1.0)
            if indices_to_splash and sum(weights) > 0:
                normalized_weights = np.array(weights) / sum(weights)
                for i, target_idx in enumerate(indices_to_splash): rho_splashed[target_idx] += amount_to_splash * normalized_weights[i]
    return np.clip(rho_splashed, 0, None)
```

**`Rho1DSimulator` (v7) — verbatim** (the substantive [REVISION]: **logistic growth** `k·ρ·(1−ρ/ρ_max)` replaces the May-16 random-source term; `np.random.default_rng(global_seed)`; optional `stochastic_influx_magnitude`; vectorised collapse via `np.where`; a `collapse_details_log`; and `run()` now writes `rho_history.npy`, JSON parameter/summary files, `collapse_events.csv`, plots, and SHA-256 fingerprints/manifests):

```python
class Rho1DSimulator: # Full class definition as in v6
    def __init__(
        self, size: int = 100, timesteps: int = 500, diffusion: float = 0.08,
        source_k: float = 0.18, dissipation_c: float = 0.002, rho_max: float = 1.0,
        stochastic_influx_magnitude: float = 0.0,
        threshold: float = 0.7, reset_val: float = 0.05,
        initial_factor: float = 0.1, global_seed: int = None, 
        splash_enabled: bool = True, splash_radius: int = 3, splash_fraction: float = 0.2,
        splash_kernel: str = 'uniform', splash_sigma: float = 1.5,
        refractory_enabled: bool = False, refractory_period: int = 5,
        run_name_suffix: str = "" ):
        self.p = { 'size': size, 'timesteps': timesteps, 'diffusion': diffusion, 'source_k': source_k, 
                   'dissipation_c': dissipation_c, 'rho_max': rho_max, 'stochastic_influx_magnitude': stochastic_influx_magnitude,
                   'threshold': threshold, 'reset_val': reset_val, 'initial_factor': initial_factor, 
                   'global_seed': global_seed, 'splash_enabled': splash_enabled, 'splash_radius': splash_radius,
                   'splash_fraction': splash_fraction, 'splash_kernel': splash_kernel, 'splash_sigma': splash_sigma,
                   'refractory_enabled': refractory_enabled, 'refractory_period': refractory_period, 
                   'run_name_suffix': run_name_suffix }
        if self.p['global_seed'] is None:
            ts_pid_seed = int(datetime.now().timestamp() * 1e6 + os.getpid()) % (2**32) 
            self.rng = np.random.default_rng(ts_pid_seed)
        else: self.rng = np.random.default_rng(self.p['global_seed'])
        self.rho = None; self.history = np.array([]); self.events = []; self.collapse_details_log = []; self.timer = None
    def initialize(self, custom_initial_rho: np.ndarray = None): # Logic as in v6
        if custom_initial_rho is not None:
            if len(custom_initial_rho) == self.p['size']: self.rho = custom_initial_rho.copy()
            else: raise ValueError(f"Custom initial_rho length error for {self.p.get('run_name_suffix')}.")
        else: self.rho = self.rng.random(self.p['size']) * self.p['initial_factor']
        self.history = np.zeros((self.p['timesteps'], self.p['size']))
        self.events = []; self.collapse_details_log = []
        if self.p['refractory_enabled']: self.timer = np.zeros(self.p['size'], dtype=int)
    def step(self, t: int): # Vectorized logic as in v6
        if self.rho is None: print("Error: Rho not initialized."); return
        if self.p['refractory_enabled'] and self.timer is not None: self.timer[self.timer > 0] -= 1
        lap = np.roll(self.rho,1) + np.roll(self.rho,-1) - 2*self.rho; diffusion_term = self.p['diffusion']*lap
        k=self.p['source_k']; cap=self.p['rho_max']; gf=np.zeros_like(self.rho)
        if cap > 1e-9: m=self.rho<cap; gf[m]=(1-self.rho[m]/cap)
        lg_term = k*self.rho*gf; decay_term = self.p['dissipation_c']*self.rho; sto_term=0.0
        if self.p['stochastic_influx_magnitude'] > 0: sto_term = self.p['stochastic_influx_magnitude']*self.rng.random(self.p['size'])
        new_rho = np.clip(self.rho + diffusion_term + lg_term - decay_term + sto_term, 0, None)
        potential_coll_idx = np.where(new_rho >= self.p['threshold'])[0]; splash_inputs = []
        if len(potential_coll_idx) > 0:
            if self.p['refractory_enabled'] and self.timer is not None: actual_coll_idx = potential_coll_idx[self.timer[potential_coll_idx] == 0]
            else: actual_coll_idx = potential_coll_idx
            if len(actual_coll_idx) > 0:
                for idx_c in actual_coll_idx:
                    rho_pre = new_rho[idx_c]; splash_inputs.append((t,idx_c,rho_pre))
                    self.collapse_details_log.append({'time':t,'index':idx_c,'rho_before_reset':rho_pre,'rho_after_reset':self.p['reset_val']})
                    self.events.append((t,idx_c))
                new_rho[actual_coll_idx] = self.p['reset_val']
                if self.p['refractory_enabled'] and self.timer is not None: self.timer[actual_coll_idx] = self.p['refractory_period']
        if self.p['splash_enabled'] and splash_inputs:
            new_rho = apply_splash(new_rho, splash_inputs, self.p['splash_radius'], self.p['splash_fraction'], self.p['splash_kernel'], self.p['splash_sigma'])
        self.rho = np.clip(new_rho, 0, self.p['threshold']*1.5) 
        if t < self.p['timesteps']: self.history[t] = self.rho
```

The v7 `run()` method (lines 628–649) drives the loop (with an optional `tqdm` bar), then persists `simulation_core_parameters.json`, `rho_history.npy`, `collapse_events.csv`, `collapse_details_log.csv`, `simulation_summary.json` (`total_collapses / avg_rho_final / max_rho_overall / simulation_parameters`), the plots, and a `generate_run_manifest_fingerprint` — returning `(history, events, summary_dict)`. *[`run()`/`summary()` bodies preserved as behaviour — I/O + fingerprinting plumbing around the `step` dynamics above.]*

**`run_fft_analysis` (v7) — NEW named artifact, verbatim** (unifies spatial/temporal FFT and adds the **`k_peak`/`λ_peak`** calculation with windowed peak-picking around a target `k_r` — the operational core of the "Small Win" `k_peak ≈ ln 2` test):

```python
def run_fft_analysis(rho_data_history, p_sim, mode='spatial', cell_or_time_idx=None, analysis_output_dir=".", calculate_k_lambda=False, spatial_slice_for_k_lambda=None, k_target_for_window=None, k_window_halfwidth=None, verbose_fingerprint=False):  # Full logic as in v6
    if rfft is None or rfftfreq is None : return (None,None,None,None) if calculate_k_lambda else (None,None)
    os.makedirs(analysis_output_dir, exist_ok=True)
    title_suffix = p_sim.get('run_name_suffix', ''); fn_suffix = "".join(c if c.isalnum() else "_" for c in title_suffix).strip("_").replace(" ", "_")
    data_slice=None; dx_interval=1.0; axis_label=""; plot_title=""; filename_base=""
    if mode=='spatial':
        slice_desc="N/A"
        if calculate_k_lambda and spatial_slice_for_k_lambda is not None: data_slice=spatial_slice_for_k_lambda; slice_desc="custom_slice_for_k_lambda"
        else: 
            idx=cell_or_time_idx if cell_or_time_idx is not None else p_sim['timesteps']//2; slice_desc=f"t{idx}"
            if not(0<=idx<rho_data_history.shape[0]): return (None,None,None,None) if calculate_k_lambda else (None,None)
            data_slice=rho_data_history[idx,:]
        axis_label='Spatial Freq ($f_k$, cycles/cell)'; plot_title=f'Spatial FFT Power Spectrum ({slice_desc}){title_suffix}'; filename_base=f"spatial_fft_slice{slice_desc}{fn_suffix}"
    elif mode=='temporal':
        if calculate_k_lambda: return None,None,None,None
        idx=cell_or_time_idx if cell_or_time_idx is not None else p_sim['size']//2
        if not(0<=idx<rho_data_history.shape[1]): return None,None
        data_slice=rho_data_history[:,idx]; axis_label='Temporal Freq ($f_t$, cycles/timestep)'; plot_title=f'Temporal FFT ({idx}){title_suffix}'; filename_base=f"temporal_fft_c{idx}{fn_suffix}"
    else: return (None,None,None,None) if calculate_k_lambda else (None,None)
    N=len(data_slice)
    if N<2: return (None,None,None,None) if calculate_k_lambda else (None,None)
    data_detrended=data_slice-np.mean(data_slice); yf=rfft(data_detrended); xf=rfftfreq(N,d=dx_interval); ps=(np.abs(yf)**2)/N
    plt.figure(figsize=(10,5)); plt.plot(xf,ps); plt.title(plot_title); plt.xlabel(axis_label); plt.ylabel('Power'); plt.grid(True,ls='--',alpha=0.7)
    if xf.size>0: plt.xlim(left=xf[0],right=xf[-1])
    plt.savefig(os.path.join(analysis_output_dir,f"{filename_base}_spectrum.png")); plt.close()
    data_path = os.path.join(analysis_output_dir, f"{filename_base}_data.csv")
    pd.DataFrame({'frequency_cycles_per_unit': xf, 'power': ps}).to_csv(data_path, index=False)
    save_fingerprint(data_path, verbose_fingerprint)
    k_peak, lambda_peak = None, None
    if mode=='spatial' and calculate_k_lambda:
        xf_ndc=xf[1:]; ps_ndc=ps[1:] 
        if len(ps_ndc)>0:
            peak_idx_target=-1 
            if k_target_for_window is not None and k_window_halfwidth is not None:
                f_low=(k_target_for_window-k_window_halfwidth)/(2*np.pi); f_high=(k_target_for_window+k_window_halfwidth)/(2*np.pi)
                win_idx=np.where((xf_ndc>=f_low)&(xf_ndc<=f_high))[0]
                if len(win_idx)>0: peak_idx_target=win_idx[np.argmax(ps_ndc[win_idx])]
            final_peak_idx_in_ps_ndc = peak_idx_target if peak_idx_target!=-1 else np.argmax(ps_ndc)
            f_peak=xf_ndc[final_peak_idx_in_ps_ndc] 
            if f_peak>1e-9: k_peak=2*np.pi*f_peak; lambda_peak=1.0/f_peak
        return xf, ps, k_peak, lambda_peak
    return xf, ps
```

**`analyze_splash_propagation_stats` (v7) — NEW named artifact, verbatim** (defines **R90** = 90th-percentile radial distance of triggered collapses from the seed cell — the splash-propagation success metric):

```python
def analyze_splash_propagation_stats(events, seed_cell_index, p_sim, analysis_output_dir=".", verbose_fingerprint=False): # Full logic as in v6
    os.makedirs(analysis_output_dir, exist_ok=True)
    run_suffix = p_sim.get('run_name_suffix', ''); fn_suffix = "".join(c if c.isalnum() else "_" for c in run_suffix).strip("_").replace(" ", "_")
    if not events: print(f"SplashProp ({fn_suffix}): No events."); return None, None
    grid_size = p_sim['size']; radial_distances = []
    if len(events) <= 1: R90 = 0.0 if len(events) == 1 else None; max_prop_radius = 0.0; num_secondary = 0
    else:
        for i in range(1, len(events)): 
            t, idx = events[i]
            direct_dist = np.abs(idx - seed_cell_index); periodic_dist = grid_size - direct_dist
            radial_distances.append(min(direct_dist, periodic_dist))
        if not radial_distances: R90=0.0; max_prop_radius=0.0; num_secondary = 0
        else:
            radial_distances=np.array(radial_distances); R90=np.percentile(radial_distances,90); max_prop_radius=np.max(radial_distances)
            num_secondary = len(radial_distances)
            # … histogram of radial distances vs kernel radius saved to splash_propagation_hist{fn_suffix}.png …
    prop_metrics = {'R90':R90, 'max_propagation_radius':float(max_prop_radius), 'num_secondary_collapses':num_secondary}
    # … saved to splash_propagation_metrics{fn_suffix}.json + fingerprint …
    return R90, radial_distances
```

> [lines 469–510, 661–827 preserved as signature + behaviour — the reproducibility and analysis helpers whose algorithms match earlier dossiers or are non-theory-bearing I/O: `save_fingerprint(file_path, verbose)` and `generate_run_manifest_fingerprint(run_output_dir, verbose)` (SHA-256 per-file checksums + an overall run manifest — the integrity layer); `plot_simulation_output` (v7 heatmap + final-distribution, as in earlier dossiers); `analyze_spatial_autocorrelation` (v7 — same `np.correlate` + `find_peaks` dip/peak-lag logic as `D20250516_011140` Segment 4, now writing `spatial_autocorr_metrics.json` + fingerprint); `analyze_collapse_neighborhood` (v7 — same ±half-size window averaging, now `p_sim`-driven with `global_seed_for_sampling` and JSON output); `analyze_collapse_influence` (v7 — same `(Δt,Δx)` histogram2d as `D20250516_011140` Segment 4, now returning the full `counts` matrix and saving `collapse_influence_counts.npy`); `analyze_cwt_of_collapse_counts` (v7 — PyWavelets `'morl'`, `np.geomspace(min_p,max_p,n_scales)`, scalogram output — the PyWavelets sibling of the routines in `D20250516_010307` / `D20250516_011140` Segment 10).]

**`__main__` (SECTION 6) — the Small-Win + Minimal-Validation experiment, physics-bearing config preserved verbatim:**
- Global: `RUN_UID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")`, `GLOBAL_SEED = 42`, `VERBOSE_FINGERPRINT = True`; outputs under a timestamped `IRER_SimData_UID_{RUN_UID}` master directory (Google-Drive path with local fallback).
- `base_params_general_runs`: `size=100, timesteps=250, diffusion=0.08, source_k=0.18, dissipation_c=0.002, rho_max=1.0, threshold=0.7, reset_val=0.05, initial_factor=0.0, splash_radius=3, splash_fraction=0.2, splash_kernel='gaussian', splash_sigma=1.5, refractory_enabled=False, refractory_period=5, stochastic_influx_magnitude=0.0`, plus analysis knobs (`analysis_neighborhood_time_before=10 / _after=5`, `analysis_influence_max_dx=10 / _max_dt=25`, `cwt_wavelet='morl', cwt_min_period=5, cwt_max_period=50, cwt_num_scales=60`).
- **Small Win block:** `SW_GRID_SIZE_N=128`, `SW_SOURCE_K=0.18`, `D_sweep_values = np.round(np.logspace(np.log10(0.03), np.log10(0.5), num=10), 3)`, `SW_TIMESTEPS=300`, `T_START_FFT_ANALYSIS=150`, **`SW_K_TARGET_FOR_WINDOW = np.log(2)`** (≈0.6931 rad/cell), `SW_K_WINDOW_HALFWIDTH=0.15`. Each D-sweep run uses a **deterministic seed IC** — a flat `reset_val` field with a single seed cell at `size//2` set to `threshold` — runs the simulator, then computes `k_peak` from the time-averaged (`t ≥ 150`) ρ slice via `run_fft_analysis(..., calculate_k_lambda=True, k_target_for_window=ln2, k_window_halfwidth=0.15)`; results tabulated to `k_peak_vs_D_sweep.csv` and plotted vs the `ln 2` target. The **"best D"** minimises `|k_peak − ln 2|`; it is re-run **splash ON** (→ `analyze_splash_propagation_stats` for R90, `analyze_collapse_influence`) and **splash OFF** (baseline influence).
- **Small Win Criteria Check** (PASS/FAIL thresholds, verbatim): `k_peak` error `< 0.05` ⇒ WIN; `|R90 − splash_radius| ≤ 1` ⇒ WIN; Influence Hotspot: splash `(counts[0,2]+counts[0,3])` `> 5×` no-splash hotspot (or no-splash = 0) ⇒ WIN.
- **Minimal Validation Grid:** at the best D, a 4-row grid `{A: sf=0.05, rp=0, dc=0.002; B: sf=0.05, rp=10, dc=0.002; C: sf=0.05, rp=10, dc=0.003; D: sf=0.10, rp=10, dc=0.003}` (`sf`=splash_fraction, `rp`=refractory_period, `dc`=dissipation_c), each run splash-ON and splash-OFF, recording `k_peak / R90_splash / hotspot_splash / hotspot_nosplash` to `minimal_validation_grid_results.csv`.

**Notes:** The v7 `Rho1DSimulator` is a genuine physics [REVISION]: the growth term is now **logistic** (`k·ρ·(1−ρ/ρ_max)`, carrying capacity `ρ_max=1.0`) rather than the May-16 uniform-random source — a design decision that gives the field a stable non-zero fixed point and makes the "characteristic length-scale" (hence `k_peak`) well-defined, which is precisely what the Small-Win `k_peak ≈ ln 2` test requires. The **Small Win** experiment is the log-prime hypothesis in its most operational May form: *not* "do FFT peaks equal `log(prime)`?" but "can we tune `D` so the dominant spatial mode `k_peak` **lands on `ln 2`**, within a ±0.15 window, with error < 0.05?" — a length-scale-matching restatement. The **windowed peak-picking** (search a band around `ln 2` rather than the global maximum) is a methodological choice worth flagging when reading H1's eventual NULL (0/60): it biases toward finding a peak near the target. The deterministic single-seed IC (flat field + one supra-threshold cell) turns each run into a controlled splash-propagation experiment (hence R90). All values here are **design/criteria** — the script is pasted, not run; no `k_peak`, R90, SSE, or α outputs exist in-transcript. Authorship: the code's own comments ("Full logic as in v6", "as in v6") mark it as an evolved AI-authored artifact Jake is carrying forward — author-steering-AI-code (AI→AI across v6→v7).

---

## Segment 4 — lines 1089–1121 — `application_method` (AI emits `analysis_extensions_v8.py`: the v8 upgrade begun)

> ### ChatGPT:
> {"name": "analysis_extensions_v8.py", "type": "code/python", "content": "…"}

The AI implements two of the three "v8" recommendations from its own Supplementary Guide (§5): a **unified FFT peak-selector** and a **plug-in `AnalysisRegistry`**. Reproduced verbatim (JSON `\n` escapes rendered as line breaks; text unaltered; long docstrings trimmed to their signatures with a declared marker):

**`select_fft_peak` — the unified peak-selector (IRER-consistent windowed rule), verbatim:**

```python
def select_fft_peak(
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    target_k: Optional[float] = None,
    window_halfwidth: Optional[float] = None,
    min_prominence: float = 0.01,
) -> Tuple[int, float, float]:
    """Return (idx, k_peak, lambda_peak) according to IRER-consistent rules. […]"""
    if freqs.ndim != 1 or power.ndim != 1 or freqs.shape != power.shape:
        raise ValueError("freqs and power must be 1-D arrays of equal length")

    # Discard the DC component (freq == 0) when searching for peaks.
    freqs_ndc, power_ndc = freqs[1:], power[1:]
    if freqs_ndc.size == 0:
        raise ValueError("FFT spectrum too short to locate peaks")

    # Generic peak detection.
    peak_idx_candidates, props = find_peaks(power_ndc, prominence=min_prominence)
    if peak_idx_candidates.size == 0:
        # Fall back to absolute maximum (after DC) if no prominent peaks.
        peak_idx = int(np.argmax(power_ndc))
    else:
        # If a target_k was supplied, prefer peaks inside the window.
        if target_k is not None and window_halfwidth is not None:
            f_low = (target_k - window_halfwidth) / (2 * np.pi)
            f_high = (target_k + window_halfwidth) / (2 * np.pi)
            in_window = np.where((freqs_ndc[peak_idx_candidates] >= f_low) & (freqs_ndc[peak_idx_candidates] <= f_high))[0]
            if in_window.size > 0:
                # Choose the most powerful peak inside the window.
                best_local = in_window[np.argmax(power_ndc[peak_idx_candidates[in_window]])]
                peak_idx = int(peak_idx_candidates[best_local])
            else:
                # No candidate in window – fall back to most powerful overall.
                peak_idx = int(peak_idx_candidates[np.argmax(power_ndc[peak_idx_candidates])])
        else:
            peak_idx = int(peak_idx_candidates[np.argmax(power_ndc[peak_idx_candidates])])

    f_peak = freqs_ndc[peak_idx]
    k_peak = 2 * np.pi * f_peak
    lambda_peak = 1.0 / f_peak if f_peak > 0 else np.inf
    # Return index relative to original arrays (include DC shift of +1)
    return peak_idx + 1, k_peak, lambda_peak
```

**`AnalysisRegistry` — the plug-in registry for post-simulation analyses, verbatim** (decorator-registration + `run` / `run_all` / `available`):

```python
class AnalysisRegistry:
    """Central registry for post-simulation analysis call-backs. […]"""

    def __init__(self) -> None:
        self._analyses: Dict[str, Callable] = {}

    def register(self, name: str) -> Callable[[Callable], Callable]:
        """Decorator to register a function under *name*."""
        def decorator(func: Callable) -> Callable:
            if name in self._analyses:
                raise KeyError(f"Analysis '{name}' already registered – choose a unique name.")
            self._analyses[name] = func
            return func
        return decorator

    def run(self, name: str, /, **kwargs):
        if name not in self._analyses:
            raise KeyError(f"Analysis '{name}' not found. Available: {list(self._analyses)}")
        return self._analyses[name](**kwargs)

    def run_all(self, /, **kwargs) -> Dict[str, object]:
        """Run *every* registered analysis with the same **kwargs."""
        results = {}
        for k, fn in self._analyses.items():
            sig = inspect.signature(fn)
            bound = sig.bind_partial(**kwargs)
            bound.apply_defaults()
            results[k] = fn(*bound.args, **bound.kwargs)
        return results

    def available(self) -> List[str]:
        return list(self._analyses)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AnalysisRegistry n={len(self._analyses)}>"
```

The module closes with a worked example registering `k_peak_basic` (an `analysis_k_peak_basic(history, params, *, window_start=0)` that time-averages ρ from `window_start`, FFTs it, and calls `select_fft_peak(..., target_k=params.get("target_k"), window_halfwidth=params.get("window_halfwidth"))` → `{"k_peak", "lambda_peak"}`). *[Example body preserved as behaviour; it is a thin wrapper demonstrating the registry.]*

> ### ChatGPT:
> I've added **analysis_extensions_v8.py** to the canvas. It gives you a drop-in **`select_fft_peak`** utility (shared by both sweep summaries and deep-dives) plus a minimal **`AnalysisRegistry`** plug-in system […]. Key things to try next: **Swap in the new peak selector** in `run_fft_analysis` […]; **Register custom analyses** […]; **Populate the registry** with the IRER-specific analyses you already have (`analyze_splash_propagation_stats`, etc.) […]

**Notes:** The v8 upgrade begun. `select_fft_peak` factors the exact windowed-peak logic out of the v7 `run_fft_analysis` into one canonical utility — its purpose (per the guide §5.1) is to *fix a discrepancy* between the sweep-summary and deep-dive peak choices flagged in the "v3 alpha pipeline review" (`citeturn0file3`). This is a small but consequential methodological move for the log-prime programme: it standardises how a "peak near `ln 2`" is selected. `AnalysisRegistry` is the plug-in scaffold meant to host future IRER-specific analyses (the guide names "χ² fit to prime ladder, entropy-flux estimator"). Both are first-and-only versions in this file, preserved verbatim. No execution; no numeric output.

---

## Dossier summary

**What this file is, theoretically:** the **May-19 methodology consolidation** of the IRER 1-D simulation programme. Jake pastes the v7 User Guide and asks for an IRER-directed supplement; the AI's **Supplementary Guide** (Segment 2) is the era's definitive map from IRER concepts (Collapse Duality / PAS thresholds, prime-indexed resonance modes, entropic-load/splash propagation, chorotic boundaries, law-emergence) to concrete 1-D observables and experiment templates, with integrity practices and an evidence-archiving workflow. Jake then pastes the **v7 code** (Segment 3), whose `Rho1DSimulator` is a logistic-growth [REVISION] of the May-16 simulator and whose `__main__` runs the **"Small Win"** experiment — a D-sweep to place the dominant spatial wave-number `k_peak` on **`ln(2)`**, plus **R90** and **Influence Hotspot** criteria — and a Minimal Validation Grid. The AI closes by beginning the **v8** upgrade (`analysis_extensions_v8.py`).

**Named artifacts (and their fate):**
- `Rho1DSimulator` (v7) — **verbatim; [REVISION]** (logistic growth `k·ρ·(1−ρ/ρ_max)`, `default_rng(global_seed)`, vectorised collapse, fingerprinting). The physics-of-record for this dossier.
- `run_fft_analysis` (v7) — **verbatim; NEW** (unified spatial/temporal FFT + windowed `k_peak`/`λ_peak`).
- `analyze_splash_propagation_stats` (v7) — **verbatim; NEW** (R90 = 90th-percentile propagation radius).
- `apply_splash` (v7) — **verbatim** (leaner signature, `clip(…,0,None)`).
- `select_fft_peak`, `AnalysisRegistry` (v8) — **verbatim; NEW** (`analysis_extensions_v8.py`).
- `save_fingerprint`, `generate_run_manifest_fingerprint`, `plot_simulation_output`, `analyze_spatial_autocorrelation`, `analyze_collapse_neighborhood`, `analyze_collapse_influence`, `analyze_cwt_of_collapse_counts` (v7) — preserved as signature + behaviour, with declared deltas to the equivalent routines in `D20250516_011140` / `D20250516_010307`.

**Hypothesis-status flag (no overclaiming):** the **Small Win** `k_peak ≈ ln 2` objective and the guide's "minimise SSE to {ln 2, ln 3, ln 5}" template are the log-prime resonance hypothesis (H1) in operational form — **H1 later NULLED (0/60)** per `docs/IRER_MASTER_HYPOTHESIS_CATALOG.md`. This dossier records the **methodology at the start of that arc**, not a positive result. The windowed peak-selection (`select_fft_peak`, `k_target_for_window`) should be read as a target-biased matching rule when weighing H1's NULL. The ρ-model is a toy 1-D reaction–diffusion system with **no matter/gravity claim**.

**First-ever FFT / SSE / α run outputs found:** **none.** The file is guide + pasted code + a v8 module; nothing is executed in-transcript. Every number present is a **design target or PASS/FAIL criterion** — `k_r = ln 2 ≈ 0.6931`, `k_window_halfwidth = 0.15`, `D_sweep = logspace(0.03→0.5, 10)`, `k_err < 0.05`, `R90 ≈ splash_radius ± 1`, Influence Hotspot `> 5×`, target SSE set {ln 2, ln 3, ln 5} — not a measured FFT peak, SSE, or α.

**v9-citation relevance:** a **scriptpt5**-lineage conversation (Python code for IRER, `Rho1DSimulator` + FFT stack), and its Supplementary Guide §7 explicitly prescribes archiving simulation evidence into "Sections 6/10 of the unified IRER paper" via per-section SHA-256 manifests — the Appendix-D-bound evidence practice. The `citeturn0file1` (IRER development memos) and `citeturn0file3` ("v3 alpha pipeline review") tokens document the May-19 working set of reference files then in the AI's context.

**Design vs implementation / authorship (v9 §4.4):** Jake's single substantive turn is a *steering request* ("create a supplementary guide … toward IRER goals") plus a paste of an AI-authored v7 script (its "as in v6" comments prove the AI→AI version lineage); the AI authors the Supplementary Guide, the v7 exposition, and the v8 module. This is **author-steering-AI-code**, not author-prose→AI-codes-from-scratch. Cross-authorship caveat (per sibling dossier `D20250521_012015`): the project's formal-equation apparatus (IQG master PDE etc.) and the "Ontological Informational Waves" label were later admitted partly third-party-AI-authored (Grok/Gemini) / AI-coined — neither is referenced in this file (which stays at the 1-D toy-model / methodology level), but the code-and-doc lineage pattern is consistent.

**Era note (naming):** **PAS = Phase Alignment Score** here ("Collapse Duality & PAS thresholds", `irer_tags=["PAS", …]`) — the mid-May IQG/PIF-era reading, *not* v9 Appendix A's "Potential Actualization State" (the unrecorded terminology fork logged in `D20250517_135856`). "Chorotic boundary", "VT/TFT", "informational viscosity", and "Advanced concept 7.7" tie this methodology to F4/F6 constructs and to the 90+-item concept census of `D20250517_135856`.
