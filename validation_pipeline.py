#!/usr/bin/env python3

# --- VALIDATION PIPELINE EXECUTION GRAPH ---
# 1. ArtifactLoader              -> Loads HDF5 (psi_final, rho_final, multi-schema telemetry)
# 2. SpectralFidelityEngine      -> Runs CEPP v2.0 (prime_log_sse, bragg peaks)
# 3. ContractEnforcerEngine      -> Validates bounds against metric_contracts.yaml
# 4. Early Rejection Gate        -> Skips steps 5-10 if target_sse > 15.0
# 5. TopologyEngine              -> Runs TDA to classify field geometry
# 6. LOMTelemetryEngine          -> Extracts physical collapse events & gravity maps for fabrication
# 7. Falsifiability Tests        -> Executes Phase Ablation null tests
# 8. EmpiricalBridgeEngine       -> Computes JSA and C4 interference (Quantum Optics)
# 9. TensorValidationEngine      -> Checks symmetry and shear stress
# 10. StatisticalValidationEngine-> Runs Monte Carlo p-value checks
# 11. ProvenanceAssembler        -> Compiles strictly formatted provenance JSON
# -------------------------------------------

"""
validation_pipeline.py
ASSET: A6 (Spectral Fidelity & Provenance Module)
VERSION: 3.2 (Phase 3 Scientific Mandate - 100% ASTE Compliant)
CLASSIFICATION: Final Implementation Blueprint / Governance Instrument
GOAL: Serves as the immutable source of truth that cryptographically binds
      experimental intent (parameters) to scientific fact (spectral fidelity)
      and Aletheia cognitive coherence.
"""

import argparse
import json
import os
import sys
import gc
import math
import yaml
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple, cast

# --- Import analysis/validation modules ---
from config_utils import generate_canonical_hash
import tda_profiler
import metrics.collapse_dynamics as collapse_metrics
import metrics.monte_carlo_engine as monte_carlo_engine
import metrics.spdc_empirical_bridge as spdc_empirical_bridge
import metrics.tensor_validation as tensor_validation
import quantulemapper_real as cep_profiler
from orchestrator.diagnostics.runtime_audit import log_lifecycle_event

try:
    from scipy.stats import entropy as scipy_entropy
    from scipy.ndimage import gaussian_filter
except ImportError:
    print("FATAL: Missing 'scipy'. Please install: pip install scipy", file=sys.stderr)
    sys.exit(1)

import logging

def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger("validation_pipeline")

try:
    from orchestrator.run_identity import provenance_path_for_artifact
except Exception:  # pragma: no cover - fallback keeps validation importable standalone
    def provenance_path_for_artifact(output_dir, artifact_path, config_hash):
        return os.path.join(output_dir, f"provenance_{config_hash}.json")

SCHEMA_VERSION = "SFP-v3.2-ARCS"
MAX_ARTIFACT_ELEMENTS = int(os.environ.get("ASTE_MAX_ARTIFACT_ELEMENTS", str(512**3)))
ANTI_ALIAS_MAX_SOURCE_ELEMENTS = int(
    os.environ.get("ASTE_ANTI_ALIAS_MAX_SOURCE_ELEMENTS", str(MAX_ARTIFACT_ELEMENTS * 4))
)


# ==========================================
# STAGE 1: Artifact Loader
# ==========================================
class ArtifactLoader:
    @staticmethod
    def _anti_aliased_downsample(dataset: h5py.Dataset, stride: int, label: str) -> np.ndarray:
        size = int(dataset.size)
        effective_stride = max(1, int(stride))
        if size > ANTI_ALIAS_MAX_SOURCE_ELEMENTS:
            ratio = float(size) / float(ANTI_ALIAS_MAX_SOURCE_ELEMENTS)
            fallback_stride = int(np.ceil(ratio ** (1.0 / max(1, dataset.ndim))))
            effective_stride = max(effective_stride, fallback_stride)
            logger.warning(
                f"{label} dataset too large for full anti-aliased decimation ({size} elements). "
                f"Falling back to memory-safe strided read with stride={effective_stride}."
            )
            slices = tuple(slice(None, None, effective_stride) for _ in range(dataset.ndim))
            source = np.asarray(dataset[slices])
            sigma = 0.5
            if np.iscomplexobj(source):
                real_filtered = gaussian_filter(np.real(source), sigma=sigma, mode='nearest')
                imag_filtered = gaussian_filter(np.imag(source), sigma=sigma, mode='nearest')
                return real_filtered + 1j * imag_filtered
            return gaussian_filter(source, sigma=sigma, mode='nearest')

        source = dataset[()]
        sigma = max(0.5, 0.5 * float(effective_stride))
        if np.iscomplexobj(source):
            real_filtered = gaussian_filter(np.real(source), sigma=sigma, mode='nearest')
            imag_filtered = gaussian_filter(np.imag(source), sigma=sigma, mode='nearest')
            filtered = real_filtered + 1j * imag_filtered
        else:
            filtered = gaussian_filter(source, sigma=sigma, mode='nearest')

        slices = tuple(slice(None, None, effective_stride) for _ in range(dataset.ndim))
        return filtered[slices]

    @staticmethod
    def _adaptive_load_dataset(dataset: h5py.Dataset, label: str) -> np.ndarray:
        size = int(dataset.size)
        if size <= MAX_ARTIFACT_ELEMENTS:
            return dataset[()]

        ratio = float(size) / float(MAX_ARTIFACT_ELEMENTS)
        stride = int(np.ceil(ratio ** (1.0 / max(1, dataset.ndim))))
        stride = max(1, stride)
        slices = tuple(slice(None, None, stride) for _ in range(dataset.ndim))
        logger.warning(
            f"Large {label} field detected ({size} elements). "
            f"Applying anti-aliased downsample stride={stride} to prevent spectral aliasing."
        )
        return ArtifactLoader._anti_aliased_downsample(dataset, stride, label)

    @staticmethod
    def load(h5_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        logger.info(f"[Stage 1: ArtifactLoader] Loading artifact: {h5_path}")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Input file not found: {h5_path}")

        telemetry: Dict[str, Any] = {}
        with h5py.File(h5_path, 'r') as h5f:
            # 1. Resolve psi/rho schema drift
            psi_key = 'psi_final' if 'psi_final' in h5f else 'final_psi'
            if psi_key not in h5f:
                raise ValueError("No valid psi data found.")
            psi_final_3d = ArtifactLoader._adaptive_load_dataset(h5f[psi_key], "psi")

            rho_final_3d: Optional[np.ndarray] = None
            rho_key = 'rho_final' if 'rho_final' in h5f else 'final_rho'
            if rho_key in h5f:
                rho_final_3d = ArtifactLoader._adaptive_load_dataset(h5f[rho_key], "rho")
            elif 'rho_history' in h5f:
                rho_history = h5f['rho_history']
                if int(np.prod(rho_history.shape[1:])) > MAX_ARTIFACT_ELEMENTS:
                    logger.warning(
                        f"Large rho_history final frame detected ({int(np.prod(rho_history.shape[1:]))} elements). "
                        "Applying anti-aliased downsample to prevent spectral aliasing."
                    )
                    ratio = float(np.prod(rho_history.shape[1:])) / float(MAX_ARTIFACT_ELEMENTS)
                    stride = max(1, int(np.ceil(ratio ** (1.0 / max(1, len(rho_history.shape[1:]))))))
                    frame_slices = (rho_history.shape[0] - 1,) + tuple(slice(None, None, stride) for _ in rho_history.shape[1:])
                    final_frame = np.asarray(rho_history[frame_slices])
                    sigma = max(0.5, 0.5 * float(stride))
                    filtered = gaussian_filter(final_frame, sigma=sigma, mode='nearest')
                    rho_final_3d = filtered
                else:
                    rho_final_3d = rho_history[-1]

            if rho_final_3d is not None and psi_final_3d.shape != rho_final_3d.shape:
                raise ValueError(f"Domain mismatch: psi {psi_final_3d.shape} vs rho {rho_final_3d.shape}")

            # 2. V4.0 Decoupled Telemetry (Grouped)
            if 'extended_telemetry' in h5f:
                ext_grp = h5f['extended_telemetry']
                canonical_from_extended = {
                    'J_info_l2': 'j_info_l2_mean',
                    'grad_phase_var': 'grad_phase_var_mean',
                    'phase_coherence': 'phase_coherence_mean',
                    'omega_saturation': 'omega_sat_mean',
                }
                for src_key, dst_key in canonical_from_extended.items():
                    if src_key in ext_grp and len(ext_grp[src_key]) > 0:
                        data = np.asarray(ext_grp[src_key][:], dtype=np.float64)
                        telemetry[dst_key] = float(np.mean(data))
                        if src_key == 'J_info_l2':
                            telemetry['tau_c'] = ArtifactLoader._compute_tau_c(data)

            # 3. Legacy Schema Fallback (Flat Datasets)
            else:
                legacy_mappings = {
                    'j_info_l2_history': 'j_info_l2_mean',
                    'grad_phase_var_history': 'grad_phase_var_mean',
                    'phase_coherence_history': 'phase_coherence_mean'
                }
                for old_key, new_key in legacy_mappings.items():
                    if old_key in h5f and len(h5f[old_key]) > 0:
                        data = np.asarray(h5f[old_key][:], dtype=np.float64)
                        telemetry[new_key] = float(np.mean(data))
                        if old_key == 'j_info_l2_history':
                            telemetry['tau_c'] = ArtifactLoader._compute_tau_c(data)

            # Read-only deprecated aliases from legacy writer variants
            deprecated_aliases = {
                'phase_coherence_final': 'phase_coherence_mean',
                'grad_phase_var_final': 'grad_phase_var_mean',
                'J_info_l2_final': 'j_info_l2_mean',
                'omega_saturation_final': 'omega_sat_mean',
            }
            for alias_key, canonical_key in deprecated_aliases.items():
                if alias_key in h5f and canonical_key not in telemetry:
                    alias_data = np.asarray(h5f[alias_key][:], dtype=np.float64)
                    if alias_data.size > 0:
                        telemetry[canonical_key] = float(np.mean(alias_data))

            if 'telemetry' in h5f:
                base_telemetry = h5f['telemetry']
                if 'C_invariant' in base_telemetry and len(base_telemetry['C_invariant']) > 0:
                    c_data = np.asarray(base_telemetry['C_invariant'][:], dtype=np.float64)
                    telemetry['C_invariant_final'] = float(c_data[-1])
                    telemetry['collapse_invariant'] = float(np.mean(c_data))
                    telemetry['collapse_invariant_mean'] = float(np.mean(c_data))
                if 'energy' in base_telemetry and len(base_telemetry['energy']) > 0:
                    e_data = np.asarray(base_telemetry['energy'][:], dtype=np.float64)
                    telemetry['energy_final'] = float(e_data[-1])

            # 4. Memory-Safe Geometry Loading
            if "omega_sq_final" in h5f:
                if h5f["omega_sq_final"].size > 512**3:
                    logger.warning(f"Large geometry field detected ({h5f['omega_sq_final'].size} elements). Skipping load to prevent OOM.")
                else:
                    telemetry['omega_sq_final'] = h5f["omega_sq_final"][()]

            # Optional V13 causal-field artifacts (legacy-safe: missing keys are ignored)
            if "A_final" in h5f:
                telemetry['A_final'] = ArtifactLoader._adaptive_load_dataset(h5f["A_final"], "A_final")
            # DC-v1.0: the spectral-space affect velocity dataset was renamed
            # A_dot_final -> A_dot_k_final.  Read the new name first, fall back to
            # the legacy name for pre-DC-v1.0 artifacts, and expose under both
            # telemetry keys so downstream consumers (app.py, reports) keep working.
            _a_dot_ds = (
                "A_dot_k_final" if "A_dot_k_final" in h5f
                else ("A_dot_final" if "A_dot_final" in h5f else None)
            )
            if _a_dot_ds is not None:
                _a_dot_val = ArtifactLoader._adaptive_load_dataset(h5f[_a_dot_ds], _a_dot_ds)
                telemetry['A_dot_k_final'] = _a_dot_val
                telemetry['A_dot_final'] = _a_dot_val  # legacy alias

            # Solver contract (written by worker_cupy on success; absent in legacy artifacts)
            if "solver_contract" in h5f:
                try:
                    raw = h5f["solver_contract"][0]
                    if isinstance(raw, (bytes, np.bytes_)):
                        raw = raw.decode("utf-8", errors="replace")
                    telemetry['solver_contract'] = json.loads(raw)
                except Exception:
                    telemetry['solver_contract'] = None

            # Optional quantule_events import for legacy timelines
            if "quantule_events" in h5f:
                q_grp = h5f["quantule_events"]
                omega_local = np.asarray(q_grp["omega_local"][:]) if "omega_local" in q_grp else np.array([])
                bandwidth = np.asarray(q_grp["bandwidth"][:]) if "bandwidth" in q_grp else np.array([])
                t_step = np.asarray(q_grp["t_step"][:]) if "t_step" in q_grp else np.array([])
                min_len = min(len(omega_local), len(bandwidth), len(t_step))
                if min_len > 0:
                    telemetry['quantule_omega_local'] = cast(Any, omega_local[:min_len])
                    telemetry['quantule_bandwidth'] = cast(Any, bandwidth[:min_len])
                    telemetry['quantule_t_step'] = cast(Any, t_step[:min_len])

        return psi_final_3d, rho_final_3d, telemetry

    @staticmethod
    def _compute_tau_c(time_series: np.ndarray) -> float:
        if len(time_series) < 2: return 0.0
        ts_norm = time_series - np.mean(time_series)
        if np.all(ts_norm == 0): return 0.0
        autocorr = np.correlate(ts_norm, ts_norm, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr /= (autocorr[0] + 1e-12)
        tau_c = np.where(autocorr < np.exp(-1))[0]
        return float(tau_c[0]) if len(tau_c) > 0 else float(len(autocorr))


# ==========================================
# STAGE 2: Spectral Fidelity Engine
# ==========================================
class SpectralFidelityEngine:
    @staticmethod
    def run(rho_final: np.ndarray) -> Dict[str, Any]:
        logger.info("[Stage 2: SpectralFidelityEngine] Running Quantule Profiler (CEPP v2.0)...")
        if rho_final.ndim != 3:
            raise ValueError(f"Expected 3D rho array, got {rho_final.shape}")

        try:
            profiler_results = cep_profiler.prime_log_sse(rho_final)
            
            # [FIX 1] Pull safely from profiler_results instead of manually recalculating!
            n_bragg_peaks = profiler_results.get("bragg_peaks_detected", 0)
            bragg_prime_sse = float(profiler_results.get("bragg_lattice_sse", 999.0))

            log_prime_sse = float(profiler_results.get("log_prime_sse", 999.0))
            validation_status = "PASS" if log_prime_sse < 1.0 else "FAIL: HIGH_SSE"

            return {
                "validation_status": validation_status,
                "log_prime_sse": log_prime_sse,
                "scaling_factor_S": profiler_results.get("scaling_factor_S", 0.0),
                "dominant_peak_k": profiler_results.get("dominant_peak_k", 0.0),
                "secondary_peak_k": profiler_results.get("secondary_peak_k", 0.0),
                "analysis_protocol": "CEPP v2.0",
                "prime_log_targets": getattr(cep_profiler, "LOG_PRIME_TARGETS", np.array([])).tolist(),
                "sse_null_phase_scramble": float(profiler_results.get("sse_null_phase_scramble", 999.0)),
                "sse_null_target_shuffle": float(profiler_results.get("sse_null_target_shuffle", 999.0)),
                "primary_harmonic_error": float(profiler_results.get("primary_harmonic_error", 999.0)),
                "missing_peak_penalty": float(profiler_results.get("missing_peak_penalty", 0.0)),
                "noise_penalty": float(profiler_results.get("noise_penalty", 0.0)),
                "best_single_error": float(profiler_results.get("best_single_error", 999.0)),
                "bragg_lattice_sse": bragg_prime_sse,
                "bragg_peaks_detected": n_bragg_peaks,
                "n_bragg_peaks": n_bragg_peaks,
                "bragg_prime_sse": bragg_prime_sse,
                "measured_peaks": profiler_results.get("measured_peaks", []),
                "scaled_peaks": profiler_results.get("scaled_peaks", []),
                
                # Restored Diagnostics for Hunter database
                "n_peaks_found_main": profiler_results.get("n_peaks_found_main", 0),
                "failure_reason_main": profiler_results.get("failure_reason_main", None),
                "n_peaks_found_null_a": profiler_results.get("n_peaks_found_null_a", 0),
                "failure_reason_null_a": profiler_results.get("failure_reason_null_a", None),
                "n_peaks_found_null_b": profiler_results.get("n_peaks_found_null_b", 0),
                "failure_reason_null_b": profiler_results.get("failure_reason_null_b", None),

                # Directional (axis-resolved) spectral SSE — falsifiability supplement
                "sse_directional_x": float(profiler_results.get("sse_directional_x", 999.0)),
                "sse_directional_y": float(profiler_results.get("sse_directional_y", 999.0)),
                "sse_directional_z": float(profiler_results.get("sse_directional_z", 999.0)),
                "sse_directional_min": float(profiler_results.get("sse_directional_min", 999.0)),
                "directional_consistency": float(profiler_results.get("directional_consistency", 0.0)),

                "collapse_event_count": 0 # Default, overwritten by LOM Engine
            }
        except Exception as e:
            logger.critical(f"CRITICAL: Spectral Fidelity Profiler failed: {e}")
            # [FIX 2] DO NOT RAISE. Return a maximum penalty gracefully so the loop survives!
            return {
                "validation_status": "FAIL: DEGENERATE_FIELD",
                "log_prime_sse": 999.0,
                "dominant_peak_k": 0.0,
                "secondary_peak_k": 0.0,
                "primary_harmonic_error": 999.0,
                "missing_peak_penalty": 7.0,
                "noise_penalty": 0.0,
                "best_single_error": 999.0,
                "sse_null_phase_scramble": 999.0,
                "sse_null_target_shuffle": 999.0,
                "bragg_lattice_sse": 999.0,
                "bragg_peaks_detected": 0,
                "n_bragg_peaks": 0,
                "bragg_prime_sse": 999.0,
                "measured_peaks": [],
                "scaled_peaks": [],
                "n_peaks_found_main": 0,
                "failure_reason_main": f"profiler_exception: {e}",
                "collapse_event_count": 0
            }


# ==========================================
# STAGE 3: Contract Enforcer
# ==========================================
class ContractEnforcerEngine:
    @staticmethod
    def enforce(spec_results: Dict[str, Any]) -> None:
        logger.info("[Stage 3: ContractEnforcerEngine] Validating metric contracts...")
        try:
            if os.path.exists("metric_contracts.yaml"):
                with open("metric_contracts.yaml", "r") as yf:
                    contracts = yaml.safe_load(yf)
                for k, bounds in contracts.get("spectral_fidelity", {}).items():
                    val = spec_results.get(k)
                    if val is not None:
                        if not (bounds.get("min", -float('inf')) <= val <= bounds.get("max", float('inf'))):
                            spec_results["validation_status"] = "FAIL: NUMERICAL_INVALID"
                            spec_results["primary_harmonic_error"] = 999.0
                            logger.warning(f"  -> Contract violation for {k}: {val}")
        except Exception as e:
            logger.warning(f"  -> Contract enforcement bypassed/failed: {e}")


# ==========================================
# STAGES 5-10: Deep Analysis Engines
# ==========================================
class TopologyEngine:
    @staticmethod
    def null_result() -> Dict[str, Any]:
        return {
            "q_type": "Transient",
            "persistent_loops": 0,
            "persistent_voids": 0,
            "betti_0": 1,
            "betti_1": 0,
            "betti_2": 0,
        }

    @staticmethod
    def run_tda(rho_final: np.ndarray, config_hash: str, output_dir: str) -> Dict[str, Any]:
        logger.info("[Stage 5: TopologyEngine] Executing Persistent Homology...")
        try:
            csv_content, taxonomy = tda_profiler.extract_and_classify_topology(rho_final)
            logger.info(f"  Topological Taxonomy Detected: {taxonomy}")
            
            # Safe nested TDA pathing
            tda_dir = os.path.join(output_dir, "tda")
            os.makedirs(tda_dir, exist_ok=True)
            out_path = os.path.join(tda_dir, f"{config_hash}_quantule_events.csv")
            
            with open(out_path, 'w') as f:
                f.write(csv_content)

            q_theta = int(taxonomy.get("Q_theta", 0))
            q_nu = int(taxonomy.get("Q_nu", 0))
            q_transient = int(taxonomy.get("Transient", 0))

            q_type = "Transient"
            if q_theta > 0:
                q_type = "Q_theta"
            elif q_nu > 0:
                q_type = "Q_nu"
            elif q_transient > 0:
                q_type = "Transient"

            return {
                "q_type": q_type,
                "persistent_loops": q_nu,
                "persistent_voids": q_theta,
                "betti_0": 1,
                "betti_1": q_nu,
                "betti_2": q_theta,
            }
        except Exception as e:
            logger.warning(f"TDA analysis failed: {e}")
            return TopologyEngine.null_result()


class LOMTelemetryEngine:
    @staticmethod
    def extract(config_hash: str, output_dir: str, params_dict: dict, psi_final: np.ndarray, rho_final: np.ndarray, telemetry: dict) -> int:
        logger.info("[Stage 6: LOMTelemetryEngine] Extracting offline collapse events...")
        try:
            L_domain = float(params_dict.get("simulation", {}).get("L_domain", 10.0))
            N_grid = int(params_dict.get("simulation", {}).get("N_grid", 64))
            
            omega_temporal_mean = 0.0
            bandwidth_dk_val = 0.0
            emergence_t_step_val = 0
            omega_sq_field = telemetry.get('omega_sq_final')
            
            # Consume previously extracted telemetry logs safely
            quantule_omega_local = telemetry.get('quantule_omega_local', [])
            quantule_bandwidth = telemetry.get('quantule_bandwidth', [])
            quantule_t_step = telemetry.get('quantule_t_step', [])
            min_len = min(len(quantule_omega_local), len(quantule_bandwidth), len(quantule_t_step))
            if min_len > 0:
                quantule_omega_local = quantule_omega_local[:min_len]
                quantule_bandwidth = quantule_bandwidth[:min_len]
                quantule_t_step = quantule_t_step[:min_len]

                omega_temporal_mean = float(np.mean(quantule_omega_local))
                bandwidth_dk_val = float(quantule_bandwidth[0])
                emergence_t_step_val = int(quantule_t_step[0])

                pd.DataFrame({
                    "t_step": quantule_t_step,
                    "omega_local": quantule_omega_local,
                    "spectral_bandwidth_dk": quantule_bandwidth
                }).to_csv(os.path.join(output_dir, f"{config_hash}_gravity_timeline.csv"), index=False)
                        
            theta = np.angle(psi_final)
            
            # --- ASTE V4: Adaptive Quantule Thresholding ---
            # Replaces the hard-coded 0.8 to prevent "Solid Block" saturation dumps
            mu_rho = float(np.mean(rho_final))
            sigma_rho = float(np.std(rho_final))
            
            # --- Explicit Zero-Variance Guard ---
            if sigma_rho < 1e-12:
                logger.warning("  -> LOM Telemetry Guard: Zero variance detected (flatline). Rejecting.")
                return 0
                
            critical_threshold = mu_rho + (3.0 * sigma_rho)
            
            z_indices, y_indices, x_indices = np.where(rho_final > critical_threshold)
            
            # --- Explosion / Saturation Guard ---
            # If the variance is 0 (a flat block), everything might trigger the threshold.
            # If more than 20% of the box is "collapsing", it's garbage, not a prime lock.
            max_valid_events = (N_grid**3) * 0.20
            if len(z_indices) == 0 or len(z_indices) > max_valid_events: 
                logger.warning(f"  -> LOM Telemetry Guard: {len(z_indices)} events rejected (flatline/explosion).")
                return 0
                
            dx = L_domain / N_grid
            phys_x = (x_indices - N_grid / 2) * dx
            phys_y = (y_indices - N_grid / 2) * dx
            phys_z = (z_indices - N_grid / 2) * dx
            normalized_phase = (theta[z_indices, y_indices, x_indices] + np.pi) / (2 * np.pi)
            
            # Pure analysis - Do not silently recompute the conformal geometry fallback
            if omega_sq_field is not None:
                spatial_gravity_pressure = omega_sq_field[z_indices, y_indices, x_indices]
            else:
                spatial_gravity_pressure = np.zeros(len(z_indices))
                logger.info("  -> 'omega_sq_final' not found in artifact; skipping spatial gravity map reconstruction.")
            
            df = pd.DataFrame({
                'idx_x': x_indices, 'idx_y': y_indices, 'idx_z': z_indices,
                'phys_x': phys_x, 'phys_y': phys_y, 'phys_z': phys_z,
                'rho_intensity': rho_final[z_indices, y_indices, x_indices],
                'complex_phase_normalized': normalized_phase,
                'temporal_omega_mean': omega_temporal_mean,
                'spatial_gravity_omega_sq': spatial_gravity_pressure,
                'bandwidth_dk': bandwidth_dk_val,
                'emergence_t_step': emergence_t_step_val
            })
            
            scale_nm = (L_domain * 1e9) / N_grid
            df["x_nm"] = df["phys_x"] * scale_nm
            df["y_nm"] = df["phys_y"] * scale_nm
            df["z_nm"] = df["phys_z"] * scale_nm
            
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, f"{config_hash}_etch_ready.csv"), index=False)
            
            logger.info(f"  -> Preserved {len(z_indices)} physical collapse events for fabrication.")
            return len(z_indices)
        except Exception as e:
            logger.warning(f"  -> LOM Telemetry extraction failed: {e}")
            return 0


class EmpiricalBridgeEngine:
    @staticmethod
    def run(psi_final: np.ndarray) -> Tuple[float, float]:
        logger.info("[Stage 7 & 8: EmpiricalBridgeEngine] Generating Quantum Optics Bridge & Phase Ablation Null Tests...")
        c4_contrast = 0.0
        ablated_c4_contrast = 0.0
        try:
            jsa = spdc_empirical_bridge.calculate_joint_spectral_amplitude(psi_final)
            c4 = spdc_empirical_bridge.deconvolve_to_c4_interference(jsa)
            c4_contrast = float(np.max(c4) - np.mean(c4))
            
            # Falsifiability Phase Ablation
            psi_null = np.abs(psi_final).astype(np.complex64)
            jsa_null = spdc_empirical_bridge.calculate_joint_spectral_amplitude(psi_null)
            c4_null = spdc_empirical_bridge.deconvolve_to_c4_interference(jsa_null)
            ablated_c4_contrast = float(np.max(c4_null) - np.mean(c4_null))
        except Exception as e:
            logger.warning(f"Empirical Bridge computation failed: {e}")
        return c4_contrast, ablated_c4_contrast


class TensorValidationEngine:
    @staticmethod
    def run(rho_final: np.ndarray, psi_final: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        logger.info("[Stage 9: TensorValidationEngine] Computing Stress-Energy Tensor...")
        try:
            phase_field = np.angle(psi_final) if np.iscomplexobj(psi_final) else psi_final
            T = tensor_validation.construct_T_info(rho_final, phase_field)
            symmetry_error = tensor_validation.tensor_symmetry_test(T)
            shear_stress = tensor_validation.perfect_fluid_reduction_test(T)
            return symmetry_error, shear_stress
        except Exception as e:
            logger.warning(f"Tensor validation failed: {e}")
            return None, None


class StatisticalValidationEngine:
    @staticmethod
    def run(target_sse: float, grid_shape: tuple, n_iterations: int = 500) -> Tuple[float, Optional[float]]:
        logger.info(f"[Stage 10: StatisticalValidationEngine] Running Monte Carlo ({n_iterations} iterations)...")
        try:
            np.random.seed(42)  # Deterministic validation
            return monte_carlo_engine.run_monte_carlo_p_value(target_sse, grid_shape=grid_shape, n_iterations=n_iterations)
        except Exception as e:
            logger.warning(f"Statistical validation failed: {e}")
            return 1.0, None


class ValidationDerivedMetricsEngine:
    @staticmethod
    def run(psi_final: np.ndarray, rho_final: np.ndarray, telemetry: Dict[str, Any]) -> Dict[str, float]:
        rho = np.maximum(rho_final.astype(np.float64, copy=False), 1e-12)
        max_amp_peak = float(np.max(np.abs(psi_final)))
        phase = np.angle(psi_final)

        grad_phase = np.gradient(phase)
        grad_phase_sq = np.zeros_like(rho, dtype=np.float64)
        for comp in grad_phase:
            grad_phase_sq += np.asarray(comp, dtype=np.float64) ** 2

        grad_rho = np.gradient(rho)
        grad_rho_sq = np.zeros_like(rho, dtype=np.float64)
        for comp in grad_rho:
            grad_rho_sq += np.asarray(comp, dtype=np.float64) ** 2

        fft_rho = np.fft.fftn(rho)
        power = np.abs(fft_rho) ** 2
        grid_shape = rho.shape
        freq_axes = [np.fft.fftfreq(n) for n in grid_shape]
        kx, ky, kz = np.meshgrid(*freq_axes, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        denom = float(np.sum(power) + 1e-12)

        collapse_invariant = float(np.mean(rho**2))
        phase_coherence = float(np.abs(np.mean(np.exp(1j * phase))))
        derived = {
            'max_amp_peak': max_amp_peak,
            'phase_coherence_final': phase_coherence,
            'phase_coherence_mean': phase_coherence,
            'grad_phase_var_mean': float(np.var(grad_phase_sq)),
            'j_info_l2_mean': float(np.mean(grad_rho_sq / (4.0 * rho))),
            'omega_sat_mean': float(np.mean(np.asarray(telemetry.get('omega_sq_final', rho), dtype=np.float64))),
            'spectral_bandwidth_mean': float(np.sum(k_mag * power) / denom),
            'collapse_invariant': collapse_invariant,
            'collapse_invariant_mean': collapse_invariant,
        }
        return derived


class AletheiaMetricsEngine:
    @staticmethod
    def run(rho: np.ndarray) -> dict:
        metrics = {
            "pli": 0.0, "ic": 1.0, 
            "nonlinear_balance": None, "correlation_length": None, "fractal_dimension": None
        }
        
        # Phase 3 Principled Localization Index (PLI)
        sum_rho = np.sum(rho)
        if sum_rho != 0:
            metrics["pli"] = float(np.sum((rho / sum_rho)**2) * rho.size)

        # Informational Compressibility (IC)
        try:
            proxy_E = np.sum(rho**2)
            rho_flat = rho.flatten()
            sum_rho_flat = np.sum(rho_flat)
            if sum_rho_flat != 0:
                proxy_S = scipy_entropy((rho_flat / sum_rho_flat) + 1e-9)
                # Apply a non-uniform thermal perturbation to test informational rigidity
                rho_p = rho + (0.01 * np.mean(rho))
                proxy_E_p = np.sum(rho_p**2)
                rho_p_flat = rho_p.flatten()
                proxy_S_p = scipy_entropy((rho_p_flat / np.sum(rho_p_flat)) + 1e-9)
                dE, dS = proxy_E_p - proxy_E, proxy_S_p - proxy_S
                if dE != 0 and not np.isnan(dE) and not np.isnan(dS): 
                    metrics["ic"] = float(dS / dE)
        except Exception as e:
            logger.warning(f"IC calculation failed: {e}")

        # Collapse Dynamics Metrics
        try:
            metrics["nonlinear_balance"] = collapse_metrics.compute_nonlinear_balance(rho)
        except Exception as exc:
            logger.warning(f"nonlinear_balance calculation failed: {exc}")
        try:
            metrics["correlation_length"] = collapse_metrics.compute_correlation_length(rho)
        except Exception as exc:
            logger.warning(f"correlation_length calculation failed: {exc}")
        try:
            metrics["fractal_dimension"] = collapse_metrics.compute_fractal_dimension_boxcount(rho, threshold=0.1)
        except Exception as exc:
            logger.warning(f"fractal_dimension calculation failed: {exc}")

        return metrics


# ==========================================
# STAGE 11: Provenance Assembler
# ==========================================
class ProvenanceAssembler:
    @staticmethod
    def assemble(config_hash: str, legacy_hash: Optional[str], spec_results: dict, telemetry: dict, metrics: dict,
                 c4: float, c4_ablated: float, sym_err: float, shear: float, p_val: float, rand_sse: float,
                 tda_results: dict, run_metadata: Optional[Dict[str, Any]] = None) -> dict:
        logger.info("[Stage 11: ProvenanceAssembler] Assembling canonical payload...")
        metadata_block: Dict[str, Any] = {
            "config_hash": config_hash,
            "legacy_hash_reference": legacy_hash,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "schema_version": SCHEMA_VERSION
        }
        if run_metadata:
            metadata_block["run_metadata"] = run_metadata

        return {
            "metadata": metadata_block,
            "spectral_fidelity": spec_results,
            "aletheia_metrics": {
                "pcs": telemetry.get('phase_coherence_mean', None),
                "pli": metrics.get('pli', 0.0),
                "ic": metrics.get('ic', 0.0),
                "phase_coherence_mean": telemetry.get('phase_coherence_mean', None),
                
                # GPU Telemetry mapped for Hunter penalties
                "j_info_l2_mean": telemetry.get('j_info_l2_mean', None),
                "grad_phase_var_mean": telemetry.get('grad_phase_var_mean', None),
                "grad_phase_var_tau_c": telemetry.get('grad_phase_var_tau_c', None),
                "max_amp_peak": telemetry.get('max_amp_peak', None),
                "clamp_fraction_mean": telemetry.get('clamp_fraction_mean', None),
                "omega_sat_mean": telemetry.get('omega_sat_mean', None),
                "spectral_bandwidth_mean": telemetry.get('spectral_bandwidth_mean', None),
                "collapse_invariant": telemetry.get('collapse_invariant', None),
                "collapse_invariant_mean": telemetry.get('collapse_invariant_mean', None),
                
                # Conservation Invariants 
                "C_invariant_final": telemetry.get('C_invariant_final', None),
                "energy_final": telemetry.get('energy_final', None),

                # Phantom Filter Epistemic Guardrails
                "tau_c": telemetry.get('tau_c', None),
                "relative_variance": telemetry.get('relative_variance', None),
                
                # Collapse Dynamics Restored
                "nonlinear_balance": metrics.get("nonlinear_balance"),
                "correlation_length": metrics.get("correlation_length"),
                "fractal_dimension": metrics.get("fractal_dimension")
            },
            "empirical_bridge": {
                "c4_interference_contrast": c4,
                "ablated_c4_contrast": c4_ablated
            },
            "tensor_validation": {
                "symmetry_error": sym_err,
                "shear_stress": shear
            },
            "topology": tda_results,
            "statistical_validation": {
                "p_value": p_val,
                "mean_random_sse": rand_sse
            },
            "solver_contract": telemetry.get("solver_contract", None),
        }


# ==========================================
# FALSIFIABILITY LADDER
# ==========================================

class RefinementManifestGenerator:
    """Produce variant parameter dicts for the falsifiability ladder.

    Generates four variants from a base run:
      - dt_half     : timestep halved (temporal convergence check)
      - dt_quarter  : timestep quartered (second temporal step)
      - dealias_2_3 : dealias cutoff raised from 0.5 to 2/3 (alias-sensitivity check)
      - grid_double : grid doubled in each dimension (spatial convergence check)

    The grid_double variant is only emitted when the base grid is ≤ 64 to avoid
    scheduling runs that would exceed a reasonable VRAM budget.  Callers should
    check the 'feasible' flag on each entry.
    """

    @staticmethod
    def generate(base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns a list of variant dicts, each with keys:
          label        : human-readable variant name
          params       : full parameter dict to pass to the worker
          feasible     : bool — False if the variant is likely to OOM
        """
        sim = base_params.get("simulation", {})
        base_dt = float(sim.get("dt", 0.001))
        base_n = int(sim.get("n_grid") or sim.get("N_grid", 64))
        base_steps = int(sim.get("t_steps") or sim.get("T_steps", 250))

        def _variant(label: str, dt_override=None, n_override=None,
                     steps_override=None, dealias_override=None,
                     feasible: bool = True) -> Dict[str, Any]:
            v = {k: v for k, v in base_params.items() if k != "simulation"}
            v["simulation"] = dict(sim)
            if dt_override is not None:
                v["simulation"]["dt"] = dt_override
            if n_override is not None:
                v["simulation"]["n_grid"] = n_override
                v["simulation"]["N_grid"] = n_override
            if steps_override is not None:
                v["simulation"]["t_steps"] = steps_override
                v["simulation"]["T_steps"] = steps_override
            if dealias_override is not None:
                v["param_dealias_fraction"] = dealias_override
            v["_refinement_label"] = label
            return {"label": label, "params": v, "feasible": feasible}

        variants = [
            _variant("dt_half",
                     dt_override=base_dt / 2.0,
                     steps_override=base_steps * 2),
            _variant("dt_quarter",
                     dt_override=base_dt / 4.0,
                     steps_override=base_steps * 4),
            _variant("dealias_2_3",
                     dealias_override=2.0 / 3.0),
            _variant("grid_double",
                     n_override=base_n * 2,
                     feasible=(base_n <= 64)),
        ]
        return variants


class RefinementAudit:
    """Assess whether a resonance signal survives the falsifiability ladder.

    Accepts a list of (label, provenance_dict) pairs — one for the baseline
    run and one for each refinement variant that was actually executed.
    Returns a structured convergence report.

    Acceptance criteria (all must hold to classify a signal as STABLE):
      1. log_prime_sse < 1.0 in baseline.
      2. log_prime_sse remains < convergence_threshold in every refinement run.
      3. dominant_peak_k shifts by less than peak_drift_tol across all runs.
      4. Every variant reports a finite sse_directional_min below the convergence threshold.
      5. Every variant reports directional_consistency below the directional threshold.
    """

    CONVERGENCE_THRESHOLD = 2.0   # SSE may loosen slightly under refinement
    DIRECTIONAL_SSE_THRESHOLD = 2.0  # Directional SSE must also remain below the convergence ceiling
    DIRECTIONAL_CONSISTENCY_THRESHOLD = 0.25  # Too much axis-to-axis SSE spread is suspicious
    PEAK_DRIFT_TOL = 0.05          # Fraction of Nyquist
    MIN_VARIANTS = 2               # Need at least baseline + 2 variants to assess

    @staticmethod
    def assess(results: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        results : list of (label, provenance_dict)
            provenance_dict must contain at least a 'spectral_fidelity' sub-dict.

        Returns
        -------
        dict with keys: verdict ('STABLE'|'DRIFTING'|'ABSENT'|'INSUFFICIENT_DATA'),
                        baseline_sse, variant_sses, peak_drift, per_variant_details
        """
        if len(results) < RefinementAudit.MIN_VARIANTS:
            return {
                "verdict": "INSUFFICIENT_DATA",
                "reason": f"Need ≥{RefinementAudit.MIN_VARIANTS} runs, got {len(results)}",
                "baseline_sse": None,
                "variant_sses": {},
                "peak_drift": None,
                "per_variant_details": {},
            }

        baseline_label, baseline_prov = results[0]
        baseline_sf = baseline_prov.get("spectral_fidelity", {})
        baseline_sse = float(baseline_sf.get("log_prime_sse", 999.0))
        baseline_peak = float(baseline_sf.get("dominant_peak_k", 0.0))

        if baseline_sse >= 1.0:
            return {
                "verdict": "ABSENT",
                "reason": f"Baseline SSE {baseline_sse:.4f} ≥ 1.0; no signal to audit.",
                "baseline_sse": baseline_sse,
                "variant_sses": {},
                "peak_drift": None,
                "per_variant_details": {},
            }

        variant_sses: Dict[str, float] = {}
        peak_values: List[float] = [baseline_peak]
        per_variant: Dict[str, Dict] = {}
        failures: List[str] = []

        for label, prov in results[1:]:
            sf = prov.get("spectral_fidelity", {})
            sse = float(sf.get("log_prime_sse", 999.0))
            peak = float(sf.get("dominant_peak_k", 0.0))
            dir_min = float(sf.get("sse_directional_min", float('nan')))
            directional_consistency = float(sf.get("directional_consistency", float('nan')))
            variant_sses[label] = sse
            if peak > 0.0:
                peak_values.append(peak)
            per_variant[label] = {
                "log_prime_sse": sse,
                "dominant_peak_k": peak,
                "sse_directional_min": dir_min,
                "directional_consistency": directional_consistency,
            }
            if sse >= RefinementAudit.CONVERGENCE_THRESHOLD:
                failures.append(f"{label}: SSE={sse:.4f}")
            if not math.isfinite(dir_min) or dir_min >= RefinementAudit.DIRECTIONAL_SSE_THRESHOLD:
                failures.append(f"{label}: directional_sse_min={dir_min:.4f}")
            if math.isfinite(directional_consistency) and directional_consistency >= RefinementAudit.DIRECTIONAL_CONSISTENCY_THRESHOLD:
                failures.append(f"{label}: directional_consistency={directional_consistency:.4f} ≥ tol={RefinementAudit.DIRECTIONAL_CONSISTENCY_THRESHOLD}")

        peak_drift = float(max(peak_values) - min(peak_values)) if len(peak_values) > 1 else 0.0
        if peak_drift >= RefinementAudit.PEAK_DRIFT_TOL:
            failures.append(f"peak_drift={peak_drift:.4f} ≥ tol={RefinementAudit.PEAK_DRIFT_TOL}")

        if failures:
            verdict = "DRIFTING"
            reason = "Signal does not survive refinement: " + "; ".join(failures)
        else:
            verdict = "STABLE"
            reason = "Signal survives timestep, dealias, and/or grid refinement."

        return {
            "verdict": verdict,
            "reason": reason,
            "baseline_sse": baseline_sse,
            "variant_sses": variant_sses,
            "peak_drift": peak_drift,
            "per_variant_details": per_variant,
        }


# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
class ValidationPipeline:
    def __init__(self, input_path: str, params_path: str, output_dir: str, mc_iterations: int = 500):
        self.input_path = input_path
        self.params_path = params_path
        self.output_dir = output_dir
        self.mc_iterations = mc_iterations

    def run(self) -> bool:
        logger.info("--- SFP Module (Asset A6, v3.2) Initiating Validation ---")
        psi_final: Optional[np.ndarray] = None
        rho_final: Optional[np.ndarray] = None
        telemetry: Dict[str, Any] = {}
        
        try:
            with open(self.params_path, 'r') as f:
                params_dict = json.load(f)
            config_hash = params_dict.get("config_hash") or generate_canonical_hash(params_dict)
            legacy_hash = params_dict.get("param_hash_legacy")
            run_metadata = {
                "origin": params_dict.get("origin"),
                "staged_path": params_dict.get("staged_path"),
                "staged_at": params_dict.get("staged_at"),
                "staged_config_hash": params_dict.get("staged_config_hash"),
            }
            if not any(run_metadata.values()):
                run_metadata = None
        except Exception as e:
            logger.error(f"Failed to load params: {e}")
            return False

        try:
            psi_final, rho_final_loaded, telemetry = ArtifactLoader.load(self.input_path)
        except Exception as e:
            logger.error(f"Artifact Loader failed: {e}")
            return False

        psi_final = cast(np.ndarray, psi_final)
        rho_final = np.abs(psi_final) ** 2
        rho_final = cast(np.ndarray, rho_final)

        derived_metrics = ValidationDerivedMetricsEngine.run(psi_final, rho_final, telemetry)
        for key, value in derived_metrics.items():
            telemetry.setdefault(key, value)

        try:
            # 1. CEPP v2.0
            spec_results = SpectralFidelityEngine.run(rho_final)
            ContractEnforcerEngine.enforce(spec_results)
            
            target_sse = float(spec_results.get("log_prime_sse", 999.0))
            
            if target_sse > 15.0:
                logger.warning(f"Early Rejection: target_sse {target_sse:.2f} exceeds threshold 15.0. Skipping deep analysis.")
                tda_results = TopologyEngine.null_result()
                metrics_data = AletheiaMetricsEngine.run(rho_final)
                c4 = 0.0
                c4_ablated = 0.0
                sym_err = 1.0
                shear = 1.0
                p_val = 1.0
                rand_sse = 999.0
            else:
                tda_results = TopologyEngine.run_tda(rho_final, config_hash, self.output_dir)
                
                # LOM Telemetry populates events based on actual dynamics
                event_count = LOMTelemetryEngine.extract(config_hash, self.output_dir, params_dict, psi_final, rho_final, telemetry)
                spec_results["collapse_event_count"] = event_count
                
                c4, c4_ablated = EmpiricalBridgeEngine.run(psi_final)
                sym_err, shear = TensorValidationEngine.run(rho_final, psi_final)
                sym_err = sym_err if sym_err is not None else 1.0
                shear = shear if shear is not None else 1.0
                
                metrics_data = AletheiaMetricsEngine.run(rho_final)
                p_val, rand_sse = StatisticalValidationEngine.run(target_sse, rho_final.shape, self.mc_iterations)
                rand_sse = rand_sse if rand_sse is not None else 999.0

            payload = ProvenanceAssembler.assemble(
                config_hash, legacy_hash, spec_results, telemetry, metrics_data,
                c4, c4_ablated, sym_err, shear, p_val, rand_sse, tda_results, run_metadata
            )

            os.makedirs(self.output_dir, exist_ok=True)
            # Collision-free name derived from the artifact's /identity group
            # (seed + run_id + utc).  Falls back to provenance_{config_hash}.json
            # only for legacy artifacts without an identity group.
            out_path = provenance_path_for_artifact(self.output_dir, self.input_path, config_hash)
            self.provenance_path = out_path
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            logger.info(f"SUCCESS. Provenance compiled -> {out_path}")
            logger.info(f"Final SSE: {target_sse:.6f}")
            
        except Exception as e:
            logger.critical(f"FATAL PIPELINE ERROR: {e}")
            return False
            
        finally:
            gc.collect()

        return True

def run_validation(artifact_path: str, config_hash: Optional[str] = None) -> bool:
    """Entry point for orchestrator batch validation. Handles ruthless auto-cleanup and DB update."""
    import sqlite3
    import traceback
    import requests
    base_dir = os.path.dirname(artifact_path)
    h5_path = artifact_path
    provenance_path = None
    db_path = None
    params_path = None
    log_prime_sse = None
    success = False
    error_string = ""
    try:
        if config_hash is None:
            import re
            m = re.search(r"rho_history_([a-fA-F0-9]+)\.h5", os.path.basename(artifact_path))
            if m:
                config_hash = m.group(1)
        if not config_hash:
            logger.error("run_validation: config_hash could not be determined.")
            raise RuntimeError("config_hash could not be determined")
        
        param_candidates = [
            os.path.join(base_dir, f"params_{config_hash}.json"),
            os.path.join(base_dir, f"parameters_{config_hash}.json"),
            os.path.join(base_dir, f"{config_hash}_params.json"),
            os.path.join(base_dir, f"{config_hash}.json"),
        ]
        for candidate in param_candidates:
            if os.path.exists(candidate):
                params_path = candidate
                break
        if not params_path:
            for f in os.listdir(base_dir):
                if f.endswith('.json'):
                    params_path = os.path.join(base_dir, f)
                    break
        if not params_path or not os.path.exists(params_path):
            logger.error(f"run_validation: Could not find params file for {artifact_path}")
            raise RuntimeError(f"Could not find params file for {artifact_path}")
        
        output_dir = base_dir
        
        candidate = base_dir
        for _ in range(4):
            candidate_db = os.path.join(candidate, "simulation_ledger.db")
            if os.path.exists(candidate_db):
                db_path = candidate_db
                break
            candidate = os.path.dirname(candidate)
        if not db_path:
            db_path = os.path.abspath(os.path.join(base_dir, "..", "..", "simulation_ledger.db"))
        db_path = os.environ.get("ASTE_LEDGER_DB", db_path)
        
        pipeline = ValidationPipeline(artifact_path, params_path, output_dir)
        result = pipeline.run()
        success = result
        provenance_path = getattr(pipeline, "provenance_path", None) or provenance_path_for_artifact(
            output_dir, artifact_path, config_hash
        )
        
        if os.path.exists(provenance_path):
            with open(provenance_path, 'r') as f:
                provenance = json.load(f)
                log_prime_sse = provenance.get("spectral_fidelity", {}).get("log_prime_sse")
        
        try:
            conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE runs SET status = 'SUCCESS', artifact_url = ? WHERE config_hash = ?",
                (provenance_path, config_hash)
            )
            conn.commit()
            conn.close()
            logger.info(f"[Cleanup] Updated DB for {config_hash}: status=SUCCESS, artifact_url={provenance_path}")
        except Exception as db_exc:
            logger.warning(f"[Cleanup] Failed to update DB for {config_hash}: {db_exc}\n{traceback.format_exc()}")
        
        try:
            requests.post(
                "http://localhost:8000/api/fss/ping",
                json={
                    "config_hash": config_hash,
                    "log_prime_sse": log_prime_sse,
                    "artifact_url": provenance_path
                },
                timeout=5
            )
        except Exception as ping_exc:
            logger.warning(f"FSS ping failed: {ping_exc}")
            
    except Exception as e:
        error_string = str(e)
        logger.error(f"Validation failed: {error_string}\n{traceback.format_exc()}")
        
        penalty_data = {
            "config_hash": config_hash,
            "status": "FAIL",
            "error": error_string,
            "spectral_fidelity": {"log_prime_sse": 999.0},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(provenance_path, "w", encoding="utf-8") as f:
                json.dump(penalty_data, f, indent=2)
            logger.info(f"Synthesized penalty provenance for {config_hash}")
        except Exception as json_exc:
            logger.error(f"Failed to write penalty JSON: {json_exc}")

        try:
            if db_path:
                conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE runs SET status = 'FAIL' WHERE config_hash = ?",
                    (config_hash,)
                )
                conn.commit()
                conn.close()
        except Exception as db_exc:
            logger.error(f"Failed to update ledger on fail: {db_exc}")
        return False
    finally:
        try:
            if os.path.exists(h5_path):
                os.remove(h5_path)
                logger.info(f"[Cleanup] Deleted tensor file: {h5_path}")
        except Exception as cleanup_exc:
            logger.warning(f"[Cleanup] Failed to delete tensor file {h5_path}: {cleanup_exc}")
    return success


def parse_manifest(manifest_path: str) -> List[Dict[str, str]]:
    if manifest_path.endswith('.json'):
        with open(manifest_path, 'r') as f:
            data = json.load(f)
            jobs = data.get('jobs', data) if isinstance(data, dict) else data
    elif manifest_path.endswith('.csv'):
        import csv
        with open(manifest_path, 'r', newline='') as f:
            jobs = list(csv.DictReader(f))
    else:
        raise ValueError("Manifest must be .json or .csv")
        
    for i, job in enumerate(jobs):
        if 'input' not in job or 'params' not in job:
            raise ValueError(f"Manifest row {i} missing 'input' or 'params' keys.")
            
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Fidelity & Provenance (SFP) Module v3.2")
    parser.add_argument("--input", type=str, help="Path to input HDF5.")
    parser.add_argument("--params", type=str, help="Path to parameters.json.")
    parser.add_argument("--manifest", type=str, help="Path to batch manifest.")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory.")
    parser.add_argument("--mc-iterations", type=int, default=500, help="Number of Monte Carlo iterations.")
    parser.add_argument("--dry-run", action='store_true', help="Validate args only.")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Exiting.")
        sys.exit(0)

    if args.manifest:
        jobs = parse_manifest(args.manifest)
        successes = sum(1 for j in jobs if ValidationPipeline(j['input'], j['params'], args.output_dir, args.mc_iterations).run())
        logger.info(f"Batch complete: {successes}/{len(jobs)} succeeded.")
    elif args.input and args.params:
        config_hash = None
        try:
            with open(args.params, 'r') as f:
                params_dict = json.load(f)
                config_hash = params_dict.get("config_hash")
        except Exception:
            pass
        if not run_validation(args.input, config_hash):
            sys.exit(1)
    else:
        parser.error("Specify --manifest OR both --input and --params.")