"""solver/run.py - initialize_psi + run_simulation (verbatim from worker_cupy.py)."""
import os
import sys
import time
import json
import logging
import numpy as np
import cupy as cp
import h5py

from orchestrator.contracts import SOLVER_CONTRACT_VERSION
from orchestrator.diagnostics.runtime_audit import log_lifecycle_event
from orchestrator.run_identity import write_identity_group
from .core import ETDRK4Solver

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gravity'))
try:
    from unified_omega import derive_stable_conformal_factor
except ImportError:
    raise RuntimeError("unified_omega module required")


def initialize_psi(N, L, seed):
    """Initializes standard Gaussian packet bound to domain."""
    cp.random.seed(seed)
    x = cp.linspace(-L/2, L/2, N, endpoint=False, dtype=cp.float64)
    X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')
    R2 = X**2 + Y**2 + Z**2
    psi = cp.exp(-R2 / 2.0).astype(cp.complex128)
    noise = (cp.random.randn(*psi.shape) + 1j * cp.random.randn(*psi.shape)).astype(cp.complex128)
    return psi + 0.01 * noise

def run_simulation(
    N_grid,
    L_domain,
    T_steps,
    dt,
    seed,
    psi_params,
    output_path,
    config_hash=None,
    generation=None,
    identity=None,
):
    solver = ETDRK4Solver(N_grid, L_domain, dt, psi_params)
    psi = initialize_psi(N_grid, L_domain, seed)
    
    # State is permanently held in Spectral Space
    psi_k = solver.fft_single(psi) * solver.dealias_mask
    
    collapse_threshold = psi_params.get('collapse_threshold', 1e10)
    force_nan_step_raw = os.environ.get("ASTE_FORCE_NAN_STEP", "").strip()
    force_nan_step = None
    if force_nan_step_raw:
        try:
            force_nan_step = int(force_nan_step_raw)
        except ValueError:
            logging.warning(f"Ignoring invalid ASTE_FORCE_NAN_STEP={force_nan_step_raw!r}")
    
    # Pre-allocate telemetry
    dV = (L_domain / N_grid)**3
    history = []
    extended_telemetry = []
    final_step = -1
    fail_open_reason = None
    sentinel_code = -1.0
    
    start_time = time.time()
    

    for step in range(T_steps):
        final_step = step
        # 1. Causal field update (must run once per dt step)
        psi_real_step = solver.ifft_single(psi_k)
        rho_real = cp.maximum(cp.abs(psi_real_step)**2, cp.float64(1e-7))
        rho_k = solver.fft_single(rho_real)
        rho_k *= solver.dealias_mask
        solver.update_field_of_affect(rho_k, dt)

        # --- [ALETHEIA V4.4] Update Lagged Global Stats ONCE per timestep ---
        if getattr(solver, 'distributed_enabled', False):
            solver._lagged_global_mu, solver._lagged_global_sigma = solver._compute_global_mu_sigma(solver.rho)

        # 2. Core Evolution (in Spectral Space)
        psi_k = solver.step(psi_k)

        # Test hook: deterministic NaN injection for sentinel validation.
        if force_nan_step is not None and step >= force_nan_step:
            psi_k[0, 0, 0] = cp.complex128(cp.nan + 0j)

        # 1.5 Dynamic filter update on telemetry cadence
        if step % 10 == 0 or step == T_steps - 1:
            solver.update_dynamic_filters()
        
        # 3. Spectral Phase Centering (Zero FFT cost)
        if step % 50 == 0:
            mean_psi = psi_k[0, 0, 0] / (N_grid**3)
            mean_phase = cp.angle(mean_psi)
            psi_k *= cp.exp(-1j * mean_phase)
        
        # 4. Lightweight Telemetry & Pure Math Termination Guards
        if step % 10 == 0 or step == T_steps - 1:
            psi_real = solver.ifft_single(psi_k)

            # Terminate purely on mathematical overflow (NaN/Inf)
            if not cp.isfinite(psi_real).all():
                logging.error(f"Worker Terminated: Mathematical overflow (NaN/Inf) detected at step {step}.")
                fail_open_reason = "math_explosion"
                sentinel_code = 1002.0
                break

            # Terminate on absolute amplitude threshold (Hardware/FP32 limit)
            max_amp = float(cp.max(cp.abs(psi_real)))
            if max_amp > collapse_threshold:
                logging.error(f"Worker Terminated: Explosive amplitude ({max_amp:.2e}) exceeded threshold at step {step}.")
                fail_open_reason = "math_explosion"
                sentinel_code = 1002.0
                break

            rho = cp.maximum(cp.abs(psi_real)**2, cp.float64(1e-7))

            # --- Invariant Monitoring: Amortized to step % 50, No asnumpy! ---
            if step > 0 and step % 50 == 0:
                mean_rho = float(cp.mean(rho))
                mean_phase = float(cp.mean(cp.angle(psi_real))) # [ALETHEIA V4.4] Phase Extraction
                if mean_rho < 1e-5:
                    logging.error(f"Silent Physics Drift: Rho Collapse detected at step {step}.")
                    fail_open_reason = "physics_drift"
                    sentinel_code = 1003.0
                    break

                mean_omega = float(cp.mean(solver.omega_sq))
                range_omega = float(cp.max(solver.omega_sq) - cp.min(solver.omega_sq))
                if not (0.1 < mean_omega < 1e6) or range_omega < 1e-8:
                    logging.error(f"Geometry Sanity Check Failed: omega_sq out of expected range at step {step}.")
                    fail_open_reason = "geometry_sanity"
                    sentinel_code = 1004.0
                    break

            energy = float(cp.sum(rho, dtype=cp.float64)) * dV
            c_invariant = float(cp.sum(rho * rho, dtype=cp.float64)) * dV
            history.append({
                'step': step,
                'energy': energy,
                'C_invariant': c_invariant,
                'mean_phase': mean_phase if 'mean_phase' in locals() else 0.0 # Telemetry upgrade
            })

    total_time = time.time() - start_time
    logging.info(f"Evolution complete in {total_time:.2f}s")

    if sentinel_code != -1.0:
        psi_snapshot = solver.ifft_single(psi_k)
        temp_path = f"{output_path}.tmp"
        f = h5py.File(temp_path, 'w')
        try:
            f.create_dataset(
                'psi_final',
                data=cp.asnumpy(psi_snapshot),
                chunks=True,
                compression='gzip',
                compression_opts=4,
            )
            f.create_dataset(
                'sentinel_code',
                data=np.array([sentinel_code], dtype=np.float64),
            )
            f.create_dataset(
                'sentinel_reason',
                data=np.array([fail_open_reason], dtype='S64'),
            )
            hist_grp = f.create_group('telemetry')
            hist_grp.create_dataset(
                'step',
                data=np.array([t['step'] for t in history], dtype=np.int64),
                chunks=True,
                compression='gzip',
                compression_opts=4,
            )
            hist_grp.create_dataset(
                'energy',
                data=np.array([t['energy'] for t in history], dtype=np.float64),
                chunks=True,
                compression='gzip',
                compression_opts=4,
            )
            hist_grp.create_dataset(
                'C_invariant',
                data=np.array([t['C_invariant'] for t in history], dtype=np.float64),
                chunks=True,
                compression='gzip',
                compression_opts=4,
            )
            if identity:
                write_identity_group(f, identity)
            f.flush()
        finally:
            f.close()

        os.replace(temp_path, output_path)

        if hasattr(os, 'sync'):
            os.sync()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        log_lifecycle_event(
            stage="h5_write",
            config_hash=config_hash,
            generation=generation if isinstance(generation, int) else None,
            details={"artifact_url": output_path, "sentinel": sentinel_code, "reason": fail_open_reason},
        )
        return {
            "status": "FAIL",
            "reason": fail_open_reason,
            "sentinel": sentinel_code,
            "artifact_url": output_path,
            "final_step": final_step,
        }
    
    # 4. Final Disk I/O 
    psi_final = solver.ifft_single(psi_k)
    rho_final = cp.abs(psi_final)**2
    A_final = solver.A_real
    A_dot_k_final = solver.A_dot_k  # spectral-space affect velocity (k-space)
    
    # ASTE D.5: Explicitly derive final geometry for validation ingestion
    final_geometry_params = psi_params
    if solver.distributed_enabled:
        global_mu, global_sigma = solver._compute_global_mu_sigma(cp.maximum(rho_final, 1e-12))
        final_geometry_params = dict(psi_params)
        final_geometry_params["global_stats_enabled"] = True
        final_geometry_params["use_global_stats"] = True
        final_geometry_params["distributed_enabled"] = True
        final_geometry_params["global_mu"] = global_mu
        final_geometry_params["global_sigma"] = global_sigma

    omega_sq_final = derive_stable_conformal_factor(rho_final, final_geometry_params)

    energy_history = np.array([t['energy'] for t in history], dtype=np.float64)
    if energy_history.size == 0:
        energy_sparkline = np.zeros(100, dtype=np.float64)
    elif energy_history.size == 1:
        energy_sparkline = np.full(100, float(energy_history[0]), dtype=np.float64)
    else:
        src_idx = np.linspace(0.0, float(energy_history.size - 1), num=energy_history.size, dtype=np.float64)
        dst_idx = np.linspace(0.0, float(energy_history.size - 1), num=100, dtype=np.float64)
        energy_sparkline = np.interp(dst_idx, src_idx, energy_history).astype(np.float64)

    temp_path = f"{output_path}.tmp"
    f = h5py.File(temp_path, 'w')
    try:
        f.create_dataset(
            'psi_final',
            data=cp.asnumpy(psi_final),
            chunks=True,
            compression='gzip',
            compression_opts=4,
        )
        f.create_dataset(
            'omega_sq_final',
            data=cp.asnumpy(omega_sq_final),
            chunks=True,
            compression='gzip',
            compression_opts=4,
        )
        f.create_dataset(
            'A_final',
            data=cp.asnumpy(A_final),
            chunks=True,
            compression='gzip',
            compression_opts=4,
        )
        f.create_dataset(
            'A_dot_k_final',
            data=cp.asnumpy(A_dot_k_final),
            chunks=True,
            compression='gzip',
            compression_opts=4,
        )
        if solver.last_N_a is not None:
            f.create_dataset(
                'N_a_stage',
                data=cp.asnumpy(solver.last_N_a),
                chunks=True,
                compression='gzip',
                compression_opts=4,
            )
            f.create_dataset(
                'N_b_stage',
                data=cp.asnumpy(solver.last_N_b),
                chunks=True,
                compression='gzip',
                compression_opts=4,
            )
            f.create_dataset(
                'N_c_stage',
                data=cp.asnumpy(solver.last_N_c),
                chunks=True,
                compression='gzip',
                compression_opts=4,
            )

        hist_grp = f.create_group('telemetry')
        hist_grp.create_dataset(
            'step',
            data=np.array([t['step'] for t in history], dtype=np.int64),
            chunks=True,
            compression='gzip',
            compression_opts=4,
        )
        hist_grp.create_dataset(
            'energy',
            data=np.array([t['energy'] for t in history], dtype=np.float64),
            chunks=True,
            compression='gzip',
            compression_opts=4,
        )
        hist_grp.create_dataset(
            'energy_sparkline',
            data=energy_sparkline,
            chunks=True,
            compression='gzip',
            compression_opts=4,
        )
        hist_grp.create_dataset(
            'C_invariant',
            data=np.array([t['C_invariant'] for t in history], dtype=np.float64),
            chunks=True,
            compression='gzip',
            compression_opts=4,
        )

        ext_grp = f.create_group('extended_telemetry')
        if extended_telemetry:
            step_values = np.array([t['step'] for t in extended_telemetry], dtype=np.int64)
            time_values = np.array([t['sim_time'] for t in extended_telemetry], dtype=np.float64)
            dt_values = np.array([t['dt'] for t in extended_telemetry], dtype=np.float64)
            grid_values = np.array([t['grid_shape'] for t in extended_telemetry], dtype=np.int32)
            hash_values = np.array([str(t['params_hash'] or "") for t in extended_telemetry], dtype='S128')
        else:
            step_values = np.array([final_step], dtype=np.int64)
            time_values = np.array([total_time], dtype=np.float64)
            dt_values = np.array([dt], dtype=np.float64)
            grid_values = np.array([N_grid], dtype=np.int32)
            hash_values = np.array([config_hash or ""], dtype='S128')

        ext_grp.create_dataset('step_count', data=step_values, chunks=True, compression='gzip', compression_opts=4)
        ext_grp.create_dataset('sim_time', data=time_values, chunks=True, compression='gzip', compression_opts=4)
        ext_grp.create_dataset('dt', data=dt_values, chunks=True, compression='gzip', compression_opts=4)
        ext_grp.create_dataset('grid_shape', data=grid_values, chunks=True, compression='gzip', compression_opts=4)
        ext_grp.create_dataset('params_hash', data=hash_values, chunks=True, compression='gzip', compression_opts=4)

        solver_contract = {
            "solver_contract_version": SOLVER_CONTRACT_VERSION,
            "geometry_source": "local_stage_rho",
            "auxiliary_geometry": False,
            "topology_cap_in_simulation": False,
            "linear_operator": "-D*k^2 - eta + i*omega0",
        }
        f.create_dataset(
            'solver_contract',
            data=np.array([json.dumps(solver_contract)], dtype='S512'),
        )
        if identity:
            write_identity_group(f, identity)
        f.flush()
    finally:
        f.close()


    os.replace(temp_path, output_path)

    # ASTE D.6: Force NFS metadata sync and explicitly release VRAM for daemon safety
    if hasattr(os, 'sync'):
        os.sync()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    log_lifecycle_event(
        stage="h5_write",
        config_hash=config_hash,
        generation=generation if isinstance(generation, int) else None,
        details={"artifact_url": output_path},
    )
    return {
        "status": "PENDING_VALIDATION",
        "artifact_url": output_path,
        "final_step": final_step,
    }

# =============================================================================
# Interface Entry Point
# =============================================================================

