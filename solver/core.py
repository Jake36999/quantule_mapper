"""solver/core.py - ETDRK4Solver, the buffer-owning spectral integrator core (verbatim from worker_cupy.py)."""
import os
import sys
import logging
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from cupyx.scipy.fftpack import get_fft_plan

from .kernels import (
    _as_bool,
    calculate_cov_laplacian_fused,
    calculate_nonlinear_rhs,
    fused_compute_rho,
    fused_process_omega,
    fused_scale_derivative,
    compute_kt_stage_base,
    compute_kt_stage_c,
    combine_kt_etdrk4,
)

from orchestrator.contracts import DEFAULT_PARAM_RHO_VAC, DEFAULT_PARAM_OMEGA0

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gravity'))
try:
    from unified_omega import (
        derive_stable_conformal_factor,
        derive_stable_conformal_factor_with_gradient,
    )
except ImportError:
    raise RuntimeError("unified_omega module required")


class ETDRK4Solver:
    def __init__(self, N_grid, L_domain, dt, params):
        self.N = N_grid
        self.L = L_domain
        self.dt = dt
        self.D_spatial = 3.0
        self.params = params
        self._geometry_params = dict(params)
        # Simulation geometry always uses local-density-only path: fixed bounds, no global-stats cap.
        self._simulation_geometry_params = dict(params)
        self._simulation_geometry_params['param_skip_topology_cap'] = True
        self._global_stats_eps = float(params.get("global_stats_epsilon", 1e-16))
        self._dist_backend = "none"
        self._dist_rank = 0
        self._dist_world_size = 1
        self._cupyx_dist = None
        self._mpi_module = None
        self._mpi_comm = None
        self.distributed_enabled = False
        self._init_distributed_backend()
        if self.distributed_enabled:
            self._geometry_params["global_stats_enabled"] = True
            self._geometry_params["use_global_stats"] = True
            self._geometry_params["distributed_enabled"] = True

        self.D_diff = params.get('param_D', 1.0)
        self.eta = params.get('param_eta', 0.1)
        # rho_vac is the CONFORMAL REFERENCE DENSITY (geometry): Omega^2=(rho_vac/rho)^a.
        self.rho_vac = params.get('param_rho_vac', DEFAULT_PARAM_RHO_VAC)
        # omega0 is the VACUUM OSCILLATOR FREQUENCY (the i*omega0 term in L_k), now an
        # independent knob.  For backward compatibility it defaults to rho_vac so any
        # config predating the split reproduces the historical coupled behaviour exactly.
        self.omega0 = params.get('param_omega0', params.get('param_rho_vac', DEFAULT_PARAM_OMEGA0))
        # Accept both canonical names and hunt-config alias names so that
        # param_splash_coupling / param_splash_fraction from hunt manifests map
        # correctly to the s and f nonlinear coefficients.
        self.a = params.get('param_a', 0.0)
        self.s = params.get('param_s', params.get('param_splash_coupling', 0.0))
        self.f = params.get('param_f', params.get('param_splash_fraction', 0.0))

        k = cp.fft.fftfreq(N_grid, d=L_domain / N_grid).astype(cp.float64) * (2.0 * np.float64(np.pi))
        self.kx, self.ky, self.kz = cp.meshgrid(k, k, k, indexing='ij')
        self.k_sq = self.kx**2 + self.ky**2 + self.kz**2

        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky
        self.ikz = 1j * self.kz
        self.minus_k_sq = -self.k_sq

        self.c_affect = cp.float64(params.get('param_c_affect', 1.0))
        self.c_sq_k_sq = (self.c_affect ** 2) * self.k_sq

        # [ALETHEIA V4.4] Directive 6: Remove Artificial Damping
        self.derivative_filter = cp.ones_like(self.k_sq, dtype=cp.float64)
        self.ikx_filtered = self.ikx.copy()
        self.iky_filtered = self.iky.copy()
        self.ikz_filtered = self.ikz.copy()
        self.minus_k_sq_filtered = self.minus_k_sq.copy()
        self._rho_has_state = False
        self._entropy_link = 0.0 # Disable dynamic entropy filter

        # Stiff flat Laplacian belongs in L_k so ETDRK4 exponentiates it.
        # N_op carries only the geometry correction D(Δ_g - Δ)ψ plus the polynomial nonlinearity.
        self.L_k = (-self.D_diff * self.k_sq + (-self.eta + 1j * self.omega0)).astype(cp.complex128)

        # [ALETHEIA V4.4] Directive 4: Adaptive ETDRK4 Contour (M >= 64)
        M = 64
        theta = cp.exp(1j * cp.pi * (cp.arange(1, M + 1, dtype=cp.float64) - 0.5) / M).astype(cp.complex128)
        r = cp.float64(1.0)

        w = (self.L_k * dt).astype(cp.complex128, copy=False)

        Q_acc = cp.zeros_like(w, dtype=cp.complex128)
        f1_acc = cp.zeros_like(w, dtype=cp.complex128)
        f2_acc = cp.zeros_like(w, dtype=cp.complex128)
        f3_acc = cp.zeros_like(w, dtype=cp.complex128)

        for i in range(M):
            z = r * theta[i]
            w_exp = w + z
            Q_acc += (cp.exp(w_exp / 2.0) - 1.0) / w_exp
            exp_w = cp.exp(w_exp)
            f1_acc += (-4.0 - w_exp + exp_w * (4.0 - 3.0 * w_exp + w_exp**2)) / (w_exp**3)
            f2_acc += (2.0 + w_exp + exp_w * (w_exp - 2.0)) / (w_exp**3)
            f3_acc += (-4.0 - 3.0 * w_exp - w_exp**2 + exp_w * (4.0 - w_exp)) / (w_exp**3)

        self.Q = dt * cp.real(Q_acc / M)
        self.f1 = dt * cp.real(f1_acc / M)
        self.f2 = dt * cp.real(f2_acc / M)
        self.f3 = dt * cp.real(f3_acc / M)

        self.E = cp.exp(w)
        self.E2 = cp.exp(w / 2.0)

        # param_dealias_fraction lets the refinement ladder vary the anti-aliasing cutoff
        # without touching the solver physics.  Default 0.5 preserves existing behaviour.
        _dealias_frac = float(params.get('param_dealias_fraction', 0.5))
        self.dealias_mask = (cp.sqrt(self.k_sq) <= (_dealias_frac * cp.max(cp.sqrt(self.k_sq)))).astype(cp.float64)

        shape = (N_grid, N_grid, N_grid)
        self.N_real_buf = cp.empty(shape, dtype=cp.complex128)
        self.batch_k = cp.empty((5, N_grid, N_grid, N_grid), dtype=cp.complex128)
        self.batch_real = cp.empty((5, N_grid, N_grid, N_grid), dtype=cp.complex128)
        self.rho = cp.empty(shape, dtype=cp.float64)
        self.omega = cp.empty(shape, dtype=cp.float64)
        self.omega_sq = cp.empty(shape, dtype=cp.float64)
        self.A_real = cp.zeros(shape, dtype=cp.float64)
        self.A_k = cp.zeros(shape, dtype=cp.complex128)
        self.A_dot_k = cp.zeros(shape, dtype=cp.complex128)
        self.rho_floor = cp.float64(1e-7)
        self.omega_sq_min = cp.float64(1e-9)
        self.omega_sq_max = cp.float64(1e6)
        self._rho_has_state = False
        self.last_N_a = None
        self.last_N_b = None
        self.last_N_c = None

        self.single_plan = get_fft_plan(self.N_real_buf, axes=(0,1,2))
        self.batch_plan = get_fft_plan(self.batch_k, axes=(1,2,3))

    def _init_distributed_backend(self):
        distributed_requested = _as_bool(self.params.get('distributed_enabled', False)) or _as_bool(
            self.params.get('domain_decomposed', False)
        )
        self.distributed_enabled = distributed_requested
        if not distributed_requested:
            return

        backend_pref = str(self.params.get("distributed_backend", "auto")).strip().lower()

        if backend_pref in {"auto", "cupyx", "nccl"}:
            try:
                import cupyx.distributed as cupyx_dist

                has_all_reduce = hasattr(cupyx_dist, "all_reduce") and callable(getattr(cupyx_dist, "all_reduce"))
                has_rank = hasattr(cupyx_dist, "get_rank") and callable(getattr(cupyx_dist, "get_rank"))
                has_world = hasattr(cupyx_dist, "get_world_size") and callable(getattr(cupyx_dist, "get_world_size"))
                has_init = hasattr(cupyx_dist, "init_process_group") and callable(getattr(cupyx_dist, "init_process_group"))

                if has_all_reduce and has_rank and has_world and has_init:
                    if hasattr(cupyx_dist, "is_initialized") and callable(getattr(cupyx_dist, "is_initialized")):
                        if not cupyx_dist.is_initialized():
                            cupyx_dist.init_process_group(backend="nccl")
                    else:
                        cupyx_dist.init_process_group(backend="nccl")

                    self._cupyx_dist = cupyx_dist
                    self._dist_backend = "cupyx"
                    self._dist_rank = int(cupyx_dist.get_rank())
                    self._dist_world_size = int(cupyx_dist.get_world_size())
                    logging.info(
                        f"Distributed backend initialized: cupyx(rank={self._dist_rank}, world={self._dist_world_size})"
                    )
                    return

                logging.warning(
                    "cupyx.distributed detected but required process-group collectives are unavailable; "
                    "falling back to mpi4py."
                )
            except Exception as exc:
                logging.warning(f"cupyx.distributed initialization failed ({exc}); falling back to mpi4py.")

        if backend_pref in {"auto", "mpi", "mpi4py", "cupyx", "nccl"}:
            try:
                from mpi4py import MPI

                self._mpi_module = MPI
                self._mpi_comm = MPI.COMM_WORLD
                self._dist_backend = "mpi4py"
                self._dist_rank = int(self._mpi_comm.Get_rank())
                self._dist_world_size = int(self._mpi_comm.Get_size())
                logging.info(
                    f"Distributed backend initialized: mpi4py(rank={self._dist_rank}, world={self._dist_world_size})"
                )
                return
            except Exception as exc:
                raise RuntimeError(
                    "Distributed mode requested but no collective backend initialized. "
                    "Install/configure cupyx.distributed (NCCL) or mpi4py."
                ) from exc

        raise RuntimeError(
            f"Unsupported distributed_backend='{backend_pref}'. Use 'auto', 'cupyx', 'nccl', or 'mpi4py'."
        )

    def _compute_global_mu_sigma(self, rho_field):
        local_sum = cp.sum(rho_field, dtype=cp.float64)
        local_sq_sum = cp.sum(rho_field * rho_field, dtype=cp.float64)
        local_count = cp.asarray(rho_field.size, dtype=cp.float64)

        if not self.distributed_enabled or self._dist_world_size <= 1:
            global_sum = local_sum
            global_sq_sum = local_sq_sum
            global_count = local_count
        elif self._dist_backend == "cupyx":
            payload = cp.stack((local_sum, local_sq_sum, local_count)).astype(cp.float64, copy=False)
            reduced = None
            try:
                reduced = self._cupyx_dist.all_reduce(payload)
            except TypeError:
                try:
                    reduced = self._cupyx_dist.all_reduce(payload, op="sum")
                except TypeError:
                    self._cupyx_dist.all_reduce(payload, "sum")
                    reduced = payload

            if reduced is None:
                reduced = payload
            global_sum = reduced[0]
            global_sq_sum = reduced[1]
            global_count = reduced[2]
        elif self._dist_backend == "mpi4py":
            local_arr = cp.asnumpy(cp.stack((local_sum, local_sq_sum, local_count)).astype(cp.float64, copy=False))
            global_arr = np.empty_like(local_arr)
            self._mpi_comm.Allreduce(local_arr, global_arr, op=self._mpi_module.SUM)
            global_sum = cp.asarray(global_arr[0], dtype=cp.float64)
            global_sq_sum = cp.asarray(global_arr[1], dtype=cp.float64)
            global_count = cp.asarray(global_arr[2], dtype=cp.float64)
        else:
            raise RuntimeError("Distributed mode active but no collective backend is available.")

        global_count = cp.maximum(global_count, cp.asarray(1.0, dtype=cp.float64))
        global_mu = global_sum / global_count
        variance = cp.maximum(
            (global_sq_sum / global_count) - (global_mu * global_mu),
            cp.asarray(self._global_stats_eps, dtype=cp.float64),
        )
        global_sigma = cp.sqrt(variance)
        return global_mu, global_sigma

    def fft_single(self, x):
        x_safe = x.astype(self.N_real_buf.dtype, copy=False)
        return cufft.fftn(x_safe, axes=(0,1,2), plan=self.single_plan)

    def ifft_single(self, x_k):
        x_k_safe = x_k.astype(self.batch_k.dtype, copy=False)
        return cufft.ifftn(x_k_safe, axes=(0,1,2), plan=self.single_plan)

    def ifft_batch(self, stack_k):
        return cufft.ifftn(stack_k, axes=(1,2,3), plan=self.batch_plan)

    def _update_derivative_filter(self, alpha_eff):
        alpha_eff = cp.float64(alpha_eff)
        cp.exp(-alpha_eff * self._filter_exp_base, out=self.derivative_filter)
        cp.multiply(self.ikx, self.derivative_filter, out=self.ikx_filtered)
        cp.multiply(self.iky, self.derivative_filter, out=self.iky_filtered)
        cp.multiply(self.ikz, self.derivative_filter, out=self.ikz_filtered)
        cp.multiply(self.minus_k_sq, self.derivative_filter, out=self.minus_k_sq_filtered)

    def update_dynamic_filters(self):
        # Called from outer simulation cadence only; never from N_op/step hot loop.
        if self._entropy_link == 0.0 or not self._rho_has_state:
            return

        rho_sum = cp.sum(self.rho, dtype=cp.float64)
        rho_sum_val = float(rho_sum)
        if not np.isfinite(rho_sum_val) or rho_sum_val <= 0.0:
            return

        rho_prob = self.rho / rho_sum_val
        entropy = -cp.sum(rho_prob * cp.log(rho_prob + self._entropy_eps), dtype=cp.float64)
        entropy_val = float(entropy)
        if not np.isfinite(entropy_val):
            return
        entropy_max = np.log(float(self.rho.size))
        if entropy_max <= 0.0:
            return

        entropy_norm = float(np.clip(entropy_val / entropy_max, 0.0, 1.0))
        span = self._entropy_alpha_max - self._entropy_alpha_min
        alpha_eff = self._entropy_alpha_min + (self._entropy_link * entropy_norm * span)
        alpha_eff = float(np.clip(alpha_eff, self._entropy_alpha_min, self._entropy_alpha_max))

        if self._last_alpha_eff is not None and abs(alpha_eff - self._last_alpha_eff) < self._filter_update_tol:
            return

        self._update_derivative_filter(alpha_eff)
        self._last_alpha_eff = alpha_eff

    def update_field_of_affect(self, rho_k, dt):
        """
        Integrate the auxiliary wave equation in spectral space:
        d2(A_k)/dt2 = -c^2 * k^2 * A_k + rho_k

        k=0 secular-runaway gate
        -------------------------
        At k=0 the wave operator has no restoring force (c^2 k^2 = 0), while the
        source rho_k[0,0,0] = integral(rho dV) is the (positive, ~constant) total
        mass.  Left unprojected this drives an unbounded quadratic-in-time zero
        mode A_k[0,0,0] ~ 1/2 (M) t^2 -> a diverging uniform offset in A_real.
        The affect field is physically the *response to density inhomogeneities*;
        its constant mode is pure gauge.  We therefore (a) remove the DC source
        before forcing and (b) pin the A / A_dot zero modes to zero each step.
        Canonical helper + regression proof: orchestrator.run_identity.zero_dc_mode
        and tests/test_run_identity.py::TestK0Runaway.
        """
        dt64 = cp.float64(dt)
        rho_k_safe = rho_k.astype(cp.complex128, copy=False)
        # (a) Project out the total-mass DC source (zero-mean causal forcing).
        rho_k_safe[0, 0, 0] = 0

        acceleration_k = -self.c_sq_k_sq * self.A_k + rho_k_safe
        self.A_dot_k += acceleration_k * dt64
        self.A_k += self.A_dot_k * dt64
        self.A_k *= self.dealias_mask
        # (b) Pin the gauge: keep the zero mode identically zero.
        self.A_k[0, 0, 0] = 0
        self.A_dot_k[0, 0, 0] = 0
        self.A_real[:] = self.ifft_single(self.A_k).real.astype(cp.float64, copy=False)
        return self.A_k

    def N_op(self, psi_k):
        """
        Calculates N(psi) directly from the spectral state.
        Uses batched cuFFT transforms and single fused kernels to decimate overhead.
        """
        # 1. Build derivative spectra in-place
        cp.copyto(self.batch_k[0], psi_k)
        cp.multiply(self.ikx_filtered, psi_k, out=self.batch_k[1])
        cp.multiply(self.iky_filtered, psi_k, out=self.batch_k[2])
        cp.multiply(self.ikz_filtered, psi_k, out=self.batch_k[3])
        cp.multiply(self.minus_k_sq_filtered, psi_k, out=self.batch_k[4])

        # 2. Batched Transform (1 invocation produces all 5 real-space arrays)
        self.batch_real[:] = self.ifft_batch(self.batch_k)
        
        psi = self.batch_real[0]
        grad_x = self.batch_real[1]
        grad_y = self.batch_real[2]
        grad_z = self.batch_real[3]
        lap_flat = self.batch_real[4]

        # 3. Algebraic Geometry Sub-Pipeline with Preallocated Buffers
        self.rho[:] = fused_compute_rho(psi, self.rho_floor)
        self._rho_has_state = True

        # Geometry is derived from local stage density ρ = |ψ|² only.
        # _simulation_geometry_params disables the global μ+3σ topology cap.
        omega_sq_tmp, d_omega_sq_d_rho_tmp = derive_stable_conformal_factor_with_gradient(
            self.rho, self._simulation_geometry_params
        )
        self.omega_sq[:], self.omega[:] = fused_process_omega(
            omega_sq_tmp,
            self.omega_sq_min,
            self.omega_sq_max,
        )

        # Apply smooth conformal boundary window to ∂Ω²/∂ρ, then convert to ∂Ω/∂ρ.
        # unified_omega returns ∂Ω²/∂ρ; the conformal Laplacian needs ∂Ω/∂ρ = (1/2Ω) ∂Ω²/∂ρ.
        d_omega_sq_d_rho = fused_scale_derivative(
            self.omega_sq,
            d_omega_sq_d_rho_tmp,
            self.omega_sq_min,
            self.omega_sq_max,
        )
        d_omega_d_rho = d_omega_sq_d_rho / (2.0 * cp.maximum(self.omega, cp.float64(1e-15)))

        # 4. Covariant Laplacian (Fully Fused)
        lap_cov = calculate_cov_laplacian_fused(
            psi, grad_x, grad_y, grad_z, lap_flat, 
            self.omega, self.omega_sq, d_omega_d_rho, self.D_spatial
        )

        # 5. Synthesize Non-linear Field Operator
        self.N_real_buf[:] = calculate_nonlinear_rhs(
            psi, self.rho, lap_cov, lap_flat,
            self.D_diff, self.a, self.s, self.f
        ).astype(self.N_real_buf.dtype, copy=False)

        # 6. Single Spectral Transform & Dealiasing
        N_k = self.fft_single(self.N_real_buf)
        N_k *= self.dealias_mask
        
        return N_k

    def step(self, psi_k):
        """Kassam-Trefethen ETDRK4 Integrator operating natively in Spectral Space"""
        
        # --- Stage A (n) ---
        N_n = self.N_op(psi_k)
        
        # --- Stage B (a) ---
        a_k = compute_kt_stage_base(self.E2, psi_k, self.Q, N_n)
        N_a = self.N_op(a_k)
        self.last_N_a = N_a  # zero-copy reference: no device copy in hot loop
        
        # --- Stage C (b) ---
        b_k = compute_kt_stage_base(self.E2, psi_k, self.Q, N_a)
        N_b = self.N_op(b_k)
        self.last_N_b = N_b  # zero-copy reference
        
        # --- Stage D (c) ---
        c_k = compute_kt_stage_c(self.E2, a_k, self.Q, N_b, N_a)
        N_c = self.N_op(c_k)
        self.last_N_c = N_c  # zero-copy reference

        # --- Recombination ---
        psi_next_k = combine_kt_etdrk4(psi_k, N_n, N_a, N_b, N_c, self.E, self.f1, self.f2, self.f3)
        psi_next_k *= self.dealias_mask

        return psi_next_k.astype(self.batch_k.dtype, copy=False)

# =============================================================================
# Worker Orchestration & Telemetry 
# =============================================================================

