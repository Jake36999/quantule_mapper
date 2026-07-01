#!/usr/bin/env python3
"""
worker_cupy.py - thin CLI shim.

The ETDRK4 SNCGL solver was modularized into the solver/ package
(solver.kernels, solver.core, solver.run). This entrypoint is preserved so
worker_daemon.py's `python worker_cupy.py --manifest ... --output ...` keeps working.
Public names are re-exported for backward compatibility.
"""
import argparse
import json
import logging
import os

from solver.kernels import (  # noqa: F401  (back-compat re-exports)
    calculate_cov_laplacian_fused,
    calculate_nonlinear_rhs,
    fused_compute_rho,
    fused_process_omega,
    fused_scale_derivative,
    compute_kt_stage_base,
    compute_kt_stage_c,
    combine_kt_etdrk4,
)
from solver.core import ETDRK4Solver  # noqa: F401
from solver.run import initialize_psi, run_simulation  # noqa: F401
from orchestrator.run_identity import build_identity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ASTE ETDRK4 GPU Solver")
    parser.add_argument("--params", required=False, help="Path to raw parameters JSON")
    parser.add_argument("--manifest", required=False, help="Path to distributed job manifest JSON")
    parser.add_argument("--output", required=False, help="Path to output HDF5 artifact")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    if not args.params and not args.manifest:
        parser.error("Must provide either --params or --manifest")

    params = {}
    manifest_config_hash = None
    manifest_generation = None
    manifest_hunt_name = None
    manifest_run_id = None


    output_path = args.output

    # 1. Distributed Manifest Parsing
    if args.manifest:
        with open(args.manifest, 'r') as f:
            manifest_data = json.load(f)
        params = manifest_data.get('params', {})
        manifest_config_hash = manifest_data.get("config_hash")
        manifest_generation = manifest_data.get("generation")
        manifest_hunt_name = manifest_data.get("hunt_name")
        manifest_run_id = manifest_data.get("job_id")
        config_hash = manifest_data.get("config_hash")
        if not config_hash:
            parser.error("Manifest is missing config_hash")
        if not output_path:
            output_path = f"rho_history_{config_hash}.h5"
            
    # 2. Legacy Raw Params Parsing
    elif args.params:
        with open(args.params, 'r') as f:
            params = json.load(f)
        config_hash = params.get("config_hash")
        if not config_hash:
            parser.error("Params missing config_hash")
        if not output_path:
            output_path = f"rho_history_{config_hash}.h5"
    
    # Ensure output directory exists
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Extract simulation geometry & time parameters from manifest/config payload.
    # Aggressive key extraction to prevent casing drift
    sim_params = params.get('simulation', {})
    n_grid = sim_params.get('n_grid') or sim_params.get('N_grid', 64)
    t_steps = sim_params.get('t_steps') or sim_params.get('T_steps', 250)
    dt = sim_params.get('dt', 0.001)
    l_domain = sim_params.get('l_domain') or sim_params.get('L_domain', 10.0)
    GLOBAL_SEED = int(params.get('global_seed', 42))

    N_GRID = int(n_grid)
    L_DOMAIN = float(l_domain)
    T_STEPS = int(t_steps)
    DT = float(dt)

    # Flatten nested physics parameters into a flat dictionary
    psi_params_clean = {}
    invalid_param_keys = []

    def extract_params(source_dict):
        for k, v in source_dict.items():
            if isinstance(v, dict): 
                extract_params(v)
            elif k.startswith('param_') or k == 'collapse_threshold':
                if v is None:
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                try:
                    psi_params_clean[k] = float(v)
                except (TypeError, ValueError):
                    invalid_param_keys.append(str(k))

    extract_params(params)

    if invalid_param_keys:
        parser.error(
            "Invalid numeric value(s) for physics parameters: " + ", ".join(sorted(set(invalid_param_keys)))
        )
    
    logging.info(f"Initializing ETDRK4 Worker -> Grid: {N_GRID}^3, Steps: {T_STEPS}")

    # Build the canonical run identity (stamped into the HDF5 /identity group).
    # GLOBAL_SEED is the seed actually fed to initialize_psi, so it is the
    # reproducibility-relevant seed for this artifact.
    identity = build_identity(
        config_hash=config_hash,
        seed=GLOBAL_SEED,
        generation=manifest_generation,
        N_grid=N_GRID,
        dt=DT,
        T_steps=T_STEPS,
        params=psi_params_clean,
        run_id=manifest_run_id,
        hunt_name=manifest_hunt_name,
    )

    # Launch Mathematical Core
    result_payload = run_simulation(
        N_GRID,
        L_DOMAIN,
        T_STEPS,
        DT,
        GLOBAL_SEED,
        psi_params_clean,
        output_path,
        config_hash=manifest_config_hash,
        generation=manifest_generation,
        identity=identity,
    )
    logging.info(f"Worker result payload: {json.dumps(result_payload, sort_keys=True)}")
