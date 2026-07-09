"""IRER geometry-density gravity ladder — RUNG A (coherent-load baseline) + RUNG D (geometry-null control).
Per docs/IRER_GEOMETRY_DENSITY_GRAVITY_RFC.md. NOT an emergent-gravity claim; a candidate-mechanism baseline.

Rung A: does a large coherent load create a PERSISTENT geometry (Omega^2) + tensor (T_info) response? Does the
        response fall with distance? Does tensor alignment persist in time?
Rung D: same density layout, geometry response neutralized (param_geom_off, the C2.6-verified true-flat switch).
        Only the geometry-ON minus geometry-OFF difference counts as the geometry-density mechanism.

Read-only production telemetry (the RFC substrate, used not modified):
  gravity.unified_omega.derive_stable_conformal_factor   -> Omega^2(rho) (the shared geometric field)
  metrics.tensor_validation.construct_T_info/symmetry/shear -> informational stress tensor response
  metrics.collapse_dynamics.compute_correlation_length   -> interaction-density proxy
Substrate = DISSIPATIVE production geometry (a_coupling=feb=2.31, the real IRER conformal law); load = a relaxed
multi-blob standing cluster (Phase C style). Mirror-only evolution; no production solver/Hunter/validation changes.

  wsl:  python jax_scout/gravity_ladder_A_D.py [--N 64 --Trelax 4000 --Tprobe 4000 --out DIR]
"""
import os, sys, json, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css, physics
from jax_scout.phase_d_c1_transport import _evolve_chunk
import gravity.unified_omega as uo
import metrics.tensor_validation as tv
import metrics.collapse_dynamics as cd

L_DEFAULT = css.L_


def geom_params():
    """Production conformal-geometry params (simulation path: local pointwise, skip topology cap)."""
    return {"param_rho_vac": float(css.FEB.get("param_rho_vac", 1.0)),
            "param_a_coupling": float(css.FEB.get("param_a_coupling", 2.3098)),
            "param_conformal_softclip_beta": float(css.FEB.get("param_conformal_softclip_beta", 3.0)),
            "param_omega_sq_min": 1e-9, "param_omega_sq_max": 1e6, "param_skip_topology_cap": True}


def omega_sq_field(rho):
    return np.asarray(uo.derive_stable_conformal_factor(rho, geom_params()))


def build_ops(N, L, dt, geom_off, a_mult=1.15):
    p = {**css.FEB, "param_a": float(css.FEB["param_a"]) * a_mult}      # dissipative default (production geometry ON)
    if geom_off:
        p["param_geom_off"] = True                                     # rung-D null: geom_fac=0 (C2.6-verified flat)
    return physics.build_operators(N, L, dt, p)


def multiblob_seed(N, L, n_blob=5, amp=0.9, spread=0.28, sig=0.9, bg=0.0, seed=20260709):
    rng = np.random.default_rng(seed)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    psi = np.full((N, N, N), np.sqrt(max(bg, 0.0)), dtype=np.complex128)   # optional rho~bg background (bg=rho_vac -> Omega^2~1 ambient)
    for _ in range(n_blob):
        cx, cy, cz = (rng.uniform(-spread, spread, 3) * L)
        psi += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * sig ** 2))
    psi += 0.01 * (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))
    return psi.astype(np.complex128)


def radial_profile(field, N, L, nbins=24, center=None):
    """Spherically-averaged profile of `field` about `center` (default = density-peak of field itself)."""
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    if center is None:
        center = np.unravel_index(int(np.argmax(np.abs(field))), field.shape)
    d = []
    for ax in range(3):
        dd = x - x[center[ax]]; dd = dd - L * np.round(dd / L); d.append(dd)
    R = np.sqrt(d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2)
    rmax = L / 2
    edges = np.linspace(0, rmax, nbins + 1)
    prof, rc = [], []
    for i in range(nbins):
        m = (R >= edges[i]) & (R < edges[i + 1])
        if m.any():
            prof.append(float(field[m].mean())); rc.append(0.5 * (edges[i] + edges[i + 1]))
    return np.array(rc), np.array(prof)


def response_metrics(psi, N, L):
    """All read-only geometry/tensor telemetry for a field snapshot. Omega^2 is referenced to the FAR-FIELD
    ambient (not to 1): in the low-background sim regime vacuum Omega^2 saturates at the cap, and the coherent
    load appears as a WELL (depressed Omega^2) whose depth/falloff is the geometry response."""
    rho = np.abs(psi) ** 2
    cen = np.unravel_index(int(rho.argmax()), rho.shape)
    om = omega_sq_field(rho)
    rc, omp = radial_profile(om, N, L, center=cen)
    om_far = float(np.mean(omp[-3:])) if len(omp) >= 3 else float(np.median(om))   # ambient (outer radial bins)
    om_center = float(om[cen])
    well = om_far - om_center                                           # >0 => load depresses Omega^2 (geometry well)
    dev_ambient = np.abs(om - om_far)
    # falloff: e-folding length of the normalized deviation |omp - om_far| from the load centre outward
    dprof = np.abs(omp - om_far); fall = np.nan
    if dprof[0] > 1e-3 * (abs(om_far) + 1e-30) and len(rc) > 5:
        norm = dprof / (dprof[0] + 1e-30); good = norm > 1e-3
        if good.sum() >= 4:
            cfit = np.polyfit(rc[good], np.log(norm[good] + 1e-30), 1)
            fall = float(-1.0 / cfit[0]) if cfit[0] < 0 else np.inf
    T = tv.construct_T_info(rho, np.angle(psi))
    sym = tv.tensor_symmetry_test(T); shear = tv.perfect_fluid_reduction_test(T)
    Tmean = T.mean(axis=(2, 3, 4))
    evals = np.sort(np.linalg.eigvalsh(0.5 * (Tmean + Tmean.T)))
    aniso = float((evals[-1] - evals[0]) / (np.abs(evals).sum() + 1e-30))
    xi = cd.compute_correlation_length(rho)
    return {"mass": float(rho.sum()), "amp": float(np.abs(psi).max()), "n_nodes": _nnodes(psi, L / N),
            "omega_center": om_center, "omega_far": om_far, "omega_well": float(well),
            "omega_well_frac": float(well / (abs(om_far) + 1e-30)),
            "omega_dev_integ": float(dev_ambient.sum()) / dev_ambient.size,
            "tensor_sym_err": sym, "tensor_shear": shear, "tensor_aniso": aniso,
            "corr_length": xi, "omega_falloff_len": fall,
            "radial_r": [round(float(v), 3) for v in rc], "radial_omega": [round(float(v), 3) for v in omp]}


def _nnodes(psi, dx):
    from jax_scout import transfer_diag as td
    return len(td.detect_nodes(psi, dx))


def evolve_track(psi0, ops, N, L, dt, steps, dt_chunk, tag, out, sample_every=4):
    pk = physics.initial_psi_k(jnp.asarray(psi0), ops); cur = psi0
    traj = []
    nch = steps // dt_chunk
    for c in range(nch):
        pk = _evolve_chunk(pk, ops, dt_chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            traj.append({"t_step": (c + 1) * dt_chunk, "collapsed": True}); break
        if (c % sample_every == 0) or (c == nch - 1):
            m = response_metrics(cur, N, L); m["t_step"] = (c + 1) * dt_chunk
            traj.append({k: m[k] for k in ("t_step", "mass", "amp", "n_nodes", "omega_well",
                                           "omega_dev_integ", "tensor_shear", "tensor_aniso",
                                           "corr_length", "omega_falloff_len")})
    np.save(os.path.join(out, f"load_{tag}.npy"), cur)
    return cur, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--L", type=float, default=L_DEFAULT)
    ap.add_argument("--dt", type=float, default=0.004); ap.add_argument("--Trelax", type=int, default=4000)
    ap.add_argument("--Tprobe", type=int, default=4000); ap.add_argument("--dtchunk", type=int, default=500)
    ap.add_argument("--bg", type=float, default=0.0, help="uniform background density rho (set ~param_rho_vac for an "
                    "unsaturated ambient Omega^2~1); default 0 = low-background sim regime")
    ap.add_argument("--out", default=None)
    a_ = ap.parse_args()
    N, L, dt = a_.N, a_.L, a_.dt
    out = a_.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_LADDER_AD_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    print(f"=== GRAVITY LADDER A+D | N={N} L={L} dt={dt} | production geometry a_coupling={geom_params()['param_a_coupling']} "
          f"| out={out} ===", flush=True)

    # --- build the coherent load: relax a multi-blob seed in the dissipative production substrate (geometry ON) ---
    ops_on = build_ops(N, L, dt, geom_off=False)
    seed = multiblob_seed(N, L, bg=a_.bg)
    print(f"[load] relaxing multi-blob seed (geometry ON, a*x1.15, bg={a_.bg}) for {a_.Trelax} steps...", flush=True)
    t0 = time.time()
    load, _ = evolve_track(seed, ops_on, N, L, dt, a_.Trelax, a_.dtchunk, "relaxed", out, sample_every=10 ** 9)
    load_m = response_metrics(load, N, L)
    print(f"[load] settled: mass={load_m['mass']:.1f} amp={load_m['amp']:.3f} n_nodes={load_m['n_nodes']} "
          f"({(time.time()-t0)/60:.1f}m)", flush=True)

    # --- RUNG A: characterize the geometry + tensor response of the load ---
    print("[A] coherent-load geometry/tensor response:", flush=True)
    print(f"    Omega^2 WELL: center={load_m['omega_center']:.2f} ambient~{load_m['omega_far']:.2f} "
          f"well={load_m['omega_well']:.2f} (frac {load_m['omega_well_frac']:+.3f}) "
          f"falloff_len={load_m['omega_falloff_len']} box-units", flush=True)
    print(f"    T_info: shear={load_m['tensor_shear']:.3e} aniso={load_m['tensor_aniso']:.3f} "
          f"sym_err={load_m['tensor_sym_err']:.2e} | corr_length={load_m['corr_length']:.2f}", flush=True)
    # effective geometry law characterization (C2.6 prerequisite): Omega^2 vs rho over the load
    rho = np.abs(load) ** 2
    om = omega_sq_field(rho)
    p = geom_params(); nominal = (p["param_rho_vac"] / np.maximum(rho, 1e-12)) ** p["param_a_coupling"]
    law = {"rho_range": [float(rho.min()), float(rho.max())],
           "omega_sq_range": [float(om.min()), float(om.max())],
           "nominal_omega_sq_range": [float(nominal.min()), float(nominal.max())],
           "median_ratio_impl_over_nominal": float(np.median(om / np.maximum(nominal, 1e-30)))}
    print(f"    geometry law (impl vs nominal): Omega^2 impl {law['omega_sq_range']} vs nominal "
          f"{law['nominal_omega_sq_range']} (median impl/nominal={law['median_ratio_impl_over_nominal']:.2f})", flush=True)

    # --- RUNG A persistence + RUNG D null: evolve load geometry-ON vs geometry-OFF, compare response ---
    print(f"[A/D] evolving load geometry-ON vs geometry-OFF for {a_.Tprobe} steps...", flush=True)
    ops_off = build_ops(N, L, dt, geom_off=True)
    _, traj_on = evolve_track(load, ops_on, N, L, dt, a_.Tprobe, a_.dtchunk, "probe_on", out)
    _, traj_off = evolve_track(load, ops_off, N, L, dt, a_.Tprobe, a_.dtchunk, "probe_off", out)

    def last(tr, k):
        vals = [q.get(k) for q in tr if not q.get("collapsed") and q.get(k) is not None]
        return vals[-1] if vals else np.nan
    on_shear, off_shear = last(traj_on, "tensor_shear"), last(traj_off, "tensor_shear")
    on_dev, off_dev = last(traj_on, "omega_dev_integ"), last(traj_off, "omega_dev_integ")
    on_mass, off_mass = last(traj_on, "mass"), last(traj_off, "mass")
    on_aniso, off_aniso = last(traj_on, "tensor_aniso"), last(traj_off, "tensor_aniso")
    print(f"[A] persistence (geometry-ON end vs start): mass {load_m['mass']:.1f}->{on_mass:.1f} "
          f"shear {load_m['tensor_shear']:.2e}->{on_shear:.2e} omega_dev {load_m['omega_dev_integ']:.3f}->{on_dev:.3f}", flush=True)
    print(f"[D] geometry-ON vs geometry-OFF (end): shear {on_shear:.3e} vs {off_shear:.3e} | "
          f"omega_dev {on_dev:.4f} vs {off_dev:.4f} | mass {on_mass:.1f} vs {off_mass:.1f} | "
          f"aniso {on_aniso:.3f} vs {off_aniso:.3f}", flush=True)

    # verdict (candidate-support only; interaction-density/tensor first, acceleration never here)
    persistent = np.isfinite(on_dev) and on_dev > 0.5 * load_m["omega_dev_integ"] and np.isfinite(on_mass)
    distinct = (np.isfinite(on_shear) and np.isfinite(off_shear)
                and abs(on_shear - off_shear) > 0.1 * max(abs(on_shear), abs(off_shear), 1e-30)) or \
               (np.isfinite(on_mass) and np.isfinite(off_mass) and abs(on_mass - off_mass) > 0.05 * load_m["mass"])
    verdict = ("A_D_GEOMETRY_RESPONSE_PERSISTENT_AND_DISTINCT" if persistent and distinct else
               "A_D_RESPONSE_PERSISTENT_BUT_GEOMETRY_INERT" if persistent and not distinct else
               "A_D_LOAD_NOT_PERSISTENT" if not persistent else "A_D_INCONCLUSIVE")
    summary = {"load": load_m, "geometry_law": law, "traj_on": traj_on, "traj_off": traj_off,
               "rungD": {"on_shear": on_shear, "off_shear": off_shear, "on_omega_dev": on_dev,
                         "off_omega_dev": off_dev, "on_mass": on_mass, "off_mass": off_mass,
                         "on_aniso": on_aniso, "off_aniso": off_aniso},
               "verdict": verdict}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} ===", flush=True)
    print(f"GRAVITY_AD_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
