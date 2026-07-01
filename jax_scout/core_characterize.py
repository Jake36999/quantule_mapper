"""
SINGLE-CORE CHARACTERIZATION (Phase 1) — what IS the self-sustaining rotational core?
Bare S-NCGL (no A, no coupling, no solver mod). Targets = robust N=96 sustainers (idx 1194, 1125, 488).

Probes:
  1. LONG-TIME PERSISTENCE (T~6000, >> the 1600 validation window): does the core saturate into a
     true steady dissipative soliton, or keep growing toward blow-up / decay? Track er, core density,
     v_r, v_t, node count over time.
  2. RADIAL PROFILE rho(r), v_r(r), v_t(r) around the dominant core (the 'anatomy': dense core +
     circulation shell + outer layer) at early/mid/late times.
  3. PERTURBATION RESPONSE: kick the settled core (phase bump) and measure whether it returns to the
     attractor (core density / v_t recover) vs the unperturbed continuation.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/core_characterize.py
"""
import os, sys, csv, glob, json, time
from functools import partial
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td
from jax_scout.afield_current_coupled import multiseed_ic, L as L_, dt as DT
from jax_scout.core_basin_sweep import order

SEED = 20260619
N, T_LONG, NSNAP = 96, 6000, 40           # snapshot every 150 steps
T_PERT, NSNAP_P = 2000, 40
TARGETS = ["1194", "1125", "488"]
PRIMARY = "1194"


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def _capture_bare(pvec, psi0, N, L, dt, n_steps, n_snap, rd, cd):
    ops = physics._ops_from_vec(pvec, N, L, dt, rd, cd)
    psi_k = jnp.fft.fftn(psi0) * ops.dealias_mask
    stride = n_steps // n_snap

    def outer(pk, _):
        def inner(p, _): return physics.step(p, ops), None
        pk, _ = lax.scan(inner, pk, None, length=stride)
        return pk, jnp.fft.ifftn(pk)
    psi_k, snaps = lax.scan(outer, psi_k, None, length=n_snap)
    return snaps, jnp.all(jnp.isfinite(jnp.abs(snaps[-1])))


def capture(par, psi0, n_steps, n_snap):
    pv = jnp.asarray([par[k] for k in order])
    snaps, fin = _capture_bare(pv, jnp.asarray(psi0), N, L_, DT, n_steps, n_snap, jnp.float64, jnp.complex128)
    return np.concatenate([np.asarray(psi0)[None], np.asarray(snaps)], 0), bool(fin)


def _rel(coord, c):
    return ((coord - c + N / 2) % N) - N / 2


def _vfields(psi, par, dx):
    geo = td.geometry_fields(psi, par, dx); rho = geo["rho"]; Jx, Jy, Jz = geo["J"]
    rs = np.maximum(rho, 1e-6)
    return rho, Jx / rs, Jy / rs, Jz / rs


def core_at(psi, par, dx):
    nodes = sorted(td.detect_nodes(psi, dx), key=lambda n: -n["E"])
    if not nodes:
        return None, 0
    return (np.round(nodes[0]["centroid"]).astype(int) % N), len(nodes)


def shell_metrics(psi, par, c, dx, r_in, r_out):
    rho, vx, vy, vz = _vfields(psi, par, dx)
    ax = np.arange(N)
    DX = _rel(ax[:, None, None], c[0]) * np.ones((N, N, N)); DY = _rel(ax[None, :, None], c[1]) * np.ones((N, N, N)); DZ = _rel(ax[None, None, :], c[2]) * np.ones((N, N, N))
    rr = np.sqrt(DX**2 + DY**2 + DZ**2) + 1e-9
    rhx, rhy, rhz = DX / rr, DY / rr, DZ / rr
    vr = vx * rhx + vy * rhy + vz * rhz
    vt = np.sqrt((vx - vr*rhx)**2 + (vy - vr*rhy)**2 + (vz - vr*rhz)**2)
    sh = (rr >= r_in) & (rr <= r_out)
    return {"core_rho": float(rho[tuple(c % N)]), "v_r": float(np.mean(vr[sh])) if np.any(sh) else 0.0,
            "v_t": float(np.mean(vt[sh])) if np.any(sh) else 0.0,
            "circulation": float(np.sum(vt[sh])) if np.any(sh) else 0.0}, (rho, vr, vt, rr)


def radial_profile(psi, par, c, dx, nbins=14, rmax=14):
    rho, vx, vy, vz = _vfields(psi, par, dx)
    ax = np.arange(N)
    DX = _rel(ax[:, None, None], c[0]) * np.ones((N, N, N)); DY = _rel(ax[None, :, None], c[1]) * np.ones((N, N, N)); DZ = _rel(ax[None, None, :], c[2]) * np.ones((N, N, N))
    rr = np.sqrt(DX**2 + DY**2 + DZ**2) + 1e-9
    rhx, rhy, rhz = DX / rr, DY / rr, DZ / rr
    vr = vx*rhx + vy*rhy + vz*rhz
    vt = np.sqrt((vx - vr*rhx)**2 + (vy - vr*rhy)**2 + (vz - vr*rhz)**2)
    edges = np.linspace(0, rmax, nbins + 1); prof = {"r": [], "rho": [], "v_r": [], "v_t": []}
    for i in range(nbins):
        m = (rr >= edges[i]) & (rr < edges[i+1])
        if np.any(m):
            prof["r"].append(0.5*(edges[i]+edges[i+1])); prof["rho"].append(float(np.mean(rho[m])))
            prof["v_r"].append(float(np.mean(vr[m]))); prof["v_t"].append(float(np.mean(vt[m])))
    return prof


def _bump(c):
    G = np.meshgrid(*([np.arange(N)]*3), indexing="ij")
    return sum(np.minimum((G[a]-c[a]) % N, (c[a]-G[a]) % N).astype(float)**2 for a in range(3))


def phase_kick(psi, c, r, th=0.5):
    return (psi * np.exp(1j*th*np.exp(-_bump(c)/(2*(r/1.5)**2)))).astype(np.complex128)


def load_params(idx):
    for r in csv.DictReader(open(sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "CORE_BASIN_2026*")))[0] + "/all_evals.csv")):
        if r["idx"] == idx:
            return {k: float(r[k]) for k in order}
    return None


def main():
    d = [c for c in sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "CORE_BASIN_2026*"))) if "REFINE" not in c and "CALIB" not in c][-1]
    rows = {r["idx"]: {k: float(r[k]) for k in order} for r in csv.DictReader(open(os.path.join(d, "all_evals.csv")))}
    dx = L_ / N; psi0 = multiseed_ic(N, SEED)
    outdir = os.path.join(d, "core_characterize"); os.makedirs(outdir, exist_ok=True)
    report = {"N": N, "T_long": T_LONG, "targets": {}}
    frames_save = {}
    for idx in TARGETS:
        par = rows[idx]; t0 = time.time()
        snaps, fin = capture(par, psi0, T_LONG, NSNAP)
        nfin = max((t for t in range(snaps.shape[0]) if np.all(np.isfinite(np.abs(snaps[t])))), default=0) + 1
        e0 = float(np.sum(np.abs(snaps[0])**2)) + 1e-30
        series = []
        for t in range(nfin):
            psi = snaps[t]; c, nn = core_at(psi, par, dx)
            er = float(np.sum(np.abs(psi)**2) / e0)
            row = {"t_step": t * (T_LONG // NSNAP), "er": er, "n_nodes": nn}
            if c is not None:
                m, _ = shell_metrics(psi, par, c, dx, 1, 4); row.update(m)
            series.append(row)
        ers = [r["er"] for r in series]
        # saturation test: er slope over last third
        last = series[len(series)//2:]
        slope = (last[-1]["er"] - last[0]["er"]) / max(1, last[-1]["t_step"] - last[0]["t_step"])
        outcome = ("BLEW_UP" if nfin < snaps.shape[0] else
                   "SATURATED" if abs(slope) < 2e-5 else ("STILL_GROWING" if slope > 0 else "DECAYING"))
        report["targets"][idx] = {"eta": par["param_eta"], "n_finite": nfin, "n_snap_total": snaps.shape[0],
                                  "er_final": ers[-1], "er_max": max(ers), "late_slope_per_step": slope,
                                  "outcome": outcome, "series": series}
        print(f"[idx {idx} eta={par['param_eta']:+.3f}] T->{(nfin-1)*(T_LONG//NSNAP)}: er {ers[0]:.2f}->{ers[-1]:.2f} "
              f"(max {max(ers):.2f}) late_slope={slope:+.2e} -> {outcome}  nodes {series[0]['n_nodes']}->{series[-1]['n_nodes']}  ({time.time()-t0:.0f}s)", flush=True)
        if idx == PRIMARY:
            frames_save[f"psi_{idx}"] = snaps[np.linspace(0, nfin-1, 8).astype(int)].astype(np.complex64)
            # radial profiles at early/mid/late
            profs = {}
            for lab, t in (("early", 2), ("mid", nfin//2), ("late", nfin-1)):
                psi = snaps[t]; c, _ = core_at(psi, par, dx)
                if c is not None:
                    profs[lab] = {"t_step": t*(T_LONG//NSNAP), **radial_profile(psi, par, c, dx)}
            report["primary_profiles"] = profs
            # perturbation: settled state ~ mid, kick core, compare recovery vs unperturbed
            t_settle = nfin//2; psi_s = snaps[t_settle]; c, _ = core_at(psi_s, par, dx)
            base, _ = capture(par, psi_s, T_PERT, NSNAP_P)
            kicked, _ = capture(par, phase_kick(psi_s, c, 3), T_PERT, NSNAP_P)
            pert = []
            for t in range(min(base.shape[0], kicked.shape[0])):
                cb, _ = core_at(base[t], par, dx); ck, _ = core_at(kicked[t], par, dx)
                mb = shell_metrics(base[t], par, cb, dx, 1, 4)[0] if cb is not None else {}
                mk = shell_metrics(kicked[t], par, ck, dx, 1, 4)[0] if ck is not None else {}
                pert.append({"t_step": t*(T_PERT//NSNAP_P), "base_core": mb.get("core_rho"), "kick_core": mk.get("core_rho"),
                             "base_vt": mb.get("v_t"), "kick_vt": mk.get("v_t")})
            report["primary_perturbation"] = {"t_settle_step": t_settle*(T_LONG//NSNAP), "series": pert}
            print(f"  primary profiles + perturbation done", flush=True)
    np.savez_compressed(os.path.join(outdir, "core_frames.npz"), **frames_save)
    json.dump(report, open(os.path.join(outdir, "core_characterize.json"), "w"), indent=2, default=float)
    print(f"\nwrote {outdir}/core_characterize.json + core_frames.npz")
    print("outcomes: " + ", ".join(f"{i}:{report['targets'][i]['outcome']}" for i in TARGETS))


if __name__ == "__main__":
    main()
