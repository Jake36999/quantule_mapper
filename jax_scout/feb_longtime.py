"""
DECISIVE long-time test of feb56dc7 -- the hi-fi 'stable rotating core' (strong v_t~0.5). It is bare
S-NCGL (gamma_A=0; eta=+0.07 loss-side but a=0.48 strong cubic gain). Question: does this STRONGLY-
rotating core SATURATE (true steady soliton) or also keep GROWING at T=6000 like the basin sustainers?
Reuses core_characterize bare capture + core metrics. WSL2 jax venv: python jax_scout/feb_longtime.py
"""
import os, sys, json, time
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.afield_current_coupled import multiseed_ic, L as L_
from jax_scout import core_characterize as cc

PAR = {"param_D": 2.7329, "param_eta": 0.0704, "param_rho_vac": 1.1866, "param_omega0": 0.0,
       "param_a_coupling": 2.3098, "param_s": 0.0129, "param_f": -0.4861, "param_a": 0.4802}
N, T, NSNAP = cc.N, 6000, 40


def main():
    dx = L_ / N; psi0 = multiseed_ic(N, 20260619)
    t0 = time.time()
    snaps, fin = cc.capture(PAR, psi0, T, NSNAP)
    nfin = max((t for t in range(snaps.shape[0]) if np.all(np.isfinite(np.abs(snaps[t])))), default=0) + 1
    e0 = float(np.sum(np.abs(snaps[0]) ** 2)) + 1e-30
    ser = []
    for t in range(nfin):
        psi = snaps[t]; c, nn = cc.core_at(psi, PAR, dx)
        er = float(np.sum(np.abs(psi) ** 2) / e0); row = {"t_step": t * (T // NSNAP), "er": er, "n_nodes": nn}
        if c is not None:
            row.update(cc.shell_metrics(psi, PAR, c, dx, 1, 4)[0])
        ser.append(row)
    ers = [r["er"] for r in ser]; last = ser[len(ser)//2:]
    slope = (last[-1]["er"] - last[0]["er"]) / max(1, last[-1]["t_step"] - last[0]["t_step"])
    outcome = ("BLEW_UP" if nfin < snaps.shape[0] else "SATURATED" if abs(slope) < 2e-5
               else "STILL_GROWING" if slope > 0 else "DECAYING")
    out = {"config": "feb56dc7 (bare S-NCGL, gamma_A=0)", "params": PAR, "N": N, "T": T,
           "n_finite": nfin, "er_final": ers[-1], "er_max": max(ers), "late_slope_per_step": slope,
           "outcome": outcome, "series": ser}
    d = os.path.join(ROOT, "sweep_runs", "SUBSTRATE_HUNT_20260621_161557", "feb56dc7_longtime.json")
    json.dump(out, open(d, "w"), indent=2, default=float)
    cd = [r.get("core_rho") for r in ser]; vt = [r.get("v_t") for r in ser]
    print(f"feb56dc7 bare T->{(nfin-1)*(T//NSNAP)}: er {ers[0]:.2f}->{ers[-1]:.2f} (max {max(ers):.2f}) "
          f"slope={slope:+.2e} -> {outcome}")
    print(f"  core_density {cd[0]:.2f}->{cd[-1]:.2f} | v_t {vt[0] if vt[0] is None else round(vt[0],3)}->{vt[-1] if vt[-1] is None else round(vt[-1],3)} | nodes {ser[0]['n_nodes']}->{ser[-1]['n_nodes']}  ({time.time()-t0:.0f}s)")
    print(f"wrote {d}")


if __name__ == "__main__":
    main()
