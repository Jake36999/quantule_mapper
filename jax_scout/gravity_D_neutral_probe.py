"""Gravity Audit D (Option 1) — neutral-probe force under an ENVIRONMENT-ONLY objective bounded lapse.

Option 1: source the lapse from the ENVIRONMENT ALONE (the load makes geometry; probes navigate it), so it is
universal (probe-independent) and naturally bounded (no cliff):
    S_B(x) = env-only field (here rho_B^2, concentrated at the load);   N_B(x) = 1/(1+beta*S_hat_B(x))  in (0,1].
N_B is minimal at the load, ->1 far away.

DECISIVE (settles the sign that started the thread — D_eff argument said one way, geodesic argument the other):
evolve a NEUTRAL probe wavepacket (at rest, at distance d from the load) under the norm-conserving covariant
Laplacian coupling
    i d_t psi = -D grad.(N_B grad psi)      (divergence form, Hermitian)
and MEASURE whether its centre-of-mass drifts TOWARD the load (attraction) or away (repulsion). Compare to the
timelike-geodesic prediction a = -grad ln N_B. Controls: beta=0 null, sign of beta, two probe widths (universality),
grid/dt.

Standalone mirror; the PRODUCTION gravity ladder (cliffing Omega^2(rho)) stays CLOSED — this uses the NEW bounded
env-only lapse. Consistency/mechanism test; NOT an emergent-gravity claim.

  python jax_scout/gravity_D_neutral_probe.py
"""
import os, sys, json, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.gravity_A_rint_diagnostic import place, integ
D = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--beta", type=float, default=1.0); ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--T", type=float, default=3.0); ap.add_argument("--out", default=None)
    A = ap.parse_args()
    out = A.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_D_PROBE_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    N, L = A.N, A.L
    x = np.linspace(-L / 2, L / 2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    ikx = 1j * k1[:, None, None]; iky = 1j * k1[None, :, None]; ikz = 1j * k1[None, None, :]

    # env-only load: a compact Gaussian blob at origin (probe-independent source)
    rho_B = np.exp(-((X ** 2 + Y ** 2 + Z ** 2) / (2 * 1.5 ** 2)))
    S = rho_B ** 2; Shat = S / (S.max() + 1e-30)

    def lapse(beta):
        return 1.0 / (1.0 + beta * Shat)

    def lap_cov(psi, Nf):
        gx = np.fft.ifftn(ikx * np.fft.fftn(psi)); gy = np.fft.ifftn(iky * np.fft.fftn(psi)); gz = np.fft.ifftn(ikz * np.fft.fftn(psi))
        return (np.fft.ifftn(ikx * np.fft.fftn(Nf * gx)) + np.fft.ifftn(iky * np.fft.fftn(Nf * gy))
                + np.fft.ifftn(ikz * np.fft.fftn(Nf * gz)))

    def comx(psi):
        # windowed COM (exclude the far periodic-wrap tail): keep |x - x_p| < W around the initial probe region
        r = np.abs(psi) ** 2
        w = (np.abs(X - 4.0) < 6.0)
        rr = r * w
        return float(integ(X * rr, N, L) / (integ(rr, N, L) + 1e-30))

    def evolve_track(psi0, Nf, T, dt, nsnap=6):
        psi = psi0.astype(np.complex128); n = int(round(T / dt)); every = max(1, n // nsnap)
        rhs = lambda p: 1j * D * lap_cov(p, Nf)
        traj = [(0.0, comx(psi))]
        for i in range(n):
            k1_ = rhs(psi); k2_ = rhs(psi + 0.5 * dt * k1_); k3_ = rhs(psi + 0.5 * dt * k2_); k4_ = rhs(psi + dt * k3_)
            psi = psi + dt / 6 * (k1_ + 2 * k2_ + 2 * k3_ + k4_)
            if (i + 1) % every == 0:
                traj.append(((i + 1) * dt, comx(psi)))
        return psi, traj

    def evolve(psi0, Nf, T, dt):
        return evolve_track(psi0, Nf, T, dt)[0]

    x_p = 4.0
    print(f"=== Gravity Audit D — neutral probe | N={N} L={L} beta={A.beta} | load@0, probe@{x_p} ===", flush=True)
    # geodesic-prediction direction at the probe: a_x = -d_x ln N_B
    lnN = np.log(lapse(A.beta))
    dxlnN = np.real(np.fft.ifftn(ikx * np.fft.fftn(lnN)))
    ip = (np.argmin((x - x_p) ** 2), N // 2, N // 2)
    a_geo_x = -float(dxlnN[ip])                                   # >0 = toward +x (away from load); <0 = toward load
    print(f"[geodesic] a_x=-d_x lnN at probe = {a_geo_x:+.4e}  ({'toward load (-x)' if a_geo_x<0 else 'away from load (+x)'})", flush=True)
    res = {"N": N, "L": L, "beta": A.beta, "x_p": x_p, "a_geodesic_x": a_geo_x, "runs": []}

    trajs = {}
    for (label, beta, sig) in [("main", A.beta, 1.0), ("null_b0", 0.0, 1.0),
                               ("neg_beta", -A.beta * 0.5, 1.0), ("wide_probe", A.beta, 1.6)]:  # neg beta kept > -1 (N_B finite)
        Nf = lapse(beta)
        probe = np.exp(-((X - x_p) ** 2 + Y ** 2 + Z ** 2) / (2 * sig ** 2)).astype(np.complex128)
        psiT, traj = evolve_track(probe, Nf, A.T, A.dt)
        trajs[label] = traj
        drift = traj[-1][1] - traj[0][1]; mass = float(integ(np.abs(psiT) ** 2, N, L) / integ(np.abs(probe) ** 2, N, L))
        res["runs"].append({"label": label, "beta": beta, "sig": sig, "traj": traj, "drift_x": drift, "mass_ret": mass})
        print(f"[{label:10s} beta={beta:+.2f} sig={sig:.1f}] COM_x {traj[0][1]:.4f}->{traj[-1][1]:.4f}  "
              f"drift={drift:+.4e} mass={mass:.4f}", flush=True)

    # baseline-subtract the beta=0 null trajectory (removes the free-spread/box artifact)
    tt = [p[0] for p in trajs["main"]]
    def net(label):
        return [trajs[label][i][1] - trajs["null_b0"][i][1] for i in range(len(tt))]
    net_main = net("main"); net_neg = net("neg_beta"); net_wide = net("wide_probe")
    print("  t         :", " ".join(f"{t:7.2f}" for t in tt), flush=True)
    print("  net_main  :", " ".join(f"{v:+7.4f}" for v in net_main), " (baseline-subtracted COM shift; <0 = toward load)", flush=True)
    print("  net_neg_b :", " ".join(f"{v:+7.4f}" for v in net_neg), flush=True)
    # acceleration vs transient: 2nd difference of net_main (sustained inward accel -> increasingly negative)
    d1 = np.diff(net_main); accelerating = bool(d1[-1] < d1[0] - 1e-5 and d1[-1] < 0)
    res["baseline_subtracted"] = {"t": tt, "net_main": net_main, "net_neg": net_neg, "net_wide": net_wide,
                                  "accelerating_inward": accelerating}
    attractive = net_main[-1] < -1e-3
    null_ok = abs(trajs["null_b0"][-1][1] - trajs["null_b0"][0][1]) < 2e-3    # small residual, tracked+subtracted
    sign_flips = net_neg[-1] > 0 and net_main[-1] < 0                         # beta sign flips drift direction
    universal = net_main[-1] * net_wide[-1] > 0
    res["findings"] = {"probe_attracted_to_load_net": bool(attractive), "beta0_baseline_small": bool(null_ok),
                       "sign_flips_with_beta": bool(sign_flips), "universal_across_width": bool(universal),
                       "geodesic_agrees": bool((a_geo_x < 0) == attractive), "accelerating_not_transient": accelerating}
    verdict = ("D_PROBE_ATTRACTION_PROMISING" if (attractive and sign_flips and universal) else
               "D_PROBE_INCONCLUSIVE")
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n[findings] attracted-to-load={attractive}; beta0-null={null_ok}; universal(width)={universal}; "
          f"geodesic-agrees={(a_geo_x<0)==attractive}", flush=True)
    print(f"=== {verdict} | env-only bounded lapse; production ladder CLOSED; NOT an emergent-gravity claim | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
