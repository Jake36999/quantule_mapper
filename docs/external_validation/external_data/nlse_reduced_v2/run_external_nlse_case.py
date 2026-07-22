import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

shim_dir = Path(sys.argv[1])
source_root = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
config_path = Path(sys.argv[4])
sys.path.insert(0, str(shim_dir))
sys.path.insert(1, str(source_root))

from NLSE import NLSE_1d

config = json.loads(config_path.read_text())
rows = []
histories = {}

def periodic_centroid(x, rho, L):
    theta = 2.0 * np.pi * x / L
    z = np.sum(rho * np.exp(1j * theta)) / np.sum(rho)
    angle = np.angle(z)
    if angle < 0:
        angle += 2.0 * np.pi
    return float(angle * L / (2.0 * np.pi))

def unwrap_positions(pos, L):
    out = [pos[0]]
    for p in pos[1:]:
        q = p
        while q - out[-1] > L / 2:
            q -= L
        while q - out[-1] < -L / 2:
            q += L
        out.append(q)
    return np.array(out, dtype=float)

for kval in config["k_values"]:
    N = int(config["N"])
    L = float(config["L"])
    dt = float(config["dt"])
    T = float(config["T"])
    # NLSE package linear coefficient is D = 1 / (2*k0). Choose wavelength=4*pi so k0=0.5 and D=1.
    simu = NLSE_1d(
        alpha=0.0,
        power=1.0,
        window=L,
        n2=1e-12,
        V=None,
        L=T,
        NX=N,
        Isat=np.inf,
        wvl=4.0 * np.pi,
        backend="CPU",
    )
    simu.n2 = 0.0
    simu.delta_z = dt
    simu.propagator = simu._build_propagator()
    x = simu.X.astype(np.float64)
    x0 = 0.25 * L
    sigma = float(config["sigma"])
    psi0 = np.exp(-0.5 * ((x - x0) / sigma) ** 2) * np.exp(1j * kval * x)
    psi0 = psi0.astype(np.complex128)
    samples = []

    def cb(s, A, z_total, i, samples, kval):
        if i % int(config["sample_every"]) != 0:
            return
        rho = np.abs(np.asarray(A)) ** 2
        samples.append(
            {
                "t": float((i + 1) * s.delta_z),
                "centroid": periodic_centroid(x % L, rho, L),
                "norm": float(np.sum(rho) * s.delta_X),
                "rho_max": float(np.max(rho)),
            }
        )

    rho0 = np.abs(psi0) ** 2
    samples.append(
        {
            "t": 0.0,
            "centroid": periodic_centroid(x % L, rho0, L),
            "norm": float(np.sum(rho0) * simu.delta_X),
            "rho_max": float(np.max(rho0)),
        }
    )
    out = simu.out_field(
        psi0,
        T,
        verbose=False,
        plot=False,
        precision="single",
        normalize=False,
        callback=cb,
        callback_args=(samples, kval),
    )
    rho = np.abs(out) ** 2
    final_t = float(math.ceil(T / dt) * dt)
    samples.append(
        {
            "t": final_t,
            "centroid": periodic_centroid(x % L, rho, L),
            "norm": float(np.sum(rho) * simu.delta_X),
            "rho_max": float(np.max(rho)),
        }
    )
    # Drop duplicate sample times while preserving order.
    dedup = []
    seen = set()
    for item in samples:
        key = round(item["t"], 12)
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    t = np.array([s["t"] for s in dedup], dtype=float)
    c = unwrap_positions([s["centroid"] for s in dedup], L)
    fit = np.polyfit(t, c, 1)
    v = float(fit[0])
    expected = 2.0 * float(config["D"]) * kval
    norm0 = float(dedup[0]["norm"])
    normf = float(dedup[-1]["norm"])
    rows.append(
        {
            "source": "DATA-NUM-001 NLSE package via local CPU FFT compatibility shim",
            "k": kval,
            "v_measured": v,
            "v_expected_2Dk": expected,
            "v_frac": v / expected if expected else None,
            "mass_ret": normf / norm0 if norm0 else None,
            "intercept": float(fit[1]),
            "samples": len(dedup),
        }
    )
    histories[str(kval)] = dedup

out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "external_nlse_rows.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
(out_dir / "external_nlse_history.json").write_text(json.dumps(histories, indent=2))
(out_dir / "external_nlse_runtime.json").write_text(
    json.dumps(
        {
            "package_version": "2.3.0",
            "backend": "CPU",
            "compatibility_shims": ["pyfftw->numpy FFT", "numba->plain Python decorators"],
            "D_convention": "D = 1/(2*k0); wavelength=4*pi gives k0=0.5, D=1",
        },
        indent=2,
    )
)
