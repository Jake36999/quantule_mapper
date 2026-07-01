"""
Characterize the corrected-solver gain-regime structured states (NOT prime-matching).

Loads the 128^3 escalated fields and measures WHAT the self-organized structure is,
using the production spectral tools but without forcing the prime-SSE target:
  - radial power profile + detected peak k-positions and their SPACING pattern
  - dominant wavelength, spectral slope (cascade vs coherent), spectral entropy
  - anisotropy (directional power), spatial coherence (autocorrelation length)
  - number of distinct density structures (connected components)

Goal: identify the natural observable/order of the gain-regime structure, to replace
the inherited prime-SSE lens. Native .venv (cupy + scipy).

Usage: python jax_scout/characterize.py [--sweepdir <dir>]
"""
import os
import sys
import csv
import glob
import argparse
import numpy as np
import scipy.ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def latest_sweepdir():
    return sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "CORRECTED_PHYSICS_JAX_SCOUT_*")))[-1]


def correlation_length(rho):
    """Isotropic autocorrelation length: +x radius where normalized autocorr drops to 1/e."""
    f = rho - rho.mean()
    ac = np.fft.fftshift(np.fft.ifftn(np.abs(np.fft.fftn(f)) ** 2).real)
    ac /= ac.max()
    c = np.array(ac.shape) // 2
    line = ac[c[0], c[1], c[2]:]
    below = np.where(line < 1.0 / np.e)[0]
    return int(below[0]) if below.size else rho.shape[0] // 2


def characterize(psi, N):
    import quantulemapper_real as qm
    import cupy as cp
    rho = np.abs(psi) ** 2
    power = qm.compute_power_spectrum(cp.asarray(rho))
    power_np = np.asarray(qm._to_numpy(power))
    profile = np.asarray(qm._to_numpy(qm.radial_profile(power)))
    nyq = N // 2

    peaks = sorted(float(x) for x in qm.detect_peaks(profile, nyquist_radius=nyq))
    ent = float(qm.spectral_entropy(profile))
    slope = float(qm.spectral_slope(profile))
    p = profile[2:nyq]
    kdom = int(np.argmax(p)) + 2 if p.size else 0
    spacing = [round(float(s), 3) for s in np.diff(peaks)] if len(peaks) >= 2 else []

    thr = rho.mean() + 2 * rho.std()
    _, ncomp = ndi.label(rho > thr)

    dpow = [float(power_np.sum(axis=tuple(i for i in range(3) if i != ax)).mean()) for ax in range(3)]
    anis = float(np.var(dpow) / (np.mean(dpow) ** 2 + 1e-30))
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return {"n_radial_peaks": len(peaks),
            "peak_ks": [round(x, 2) for x in peaks][:8],
            "peak_spacing": spacing[:6],
            "dominant_k": kdom,
            "dominant_wavelength_cells": round(N / kdom, 2) if kdom else 0.0,
            "spectral_slope": round(slope, 3),
            "spectral_entropy": round(ent, 3),
            "anisotropy": round(anis, 4),
            "corr_length_cells": correlation_length(rho),
            "n_structures": int(ncomp)}


def main():
    import h5py
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweepdir", default=None)
    args = ap.parse_args()
    sd = args.sweepdir or latest_sweepdir()
    files = sorted(glob.glob(os.path.join(sd, "escalation_data", "rho_scout_*.h5")))
    print(f"characterizing {len(files)} gain-regime 128^3 fields from {sd}\n")

    rows = []
    for fp in files:
        sid = os.path.basename(fp).replace("rho_scout_", "").replace(".h5", "")
        with h5py.File(fp, "r") as f:
            psi = f["psi_final"][:]
        if not np.all(np.isfinite(psi)):
            print(f"  scout_{sid}: non-finite (blow-up), skipped"); continue
        c = characterize(psi, psi.shape[0]); c["scout_idx"] = sid
        rows.append(c)
        print(f"scout_{sid}: {c['n_radial_peaks']} radial peaks at k={c['peak_ks']} spacing={c['peak_spacing']}")
        print(f"   dom_k={c['dominant_k']} (lambda~{c['dominant_wavelength_cells']} cells)  slope={c['spectral_slope']} "
              f"entropy={c['spectral_entropy']}  corr_len={c['corr_length_cells']} "
              f"n_struct={c['n_structures']} anis={c['anisotropy']}")

    if rows:
        keys = ["scout_idx", "n_radial_peaks", "dominant_k", "dominant_wavelength_cells",
                "spectral_slope", "spectral_entropy", "corr_length_cells", "n_structures",
                "anisotropy", "peak_ks", "peak_spacing"]
        out = os.path.join(sd, "structure_characterization.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in keys})
        doms = [r["dominant_k"] for r in rows]
        print(f"\ndominant_k across fields: {doms}  -> {np.mean(doms):.1f} +/- {np.std(doms):.1f}"
              f"  ({'CONSISTENT shared scale' if np.std(doms) < 2 else 'varied scales'})")
        slopes = [r["spectral_slope"] for r in rows]
        ms = np.mean(slopes)
        # ~ -5/3 turbulent cascade; much steeper (< -3) = spectrally compact / coherent
        regime = ("spectrally-compact coherent (steep)" if ms < -3 else
                  "turbulent cascade (~-5/3)" if ms < -1 else "shallow/flat")
        print(f"spectral slopes: {ms:.2f} +/- {np.std(slopes):.2f}  ({regime})")
        print(f"written -> {out}")


if __name__ == "__main__":
    main()
