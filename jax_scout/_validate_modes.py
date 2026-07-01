"""Quick validation that spectral_modes/node_count are not blind (run in WSL2 jax venv)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_scout import stable_collapse as sc

N = 48
kx = np.fft.fftfreq(N) * N
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
KR = np.sqrt(KX**2 + KY**2 + KZ**2)
rng = np.random.default_rng(0)

# proper test: explicit PERIODIC DENSITY structure at k=6 and k=12 (isotropic-ish)
xx = np.arange(N)
Xc, Yc, Zc = np.meshgrid(xx, xx, xx, indexing="ij")
rho_per = 1.0
for k in (6, 12):
    rho_per = rho_per + 0.4 * (np.cos(2*np.pi*k*Xc/N) + np.cos(2*np.pi*k*Yc/N) + np.cos(2*np.pi*k*Zc/N))
rho_per = np.abs(rho_per)
nm, ks = sc.spectral_modes(rho_per)
print(f"periodic density k=6,12 -> spectral_modes: {nm} at k={ks} (expect peaks near 6,12)")
rho_shell = rho_per

x = np.linspace(-5, 5, N)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
blob = np.exp(-(X**2 + Y**2 + Z**2) / 2.0)
print(f"single gaussian blob -> spectral_modes: {sc.spectral_modes(blob)[0]} (expect 0: monotonic)")
print(f"node_count: shell={sc.node_count(rho_shell)}  blob={sc.node_count(blob)}")
# two separated blobs -> node_count should be 2
two = np.exp(-((X-2.5)**2+Y**2+Z**2)/0.5) + np.exp(-((X+2.5)**2+Y**2+Z**2)/0.5)
print(f"two-blob field -> node_count: {sc.node_count(two)} (expect 2)")
