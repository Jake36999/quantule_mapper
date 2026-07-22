# Tool And Method Audit

Status: initial G0/G1 audit.

| purpose | tool | version | selected | reason | analytic validation case |
| --- | --- | ---: | --- | --- | --- |
| sparse generalized eigenproblem | SciPy sparse eigsh | 1.17.1 | yes | present in WSL JAX environment; sufficient for local clock eigensolves | flat `N_t=1` local KG clock eigenfrequency |
| array processing | NumPy | 2.4.6 | yes | present; used for CPU assembly and metrics | direct finite-difference operator checks |
| GPU time-domain field evolution | JAX/JAXLIB | 0.10.2 | yes | established project GPU backend | GPU preflight plus eigen/time contract |
| symbolic variation | SymPy | unavailable | no | not installed; not needed for G0/G1 | defer to G2 |
| HDF5 artifact storage | h5py | unavailable | no | not installed; CSV/JSON sufficient for bounded G1 | not applicable |
| Zarr/xarray storage | zarr/xarray | unavailable | no | not installed; not needed for local-clock runs | not applicable |

No new packages were installed.
