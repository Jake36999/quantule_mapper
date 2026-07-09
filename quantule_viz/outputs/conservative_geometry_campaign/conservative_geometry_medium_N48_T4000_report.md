# Conservative Geometry Campaign conservative_geometry_medium_N48_T4000

Standalone conservative-only Phase D/C2 diagnostic campaign using a CuPy wrapper of the C2 substrate contract.

These results do not inherit or prove the dissipative Phase C triangle behaviour.

## Results

| case | template | outcome | finite | nodes | rho max | raw energy | drift | mass CV | peak CV | z spread | late rho max trend |
|---|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c2_triangle_s0.36_N48_T4000_triangle_transfer_036 | triangle | stayed_planar | True | 3 | 0.488723 | 891.572 | -0.000827952 | 0.000751351 | 0.00253876 | 0 | -0.264075 |
| c2_triangle_s0.45_N48_T4000_triangle_transfer_045 | triangle | stayed_planar | True | 3 | 0.486986 | 870.462 | 2.13788e-05 | 0.00435601 | 0.00254909 | 0 | -0.263559 |
| c2_ablated_triangle_s0.45_N48_T4000_ablated_triangle_control | ablated_triangle | stayed_planar | True | 2 | 0.487025 | 580.624 | 0.000301278 | 0.00307103 | 0.00271906 | 0 | -0.2635 |
| c2_tetrahedron_s0.45_N48_T4000_tetrahedron_matched_045 | tetrahedron | stayed_volumetric | True | 4 | 0.480637 | 1160.1 | 0.00224273 | 4.08257e-16 | 2.00543e-15 | 0.31938 | -0.262134 |
| c2_triangular_prism_s0.45_N48_T4000_triangular_prism_6node_045 | triangular_prism | stayed_volumetric | True | 6 | 0.486105 | 1739.96 | -0.000212384 | 0.00446412 | 0.00254866 | 0.450384 | -0.263415 |

## Invariant/Profile Audit

| case | norm change | max norm change | width mean | profile overlap | centroid drift | pairwise drift | warning |
|---|---:|---:|---:|---:|---:|---:|---|
| c2_triangle_s0.36_N48_T4000_triangle_transfer_036 | -0.181479 | 0.181479 | 0.0839681 | 0.975451 | 0.000551327 | 0.000827952 | total norm changed by -18.148%; investigate numerical loss/dispersion/dealiasing |
| c2_triangle_s0.45_N48_T4000_triangle_transfer_045 | -0.186933 | 0.186933 | 0.0839954 | 0.97695 | 0.000168502 | 0.000193478 | total norm changed by -18.693%; investigate numerical loss/dispersion/dealiasing |
| c2_ablated_triangle_s0.45_N48_T4000_ablated_triangle_control | -0.185926 | 0.185926 | 0.084031 | 0.97685 | 0.000376267 | 0.000301278 | total norm changed by -18.593%; investigate numerical loss/dispersion/dealiasing |
| c2_tetrahedron_s0.45_N48_T4000_tetrahedron_matched_045 | -0.187835 | 0.187835 | 0.0839629 | 0.977049 | 0.00137339 | 0.00224273 | total norm changed by -18.784%; investigate numerical loss/dispersion/dealiasing |
| c2_triangular_prism_s0.45_N48_T4000_triangular_prism_6node_045 | -0.187951 | 0.187951 | 0.0837473 | 0.977049 | 0.000590899 | 0.000212384 | total norm changed by -18.795%; investigate numerical loss/dispersion/dealiasing |

## Caveats

- Node counts are diagnostic detector outputs, not validation-gate certifications.
- The runner is conservative-only and standalone; it does not alter production defaults or configs.
- Plotly HTML uses a CDN script when opened in a browser; JSON data are embedded in the file.
- Unstable, merging, dispersing, and non-finite outcomes are useful campaign observations, not failed automation.