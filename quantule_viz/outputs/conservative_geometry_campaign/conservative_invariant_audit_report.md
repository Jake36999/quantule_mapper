# Conservative Invariant Audit

Conservative C2 diagnostic-only audit. Dissipative results remain historical geometry priors only.

## Loop Contract

- Conservative CuPy wrapper linear operator: `L_k = -1j * param_D * k^2`.
- Nonlinear RHS is multiplied by `1j`, matching `jax_scout.physics` C2 `kfac = 1j` contract.
- `param_eta` remains present in provenance but is inactive in the conservative linear operator.
- The conservative diagnostic loop does not call dynamic filters or phase centering.
- The ETDRK4 step still applies the existing spectral dealias mask after nonlinear transforms and after recombination; this can remove high-k content and reduce total norm.
- No post-step amplitude normalization is applied.

## Results

| case | outcome | norm change | width mean | profile overlap | centroid drift | pairwise drift | late rho max trend | warning |
|---|---|---:|---:|---:|---:|---:|---:|---|
| c2_triangle_s0.36_N48_T4000_triangle_transfer_036 | stayed_planar | -0.181479 | 0.0839681 | 0.975451 | 0.000551327 | 0.000827952 | -0.264075 | total norm changed by -18.148%; investigate numerical loss/dispersion/dealiasing |
| c2_triangle_s0.45_N48_T4000_triangle_transfer_045 | stayed_planar | -0.186933 | 0.0839954 | 0.97695 | 0.000168502 | 0.000193478 | -0.263559 | total norm changed by -18.693%; investigate numerical loss/dispersion/dealiasing |
| c2_ablated_triangle_s0.45_N48_T4000_ablated_triangle_control | stayed_planar | -0.185926 | 0.084031 | 0.97685 | 0.000376267 | 0.000301278 | -0.2635 | total norm changed by -18.593%; investigate numerical loss/dispersion/dealiasing |
| c2_tetrahedron_s0.45_N48_T4000_tetrahedron_matched_045 | stayed_volumetric | -0.187835 | 0.0839629 | 0.977049 | 0.00137339 | 0.00224273 | -0.262134 | total norm changed by -18.784%; investigate numerical loss/dispersion/dealiasing |
| c2_triangular_prism_s0.45_N48_T4000_triangular_prism_6node_045 | stayed_volumetric | -0.187951 | 0.0837473 | 0.977049 | 0.000590899 | 0.000212384 | -0.263415 | total norm changed by -18.795%; investigate numerical loss/dispersion/dealiasing |

## Recommendation

numerical loss / dealiasing-dispersion warning

## Caveats

- `raw_energy` and `total_norm` are both reported as `sum(abs(psi)^2)` in this diagnostic audit.
- Node widths are estimated from thresholded rho components and are detector-sensitive.
- Profile overlap is against the initial full rho field, not an exact conservative soliton eigenprofile.
- Stability is not claimed from this medium campaign.