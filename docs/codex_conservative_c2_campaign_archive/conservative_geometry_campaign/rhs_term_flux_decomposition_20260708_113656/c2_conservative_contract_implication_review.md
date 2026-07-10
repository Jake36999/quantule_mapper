# C2 Conservative Contract Implication Review

Final decision: `DISCRETE_OPERATOR_ADJOINT_FAILURE`
Dominant nonlinear flux term: `geometry_covariant_correction`
Dominant term flux: `-0.0003885781015571192`
Worst adjoint proxy: `frozen_conformal_laplacian_proxy` classified `ADJOINT_FAIL`

## Answers

1. The dominant term is reported above from the recombined term audit.
2. Local density terms are pointwise phase-only under real coefficients; the geometry correction requires a self-adjoint effective operator to be norm-neutral.
3. The geometry correction is derivative/covariant/geometry-dependent; the remaining polynomial terms are local multiplicative density terms.
4. The operator audit is a frozen-coefficient proxy and is reported separately; it does not patch or redefine the RHS.
5. The current code comments indicate a conservative C2 kinetic mode, but this audit is meant to clarify whether that applies to the nonlinear sector for generic states.
6. If the dominant term is non-neutral and documentation does not explicitly define quasi-conservative nonlinear behavior, the conservative label remains contract-ambiguous.
7. Longer geometry campaigns should remain blocked until Jake/Claude review the dominant nonlinear flux term and intended invariant contract.
8. Question for Jake/Claude: Should C2 `kinetic_mode='conservative'` conserve total `sum(|psi|^2)` for the full nonlinear geometry-corrected RHS, or only for the linear dispersive substrate / special symmetric states?

No fix is proposed or applied in this diagnostic.