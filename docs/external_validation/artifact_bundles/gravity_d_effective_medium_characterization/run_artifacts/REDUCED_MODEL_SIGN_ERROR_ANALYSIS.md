# Reduced-Model Sign-Error Examination

Model examined: `F_cg = c * (-K_grad * grad_N_at_COM)`, with `c = 0.2800836375056448` from `model_comparison.csv`.

Sign-error count: 6 out of 193 non-flat atlas rows.

All sign errors are small-magnitude far-tail or near-null cases. They are concentrated in compact-width super-Gaussian far-field rows, plus one shell far-exterior row and one strong Gaussian far-tail row. This supports the interpretation that the coarse-grained local-COM approximation is excellent in resolved-gradient regions but can miss sign in weak tail regions where the integrated exact force is tiny and finite-width weighting samples nonlocal gradients.

| run_id | source | N_min | r | sigma | exact Fr | predicted Fr |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1_core_supergaussian4_r4p0_w0p25_n0p5 | supergaussian4 | 0.5 | 6.52561530883978 | 0.407850956802486 | 2.43705519701282E-05 | -1.16407703107332E-05 |
| E1_core_supergaussian4_r4p0_w0p25_n0p65 | supergaussian4 | 0.65 | 6.26186681226581 | 0.391366675766613 | -1.57398571986153E-05 | 8.10252594494437E-06 |
| E1_core_supergaussian4_r4p0_w0p25_n0p35 | supergaussian4 | 0.35 | 6.8878556075833 | 0.430490975473956 | 4.72298030282053E-06 | -4.95292883238558E-07 |
| E1_dist_shell_r5p0 | shell | 0.5 | 12.6246534113492 | 1 | 7.60666239413678E-07 | -2.26878644464881E-08 |
| E1_core_supergaussian4_r4p0_w0p25_n0p9 | supergaussian4 | 0.9 | 5.93907489753621 | 0.371192181096013 | 5.30543761715534E-07 | -2.32450027317517E-07 |
| E1_core_gaussian_r4p0_w0p25_n0p35 | gaussian | 0.35 | 8.084527114973 | 0.505282944685813 | -1.45690924395844E-09 | 2.32711693976735E-10 |

Documentation consequence: retain `D_REDUCED_MODEL_PARTIAL`; do not promote the local coarse-grained law as universal. The exact operator-force identity remains the primary model.
