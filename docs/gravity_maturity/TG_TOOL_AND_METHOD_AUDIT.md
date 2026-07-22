# TG Tool And Method Audit

Timestamp: 2026-07-14.

| tool | purpose | version | selected | reason | analytic validation case |
| --- | --- | --- | --- | --- | --- |
| numpy | TG-A contract checks, synthetic source nulls, TG-B0 ODE integration | 2.4.6 | yes | small analytic arrays and deterministic reduced models do not require GPU | linear T/G eigenvalues and ledger residuals |
| jax | TG-B1 full field pilot only | recorded by TG-B1 preflight | yes-for-field-evolution | full field evolution should use the established GPU mirror discipline | flat KG dispersion and source-off null |
| scipy/sympy | optional future symbolic derivation/eigensolver checks | not required for this TG-A run | no-for-current-contract | closed-form T/G stability is available from a 2x2 stiffness matrix | not used |
| existing C3 KG scripts | node lineage and baseline KG/Q-ball reference | repo-local | reference-only | TG-B1 pilot uses a reduced 1D KG node, but references C3 as the validated KG substrate family | C3 energy/charge conservation and Q-ball transport reports |