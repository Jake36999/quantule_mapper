"""a* continuation pilot (IMPLEMENTATION_PLAN_2026-10 Phase F3).

Question: is the a* standing attractor a RELATIVE EQUILIBRIUM of the fixed ETDRK4 flow (returns to itself
up to a global phase and a shift after time T), and if so, does its branch in param_a change stability
(a Floquet multiplier crossing |mu| = 1) where the B4 replay brackets a* (between x1.15 and x1.16)?

Steps (all on the fixed solver, e270cdc):
  1. settle: evolve the a* multiseed IC (K=6, seed 20260619) at N for T_settle;
  2. Newton-Krylov: solve G = S_a R_theta Phi_T(psi) - psi = 0 from the settled state (jax_scout/continuation.py);
  3. continue in param_a by pseudo-arclength across the bracket;
  4. leading Floquet multipliers at each converged point.

HONEST FAILURE MODES, reported rather than hidden:
  * Newton does not converge from the settled state  -> a* is not a relative equilibrium at this T
    (e.g. it breathes); the pilot reports NOT_A_RELATIVE_EQUILIBRIUM and the residual history.
  * converges but the branch stays stable/unstable across the bracket -> the bracket is not a
    local bifurcation of this branch; reported as such.

Runs on WSL ~/jax_irer. N=32 by default (pilot scale; N=96 is the harness scale).

  python jax_scout/astar_continuation_pilot.py [--N 32] [--T-settle 360] [--T-map 5] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jax_scout import continuation as C, core_saturation_search as css, physics  # noqa: E402
from jax_scout.provenance import write_json  # noqa: E402

HARNESS = {
    "id": "astar_continuation_pilot",
    "branch": "Branch - Stability - Index",
    "status": "ACTIVE",
    "superseded_by": None,
    "invariants": [],
    "produces": ["ASTAR_CONTINUATION_PILOT"],
    "summary": "Is a* a relative equilibrium of the fixed flow, and does its param_a branch change stability across the bracket?",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--T-settle", type=float, default=360.0)
    ap.add_argument("--T-map", type=float, default=5.0, help="return-map time T")
    ap.add_argument("--a-lo", type=float, default=1.13)
    ap.add_argument("--a-hi", type=float, default=1.18)
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "sweep_runs", "ASTAR_CONTINUATION_PILOT_%s" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    dt, L, N = css.DT, css.L_, args.N
    feb_a = float(css.FEB["param_a"])
    P = dict(css.FEB)
    P["param_a"] = feb_a * 1.15
    log = lambda m: print(m, flush=True)  # noqa: E731
    write_json(os.path.join(out, "config.json"), {"args": vars(args), "params": P, "dt": dt, "L": L})

    # 1. settle
    t0 = time.time()
    psi0, _ = css.build_ic(N, 6, seed=20260619)
    ops = physics.build_operators(N, L, dt, P)
    pk = physics.initial_psi_k(jnp.asarray(psi0), ops)
    n_settle = int(round(args.T_settle / dt))
    adv = jax.jit(lambda q: jax.lax.fori_loop(0, 1000, lambda i, z: physics.step(z, ops), q))
    for _ in range(n_settle // 1000):
        pk = adv(pk)
    psi_s = jnp.fft.ifftn(pk)
    log("settled T=%g in %.1f min; mass=%.4g" % (args.T_settle, (time.time() - t0) / 60,
                                                   float(jnp.sum(jnp.abs(psi_s) ** 2))))
    np.save(os.path.join(out, "settled.npy"), np.asarray(psi_s))

    # 2. Newton at a* (fixed p)
    prob = C.RelEqProblem(N=N, L=L, dt=dt, T=args.T_map, params=P)
    x0 = prob.pack(psi_s, 0.0, jnp.zeros(3))
    r = C.newton_krylov(prob, x0, psi_ref=psi_s, tol=1e-8, max_iter=15, log=log)
    result = {"newton_at_astar": {"converged": r.converged, "residuals": r.residuals}}
    if not r.converged:
        result["verdict"] = "NOT_A_RELATIVE_EQUILIBRIUM_AT_T_MAP"
        write_json(os.path.join(out, "summary.json"), result)
        log("=> %s (|F| %.2e -> %.2e)" % (result["verdict"], r.residuals[0], r.residuals[-1]))
        return 0
    psi_c, theta, a, _ = prob.unpack(jnp.asarray(r.x))
    # guard: psi = 0 is ALSO a relative equilibrium (any theta), and Newton can slide onto it -- the
    # validation case did exactly that from the unstable uniform root (tests/test_continuation.py docstring)
    m0, m1 = float(jnp.sum(jnp.abs(psi_s) ** 2)), float(jnp.sum(jnp.abs(psi_c) ** 2))
    result["newton_at_astar"]["mass_ratio"] = m1 / m0
    if m1 < 1e-2 * m0:
        result["verdict"] = "COLLAPSED_TO_TRIVIAL_SOLUTION"
        write_json(os.path.join(out, "summary.json"), result)
        log("=> %s (mass ratio %.2e): needs deflation or an amplitude constraint" % (result["verdict"], m1 / m0))
        return 0
    mults = C.floquet_multipliers(prob, r.x, p=None, k=8)
    result["newton_at_astar"].update(theta=float(theta), shift=np.asarray(a).tolist(), floquet=mults.tolist())
    log("relative equilibrium: theta=%.6g shift=%s |mu|=%s" % (float(theta), np.round(np.asarray(a), 6),
                                                                np.round(mults, 5)))

    # 3. continuation in param_a
    probc = C.RelEqProblem(N=N, L=L, dt=dt, T=args.T_map, params=P, cont_key="param_a")
    pa = feb_a * 1.15
    xa = jnp.concatenate([jnp.asarray(r.x), jnp.atleast_1d(pa)])
    r2 = C.newton_krylov(probc, xa.at[-1].set(pa * 1.002)[:-1], psi_ref=psi_c, p_fixed=pa * 1.002, tol=1e-8,
                         log=log)
    xb = jnp.concatenate([jnp.asarray(r2.x), jnp.atleast_1d(pa * 1.002)])
    branch = C.continue_branch(probc, xa, xb, psi_c, n_points=args.points, log=log, tol=1e-8)
    rows = []
    for x in branch:
        p = float(x[-1])
        m = C.floquet_multipliers(probc, x, p=p, k=8)
        rows.append({"param_a": p, "a_factor": p / feb_a, "floquet": m.tolist()})
        log("  a x%.4f: |mu| = %s" % (p / feb_a, np.round(m, 5)))
    result["branch"] = rows
    result["verdict"] = "BRANCH_COMPUTED"
    write_json(os.path.join(out, "summary.json"), result)
    log("=> %s (%d points) -> %s" % (result["verdict"], len(rows), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
