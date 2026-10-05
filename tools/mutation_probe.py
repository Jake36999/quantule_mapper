"""H3 -- do the physics-identity tests actually catch the bug classes that have bitten this project?

WHY NOT `mutmut`. The plan named mutmut, and mutmut is the right tool for "what fraction of arbitrary
edits does the suite notice". That is not the question the plan is really asking. The question is
whether the identity tests would have caught **C2.6, C2.8b and C3** -- three bugs that actually
happened, cost months, and were each found by chasing a contradiction rather than by a test. Blind
mutation answers that only by accident, buried in thousands of irrelevant mutants.

So the mutations here are not arbitrary. Each one reproduces the SHAPE of a bug from the project's
own instrument-integrity ledger, injected into the code the identity tests cover:

  C2.6 class   a coefficient silently scaled, or a flag that does not take effect. The original was
               `Ops.geom_fac` / `param_geom_off` leaving D_eff = D/151, which produced five
               campaigns' worth of "pinning/drag" verdicts that were all retracted.
  C2.8b class  the DYNAMICS stay correct and an OBSERVABLE is computed wrongly. The original was a
               bespoke peak-tracker that failed on overlapping cores, giving elasticity 3.21 and an
               apparent energy-conservation violation that was never in the physics.
  C3 class     a conservation law quietly broken -- charge or energy no longer conserved.
  P2 class     a plane or index placed half a cell wrong. The original opened the momentum ledger
               at O(1) and was caught only because two flux variants disagreed.
  sign class   a polarity flipped. The sector's central open problem IS a sign, so a test suite
               that cannot feel a sign flip is not guarding the thing that matters.

HOW TO READ THE OUTPUT. A mutation that leaves the suite GREEN is a **survivor**: the tests do not
constrain that line, and a real bug of that shape would pass CI. Survivors are the finding here.
A mutation that turns the suite RED is caught, which is the desired outcome and is not interesting
beyond the count.

A survivor is not automatically a defect -- some lines genuinely are not the identity tests' job.
The value is in knowing WHICH, rather than assuming the suite covers what its name suggests.

Safety: each mutation is applied to the file, the suite is run, and the file is restored in a
`finally`, so an interrupt cannot leave mutated physics on disk. The script refuses to start if the
working tree already has uncommitted changes to the files it edits, because then "restore" has no
well-defined meaning.

Usage (WSL jax_irer venv; CPU so it does not fight a GPU run for the device):
    JAX_PLATFORMS=cpu python tools/mutation_probe.py
    JAX_PLATFORMS=cpu python tools/mutation_probe.py --list
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Identity tests guard conservation laws and nulls; stepper-order tests guard accuracy. Both run
#: against every mutation, so a stepper bug that keeps the identities green is still caught.
TESTS = ["tests/test_physics_identities.py", "tests/test_stepper_order_jax.py"]

#: (label, bug-class, relative path, exact text to find, replacement)
MUTATIONS = [
    # --- C2.6 class: a coefficient silently scaled / a flag that does not take effect -----------
    ("geom_flag_ignored", "C2.6",
     "jax_scout/gravity_TG_B2_two_node_awell.py",
     "    A = jnp.exp(a_sign * eps_G * G * geom_en * fb_en)",
     "    A = jnp.exp(a_sign * eps_G * G * fb_en)"),
    ("coupling_scaled_by_151", "C2.6",
     "jax_scout/gravity_TG_B2_two_node_awell.py",
     "    A = jnp.exp(a_sign * eps_G * G * geom_en * fb_en)",
     "    A = jnp.exp(a_sign * eps_G * G * geom_en * fb_en / 151.0)"),
    ("feedback_flag_ignored", "C2.6",
     "jax_scout/gravity_TG_B2_two_node_awell.py",
     "    A = jnp.exp(a_sign * eps_G * G * geom_en * fb_en)",
     "    A = jnp.exp(a_sign * eps_G * G * geom_en)"),

    # --- sign class: the sector's central open problem is a sign --------------------------------
    ("a_sign_polarity_flipped", "sign",
     "jax_scout/gravity_TG_B2_two_node_awell.py",
     "    A = jnp.exp(a_sign * eps_G * G * geom_en * fb_en)",
     "    A = jnp.exp(-a_sign * eps_G * G * geom_en * fb_en)"),
    ("mass_term_sign_flipped", "sign",
     "jax_scout/gravity_TG_B2_two_node_awell.py",
     "b1s.div_A_grad(phi, A, g) - m * m * phi +",
     "b1s.div_A_grad(phi, A, g) + m * m * phi +"),
    ("kappa_backreaction_sign", "sign",
     "jax_scout/gravity_TG_B2_two_node_awell.py",
     "            - kappa * T - absorb * VG) * geom_en",
     "            + kappa * T - absorb * VG) * geom_en"),

    # --- C3 class: a conservation law quietly broken ---------------------------------------------
    ("kg_dispersion_drops_mass", "C3",
     "jax_scout/phase_d_c3_wave.py",
     "    w = np.sqrt(c ** 2 * k_sq + m ** 2)",
     "    w = np.sqrt(c ** 2 * k_sq)"),
    ("kg_propagator_detuned", "C3",
     "jax_scout/phase_d_c3_wave.py",
     "    C = np.cos(w * dt)",
     "    C = np.cos(w * dt * 1.001)"),

    # --- C2.8b class: dynamics fine, OBSERVABLE wrong --------------------------------------------
    ("energy_drops_gradient_term", "C2.8b",
     "jax_scout/gravity_TG_B2_midplane_stress_flux.py",
     "    E_grad_R = jnp.sum(jnp.where(RM, e_grad, 0.0)) * dV",
     "    E_grad_R = jnp.sum(jnp.where(RM, e_grad, 0.0)) * dV * 0.0"),
    ("energy_kinetic_double_counted", "C2.8b",
     "jax_scout/gravity_TG_B2_midplane_stress_flux.py",
     "    E_kin_R = jnp.sum(jnp.where(RM, e_kin, 0.0)) * dV",
     "    E_kin_R = jnp.sum(jnp.where(RM, e_kin, 0.0)) * dV * 2.0"),

    # --- stepper class (added 2026-10-04): the equation is right, the integrator is inaccurate --
    # The two ETDRK4 bugs of October 2026 (docs/instrument_integrity/ETDRK4_INTEGRATOR_BUGS_2026-10.md)
    # kept every identity green. These reproduce their shapes, plus the analogous slips in the
    # other active steppers, and must be caught by tests/test_stepper_order_jax.py.
    ("etdrk4_stage_c_uses_Na", "stepper",
     "jax_scout/physics.py",
     "ops.Q * (2.0 * n_b - n_n)",
     "ops.Q * (2.0 * n_b - n_a)"),
    ("etdrk4_contour_real_part", "stepper",
     "jax_scout/physics.py",
     "    Q = (dt * Q_acc / M).astype(cd)",
     "    Q = (dt * jnp.real(Q_acc / M)).astype(cd)"),
    ("kg_strang_kick_asymmetric", "stepper",
     "jax_scout/phase_d_c3_wave.py",
     "        pi_new = kick_half(psi_new, pi_new)",
     "        pi_new = kick_half(psi_k, pi_new)"),
    ("tg_b1s_rk4_stage3_uses_k1", "stepper",
     "jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py",
     "    s3 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k2))",
     "    s3 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k1))"),
    ("tg_b2_rk4_stage3_uses_k1", "stepper",
     "jax_scout/gravity_TG_B2_two_node_awell.py",
     "    s3 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k2))",
     "    s3 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k1))"),
    ("gravity_d_rk4_stage3_uses_k1", "stepper",
     "jax_scout/gravity_D_neutral_probe_gpu.py",
     "    k3 = rhs(psi + 0.5 * dt * k2, Nf, grid)",
     "    k3 = rhs(psi + 0.5 * dt * k1, Nf, grid)"),
]


def git_dirty(paths):
    r = subprocess.run(["git", "status", "--porcelain", "--"] + list(paths),
                       cwd=ROOT, capture_output=True, text=True)
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def run_suite(timeout, tests=None):
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", *(tests or TESTS), "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
                           cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def main():
    ap = argparse.ArgumentParser(description="targeted mutation probe for the physics identities")
    ap.add_argument("--list", action="store_true", help="print the mutations and exit")
    ap.add_argument("--only", help="substring filter on the label")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tests", nargs="+", default=None,
                    help="override the test files (e.g. identities only, to measure what they miss)")
    ap.add_argument("--out", default="runtime_logs/mutation_probe.json")
    args = ap.parse_args()

    muts = [m for m in MUTATIONS if not args.only or args.only in m[0]]
    if args.list:
        for label, cls, path, old, new in muts:
            print("%-28s %-6s %s" % (label, cls, path))
            print("    -  %s" % old.strip())
            print("    +  %s" % new.strip())
        return 0

    files = sorted({m[2] for m in muts})
    dirty = git_dirty(files)
    if dirty:
        print("refusing to run: uncommitted changes in the files this edits, so 'restore' is")
        print("not well defined. Commit or stash first:")
        for d in dirty:
            print("   ", d)
        return 2

    print("baseline (unmutated) suite ...", flush=True)
    rc, out = run_suite(args.timeout, args.tests)
    if rc != 0:
        print("refusing to run: the suite is not green before mutation.")
        print(out[-2000:])
        return 2
    print("  green\n")

    results = []
    for label, cls, rel, old, new in muts:
        path = os.path.join(ROOT, rel)
        # BINARY. Text-mode round-tripping rewrites CRLF as LF, which git reports as a modified
        # file even though nothing changed -- a false "not restored cleanly" warning, which is
        # worse than none because it teaches you to ignore the real one.
        with io.open(path, "rb") as fh:
            original = fh.read()
        old_b, new_b = old.encode(), new.encode()
        if old_b not in original:
            print("%-28s %-6s SKIP  anchor text not found (code moved?)" % (label, cls))
            results.append({"label": label, "bug_class": cls, "file": rel,
                            "outcome": "SKIPPED_ANCHOR_MISSING"})
            continue
        if original.count(old_b) != 1:
            print("%-28s %-6s SKIP  anchor is not unique (%d matches)"
                  % (label, cls, original.count(old_b)))
            results.append({"label": label, "bug_class": cls, "file": rel,
                            "outcome": "SKIPPED_ANCHOR_AMBIGUOUS"})
            continue
        t0 = time.time()
        try:
            with io.open(path, "wb") as fh:
                fh.write(original.replace(old_b, new_b, 1))
            rc, out = run_suite(args.timeout, args.tests)
        finally:
            with io.open(path, "wb") as fh:
                fh.write(original)
        caught = rc != 0
        dt = time.time() - t0
        print("%-28s %-6s %s   (%.0fs)"
              % (label, cls, "caught" if caught else "*** SURVIVED ***", dt), flush=True)
        results.append({"label": label, "bug_class": cls, "file": rel,
                        "outcome": "CAUGHT" if caught else "SURVIVED",
                        "returncode": rc, "seconds": round(dt, 1),
                        "tail": out.strip().splitlines()[-1] if out.strip() else ""})

    # a final restore check: the tree must be exactly as we found it
    left = git_dirty(files)
    tested = [r for r in results if r["outcome"] in ("CAUGHT", "SURVIVED")]
    survived = [r for r in tested if r["outcome"] == "SURVIVED"]
    print("")
    print("%d mutation(s) run: %d caught, %d SURVIVED"
          % (len(tested), len(tested) - len(survived), len(survived)))
    if survived:
        print("survivors -- the suite does not constrain these, so a real bug of this shape "
              "would pass CI:")
        for r in survived:
            print("   %-28s %-6s %s" % (r["label"], r["bug_class"], r["file"]))
    if left:
        print("")
        print("WARNING: files not restored cleanly -- inspect before committing:")
        for d in left:
            print("   ", d)

    outp = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with io.open(outp, "w", encoding="utf-8") as fh:
        json.dump({"tests": args.tests or TESTS, "results": results,
                   "n_caught": len(tested) - len(survived), "n_survived": len(survived),
                   "restored_clean": not left}, fh, indent=2)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
