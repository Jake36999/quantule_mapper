"""Phase D / C2.10 — RUN-2: NLS (C2) ANTI-PHASE COLLISION LADDER.

Question (cross-substrate universality of COLLISIONS): the C3 (KG) collision phase diagram found transmission ONLY
at exact anti-phase (dphi=pi) and low/moderate speed — a destructive-interference NODE feature; in-phase and
off-phase capture generically. Does the FIRST-ORDER conservative NLS substrate (C2) show the SAME anti-phase
node-protected transmission channel, or does it capture at all speeds (no NLS transmission window)?

Method: the validated GALILEAN soliton family (a=0.8,s=-0.5,f=-0.1,D=0.3,mu=0.070; phi_iso from C2.7/C2.8),
symmetric head-on at sep=10 in the L=20/N=96 validated grid (dx/CFL matched). Each soliton boosted phi*exp(+-ikx),
v=2Dk; RIGHT soliton carries a relative phase exp(i*dphi). Sweep boost integer n (speed) at a fixed dphi
(pi = anti-phase channel; 0 = in-phase control).

Telemetry: total mass (norm) + momentum P conservation meters (NLS invariants); windowed-COM separation vs time;
radiation fraction (mass outside the two core windows); peak amplitude (coherence). Classifier PORTED from the C3
ladder (approached/overlapped/re_separated/merged/bounced + coherence): CAPTURE / PASS_THROUGH / BOUNCE / DISRUPT /
INCONCLUSIVE. Mirror-only, true-flat substrate (param_geom_off), no production/Hunter/config changes, no matter claim.

  wsl:  ~/jax_irer/bin/python jax_scout/phase_d_c2_10_antiphase_collision.py \
            --phi-iso sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy \
            --dphi 3.14159 --nvals 1,2,4 [--out DIR]
"""
import os, sys, json, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics
from jax_scout.phase_d_c1_transport import _evolve_chunk
from jax_scout.phase_d_c2_3_exact_soliton import momentum_x
from jax_scout.phase_d_c2_8_two_node import build, axis, place, FAM, petviashvili_L, isolate_L
from jax_scout.phase_d_c2_9_two_node_robust import profiles, core_window, _mi, W_WIN

SEP_CLEAN = 2 * W_WIN          # cores clearly re-separated beyond this (well outside the merge scale)
# The MERGE scale is set by the soliton core (ell ~ 2.07), NOT the tracking window W_WIN=3.5. Two cores are
# genuinely overlapped/merged only when sep < ~1.7*ell; using 2*W_WIN=7 as the merge threshold spuriously labels a
# clean bounce (min-sep ~5, ~1% radiation) as CAPTURE. MERGE_SEP is the physical overlap threshold.
MERGE_SEP = 3.5


def _anchor_two(rho_x, x, L):
    idx = np.argsort(rho_x)[::-1]; p1 = int(idx[0]); p2 = None; N = len(rho_x)
    for j in idx[1:]:
        d = abs(int(j) - p1); d = min(d, N - d)
        if d > 6:
            p2 = int(j); break
    if p2 is None:
        return float(x[p1]), None
    a, b = sorted([float(x[p1]), float(x[p2])])
    return a, b


def collide(phi_iso, ops, N, L, dt, x, Xax, n, dphi, amp_single, out):
    D = FAM["D"]
    k = 2 * np.pi * n / L
    v = 2 * D * k
    psiL = place(phi_iso, -5.0, N, L) * np.exp(1j * k * (Xax + 5.0))
    psiR = np.exp(1j * dphi) * place(phi_iso, +5.0, N, L) * np.exp(-1j * k * (Xax - 5.0))
    psi = (psiL + psiR).astype(np.complex128)
    M0 = float(np.sum(np.abs(psi) ** 2)); P0 = momentum_x(psi, ops.ikx)
    Tp = min(10.0 / (2 * v) + 22.0, 60.0)
    steps = int(round(Tp / dt)); chunk = 250
    pk = physics.initial_psi_k(jnp.asarray(psi), ops)
    traj = []; xcL, xcR = -5.0, 5.0; dmass_max = 0.0
    for cix in range(steps // chunk):
        pk = _evolve_chunk(pk, ops, chunk); cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return {"n": n, "dphi": dphi, "outcome": "DISRUPT", "reason": "non-finite", "traj": traj}
        rho_x, px_x = profiles(cur, ops)
        ML, vL, xcL_n, aL = core_window(rho_x, px_x, x, xcL, L, D)
        MR, vR, xcR_n, aR = core_window(rho_x, px_x, x, xcR, L, D)
        sep = abs(_mi(xcR_n, xcL_n, L))
        overlap = sep < 2 * W_WIN
        if not overlap:
            a, b = _anchor_two(rho_x, x, L)
            xcL, xcR = (a, b) if b is not None else (xcL_n, xcR_n)
        else:
            xcL, xcR = xcL_n, xcR_n
        core_mask = (np.abs(_mi(x, xcL, L)) < W_WIN) | (np.abs(_mi(x, xcR, L)) < W_WIN)
        rad = 1.0 - float(rho_x[core_mask].sum()) / (float(rho_x.sum()) + 1e-30)
        Mtot = float((np.abs(cur) ** 2).sum())
        dmass_max = max(dmass_max, abs(Mtot - M0) / (M0 + 1e-30))
        traj.append({"t": round((cix + 1) * chunk * dt, 3), "sep": round(sep, 3),
                     "mass_ret": round(Mtot / M0, 4), "amp": round(float(np.abs(cur).max()), 3),
                     "rad_frac": round(rad, 4), "P": round(momentum_x(cur, ops.ikx), 1)})
    np.savez_compressed(os.path.join(out, f"collide_n{n:.2f}_p{dphi:.2f}.npz"),
                        t=np.array([q["t"] for q in traj]), sep=np.array([q["sep"] for q in traj]),
                        amp=np.array([q["amp"] for q in traj]), rad=np.array([q["rad_frac"] for q in traj]),
                        mass=np.array([q["mass_ret"] for q in traj]))
    return _classify(n, v, traj, dmass_max, amp_single, dphi, P0)


def _classify(n, v, traj, dmass_max, amp_single, dphi, P0):
    seps = np.array([q["sep"] for q in traj]); ts = np.array([q["t"] for q in traj])
    sep0 = float(seps[0]); i_min = int(np.argmin(seps)); sep_min = float(seps[i_min])
    sep_end = float(seps[-1]); amp_end = float(traj[-1]["amp"]); rad_end = float(traj[-1]["rad_frac"])
    mass_end = float(traj[-1]["mass_ret"])
    # Post-min PEAK separation is the robust re-separation signal: in a periodic box the pair can pass through,
    # separate fully, then wrap back and approach again, so sep_end understates how far they re-separated. Coherence
    # is judged at that re-emergence peak, not at the (possibly wrapped/second-approach) final frame.
    i_peak = i_min + int(np.argmax(seps[i_min:])); sep_peak = float(seps[i_peak])
    rad_peak = float(traj[i_peak]["rad_frac"]); amp_peak = float(traj[i_peak]["amp"])
    seg = [(ts[i], seps[i]) for i in range(i_min + 1, i_peak + 1)]
    vout_rel = float(np.polyfit([p[0] for p in seg], [p[1] for p in seg], 1)[0]) if len(seg) >= 3 else np.nan
    approached = sep_min < sep0 - 1.5
    overlapped = sep_min < MERGE_SEP           # genuine core overlap (soliton-width scale), not the tracking window
    re_separated = overlapped and sep_peak > SEP_CLEAN
    merged = overlapped and sep_peak < MERGE_SEP
    bounced = approached and not overlapped
    coherent = amp_peak > 0.55 * amp_single and rad_peak < 0.5 and mass_end > 0.7
    if not np.isfinite(sep_end):
        outcome, reason = "DISRUPT", "non-finite separation"
    elif not approached:
        outcome, reason = "INCONCLUSIVE", f"cores did not approach (sep_min={sep_min:.2f} vs sep0={sep0:.2f})"
    elif bounced:
        rev = "re-separating" if sep_end > sep_min + 0.3 else "near turning point (window may be short)"
        outcome, reason = "BOUNCE", f"repelled before overlap (sep_min={sep_min:.2f}, {rev}, sep_end={sep_end:.2f}, rad={rad_end:.2f})"
    elif merged:
        outcome, reason = "CAPTURE", f"overlapped & stayed bound (sep_min={sep_min:.2f}, sep_end={sep_end:.2f}, rad={rad_end:.2f})"
    elif re_separated and coherent:
        outcome, reason = "PASS_THROUGH", (f"two coherent cores overlapped then re-separated (sep_min={sep_min:.2f}->end "
                                           f"{sep_end:.2f}, vout_rel={vout_rel:+.3f} vs 2v_in={2*v:.3f}, rad={rad_end:.2f}, mass={mass_end:.3f})")
    elif re_separated and not coherent:
        outcome, reason = "DISRUPT", f"separated but incoherent (amp {amp_end:.2f}/{amp_single:.2f}, rad {rad_end:.2f}, mass {mass_end:.3f})"
    else:
        outcome, reason = "INCONCLUSIVE", f"intermediate (sep_min={sep_min:.2f} sep_end={sep_end:.2f} rad={rad_end:.2f})"
    return {"n": n, "v": v, "v_each": v, "closing": 2 * v, "dphi": dphi, "outcome": outcome, "reason": reason,
            "sep_min": sep_min, "sep_end": sep_end, "vout_rel": vout_rel, "closing_in": 2 * v,
            "elasticity_est": float(abs(vout_rel) / (2 * v)) if np.isfinite(vout_rel) else np.nan,
            "amp_end": amp_end, "amp_single": amp_single, "rad_frac_end": rad_end, "mass_ret_end": mass_end,
            "dmass_rel_max": dmass_max, "P0": P0, "P_end": traj[-1]["P"], "identity_ambiguous": True,
            "traj_tail": traj[-4:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default=None, help="reuse an isolated soliton profile; else generate via Petviashvili")
    ap.add_argument("--N", type=int, default=96); ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--dphi", type=float, default=np.pi, help="relative phase (pi = anti-phase channel; 0 = in-phase control)")
    ap.add_argument("--nvals", default="1,2,4", help="boost integers (speed ladder); v_each = 2D*2pi*n/L")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    N, L, dt = A.N, A.L, A.dt
    tag = "ANTIPHASE" if abs(A.dphi - np.pi) < 0.3 else ("INPHASE" if abs(A.dphi) < 1e-6 else f"DPHI{A.dphi:.2f}")
    out = A.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C2_10_{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    ops = build(N, L, dt); x, Xax = axis(N, L)
    print(f"=== C2.10 NLS COLLISION LADDER | {FAM} | N={N} L={L} dt={dt} | dphi={A.dphi:.3f} ({tag}) | out={out} ===", flush=True)

    # soliton profile
    if A.phi_iso and os.path.exists(os.path.join(ROOT, A.phi_iso) if not os.path.isabs(A.phi_iso) else A.phi_iso):
        p = A.phi_iso if os.path.isabs(A.phi_iso) else os.path.join(ROOT, A.phi_iso)
        phi_iso = np.load(p); assert phi_iso.shape == (N, N, N)
        print(f"[T0] reusing phi_iso: {p}", flush=True)
    else:
        phi, prof = petviashvili_L(1.0, 1.2, ops, N, L, FAM["mu"])
        if phi is None or prof.get("residual", 1) > 1e-6:
            print(f"T0 FAIL: {prof}", flush=True); return
        phi_iso = isolate_L(phi, N, L)
        print(f"[T0] generated phi_iso: residual={prof['residual']:.2e} amp={prof['amp']:.3f}", flush=True)
    amp_single = float(np.abs(phi_iso).max())

    # hold gate
    from jax_scout.phase_d_c2_8_two_node import evolve_pair
    r0 = evolve_pair(phi_iso, ops, N, L, dt, 2.0, 1000, "t0_hold", out)
    hold = r0["traj"][-1]["mass"]
    print(f"[T0] hold T=2: mass_ret={hold:.4f} amp_single={amp_single:.3f}", flush=True)
    if hold < 0.98:
        print("T0 GATE FAIL (soliton does not hold)", flush=True); return

    nlist = [float(s) for s in A.nvals.split(",")]

    # V0 — boost validation (essential for NON-INTEGER n, whose exp(ikx) leaves a phase mismatch at the box edge;
    # harmless only if it is in the deep vacuum). Single soliton at the first ladder speed must read v = 2Dk with
    # mass conserved; a seeded-radiation artifact would show up as v-error or mass loss here.
    n0 = nlist[0]; k0 = 2 * np.pi * n0 / L
    single = (place(phi_iso, 0.0, N, L) * np.exp(1j * k0 * Xax)).astype(np.complex128)
    M0s = float(np.sum(np.abs(single) ** 2)); pk = physics.initial_psi_k(jnp.asarray(single), ops)
    xc = 0.0; vs = []
    for _ in range(8):
        pk = _evolve_chunk(pk, ops, 125); cur = np.asarray(jnp.fft.ifftn(pk))
        rho_x, px_x = profiles(cur, ops); _, vv, xc, _ = core_window(rho_x, px_x, x, xc, L, FAM["D"])
        if np.isfinite(vv):
            vs.append(vv)
    vmeas = float(np.nanmean(vs)); vpred = 2 * FAM["D"] * k0
    massv = float((np.abs(cur) ** 2).sum()) / M0s
    verr = abs(vmeas - vpred) / (abs(vpred) + 1e-30) * 100
    print(f"[V0] boost n={n0:.2f} (k={k0:.4f}): v_meas={vmeas:+.4f} vs 2Dk={vpred:+.4f} (err {verr:.2f}%) mass={massv:.4f}", flush=True)
    boost_ok = verr < 3.0 and massv > 0.98
    res = {"config": vars(A), "family": FAM, "dphi": A.dphi, "amp_single": amp_single,
           "boost_validation": {"n": n0, "v_meas": vmeas, "v_pred": vpred, "err_pct": verr, "mass": massv, "ok": boost_ok},
           "ladder": []}
    if not boost_ok:
        print(f"[V0] BOOST VALIDATION FAILED (err {verr:.1f}% mass {massv:.3f}) — non-integer boost may be seeding "
              f"artifacts; ladder results suspect.", flush=True)
    for n in nlist:
        t0 = time.time()
        r = collide(phi_iso, ops, N, L, dt, x, Xax, n, A.dphi, amp_single, out)
        res["ladder"].append(r)
        print(f"[n={n} v={r['v']:.3f} closing={r['closing']:.3f}] {r['outcome']}: {r['reason']} | "
              f"mass={r.get('mass_ret_end')} rad={r.get('rad_frac_end')} dmass={r.get('dmass_rel_max'):.1e} "
              f"P {r.get('P0'):.1f}->{r.get('P_end')} ({(time.time()-t0)/60:.1f}m)", flush=True)
        json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)

    outs = {r["outcome"] for r in res["ladder"]}
    nmax = res["ladder"][-1]["n"] if res["ladder"] else "NA"
    if "PASS_THROUGH" in outs:
        verdict = f"C2_{tag}_TRANSMISSION_FOUND"
    elif outs <= {"BOUNCE"}:
        verdict = f"C2_{tag}_BOUNCE_DOMINATED"
    elif outs == {"CAPTURE"}:
        verdict = f"C2_{tag}_CAPTURE_DOMINATED"
    else:
        verdict = f"C2_{tag}_MIXED_{'_'.join(sorted(outs))}"
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    ladder = [(r["n"], f"{r['v']:.2f}", r["outcome"]) for r in res["ladder"]]
    print(f"\n=== {verdict} | dphi={A.dphi:.3f} | ladder(n,v,outcome) {ladder} ===", flush=True)
    print(f"C2_10_COLLISION_LADDER_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
