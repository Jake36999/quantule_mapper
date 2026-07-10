"""Phase D / C3 — higher-speed collision ladder for the VK-stable Q-ball (bounded, C3 only).
Question: do the C3 Q-balls stay capture-dominated at higher speed, or is there a critical velocity where
pass-through / transmission appears? Same validated branch (a=0.8,s=-0.5,f=-0.1,c^2=0.3,w=0.964), same separation
(sep=10), same in-phase symmetric head-on. Speeds 0.15c/0.30c already CAPTURE (docs/PHASE_D_C3_TWOQBALL_RESULTS.md);
this adds 0.45c, 0.60c, (0.75c optional).

Telemetry: total energy + U(1) charge conservation (Strang error meter), |psi|^2 mass/density retention, windowed
centroid separation vs time, sep_min + post-collision re-separation, radiation-leakage estimate (mass outside the two
core windows), identity-ambiguity flag (symmetric collision: pass-through vs rebound is undefinable), and the final
classification CAPTURE / PASS_THROUGH / DISRUPT / INCONCLUSIVE. C3 higher-speed collision RESULTS only; not a general
theory claim. Bigger box (room to re-separate w/o wrap) + finer dt (violent-overlap Strang error). Standalone.

  wsl:  python jax_scout/phase_d_c3_collision_ladder.py [--vfracs 0.45,0.60,0.75 --out DIR]
"""
import os, sys, json, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.phase_d_c3_wave import build_kg, kg_evolve, qball_petviashvili, invariants
from jax_scout.phase_d_c3_two_qball import boosted_qball, _mi, W_WIN

SEP_CLEAN = 6.0        # cores clearly re-separated beyond this (> 2*W_WIN merge boundary)


def core_com(rx, x, xc, L, w=W_WIN):
    m = np.abs(_mi(x, xc, L)) < w
    M = float(rx[m].sum())
    if M < 1e-12:
        return xc, 0.0
    th = 2 * np.pi * x[m] / L
    C = float((rx[m] * np.cos(th)).sum()); S = float((rx[m] * np.sin(th)).sum())
    return L * np.arctan2(S, C) / (2 * np.pi), M


def collide(phi, w, c, vfrac, op, a, s, f, out, dphi=0.0):
    N, L, dt = op["N"], op["L"], op["dt"]
    v = vfrac * c
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    psiL, piL = boosted_qball(phi, w, c, +v, -5.0, L, op)              # left, moving right
    psiR, piR = boosted_qball(phi, w, c, -v, +5.0, L, op, dphi=dphi)   # right, moving left (relative phase dphi)
    psi = (psiL + psiR).astype(np.complex128); pi = (piL + piR).astype(np.complex128)
    amp_single = float(np.abs(phi).max())
    pk = jnp.fft.fftn(jnp.asarray(psi)); qk = jnp.fft.fftn(jnp.asarray(pi))
    inv0 = invariants(pk, qk, op, a, s, f); M0 = inv0["mass"]
    Tp = min(10.0 / (2 * v) + 22.0, 60.0)                             # room for slow collisions to fully re-separate
    steps = int(round(Tp / dt)); chunk = 250
    traj = []; xcL, xcR = -5.0, 5.0; dE_max = dQ_max = 0.0
    for cix in range(steps // chunk):
        pk, qk = kg_evolve(pk, qk, op, a, s, f, chunk)
        cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all():
            return {"vfrac": vfrac, "outcome": "DISRUPT", "reason": "non-finite", "traj": traj}
        rx = (np.abs(cur) ** 2).sum(axis=(1, 2))
        xcL, mL = core_com(rx, x, xcL, L); xcR, mR = core_com(rx, x, xcR, L)
        sep = abs(_mi(xcR, xcL, L))
        core_mask = (np.abs(_mi(x, xcL, L)) < W_WIN) | (np.abs(_mi(x, xcR, L)) < W_WIN)
        rad = 1.0 - float(rx[core_mask].sum()) / (float(rx.sum()) + 1e-30)
        inv = invariants(pk, qk, op, a, s, f)
        dEr = abs(inv["E"] - inv0["E"]) / (abs(inv0["E"]) + 1e-30)
        dQr = abs(inv["Q"] - inv0["Q"]) / (abs(inv0["Q"]) + 1e-30)
        dE_max = max(dE_max, dEr); dQ_max = max(dQ_max, dQr)
        traj.append({"t": round((cix + 1) * chunk * dt, 3), "sep": round(sep, 3), "xcL": round(xcL, 2),
                     "xcR": round(xcR, 2), "mass_ret": round(inv["mass"] / M0, 4), "amp": round(inv["amp"], 3),
                     "rad_frac": round(rad, 4), "dE_rel": dEr, "dQ_rel": dQr})
    np.savez_compressed(os.path.join(out, f"collide_v{vfrac:.2f}_p{dphi:.2f}.npz"),
                        t=np.array([q["t"] for q in traj]), sep=np.array([q["sep"] for q in traj]),
                        amp=np.array([q["amp"] for q in traj]), rad=np.array([q["rad_frac"] for q in traj]))
    return _classify(vfrac, v, c, traj, dE_max, dQ_max, amp_single, dphi)


def _classify(vfrac, v, c, traj, dE_max, dQ_max, amp_single, dphi=0.0):
    seps = np.array([q["sep"] for q in traj]); ts = np.array([q["t"] for q in traj])
    sep0 = float(seps[0]); i_min = int(np.argmin(seps)); sep_min = float(seps[i_min])
    sep_end = float(seps[-1]); amp_end = float(traj[-1]["amp"]); rad_end = float(traj[-1]["rad_frac"])
    mass_end = float(traj[-1]["mass_ret"])
    # outgoing relative speed from the post-min separation slope (clean-separated frames only)
    post = [(ts[i], seps[i]) for i in range(i_min + 1, len(seps)) if seps[i] > SEP_CLEAN]
    vout_rel = np.nan
    if len(post) >= 3:
        tp = np.array([p[0] for p in post]); sp = np.array([p[1] for p in post])
        vout_rel = float(np.polyfit(tp, sp, 1)[0])                     # d(sep)/dt of the separating pair
    approached = sep_min < sep0 - 1.5                                  # cores meaningfully closed in
    overlapped = sep_min < 2 * W_WIN                                   # cores actually reached full overlap
    re_separated = overlapped and sep_end > SEP_CLEAN                  # overlapped then re-separated (transmit/bounce)
    merged = overlapped and sep_end < 2 * W_WIN                        # overlapped and stayed bound
    bounced = approached and not overlapped                           # repelled BEFORE overlapping (repulsive channel)
    coherent = amp_end > 0.55 * amp_single and rad_end < 0.5
    numerically_ok = dE_max < 5e-2 and dQ_max < 5e-2
    if not numerically_ok:
        outcome = "INCONCLUSIVE"; reason = f"E/Q drift too high (dE={dE_max:.1e} dQ={dQ_max:.1e}) -> reduce dt"
    elif not approached:
        outcome = "INCONCLUSIVE"; reason = f"cores did not meaningfully approach (sep_min={sep_min:.2f} vs sep0={sep0:.2f})"
    elif bounced:
        rev = "re-separating" if sep_end > sep_min + 0.3 else "at/near turning point (window may be short)"
        outcome = "BOUNCE"; reason = (f"repelled before overlap (sep_min={sep_min:.2f}, {rev}, sep_end={sep_end:.2f}, "
                                      f"rad={rad_end:.2f})")
    elif merged:
        outcome = "CAPTURE"; reason = f"overlapped & stayed bound (sep_min={sep_min:.2f}, sep_end={sep_end:.2f}, rad={rad_end:.2f})"
    elif re_separated and coherent:
        outcome = "PASS_THROUGH"; reason = (f"two coherent cores overlapped then re-separated (sep_min={sep_min:.2f}->end "
                                            f"{sep_end:.2f}, vout_rel={vout_rel:+.3f} vs 2v_in={2*v:.3f}, rad={rad_end:.2f})")
    elif re_separated and not coherent:
        outcome = "DISRUPT"; reason = f"cores separated but incoherent (amp {amp_end:.2f}/{amp_single:.2f}, rad {rad_end:.2f})"
    else:
        outcome = "INCONCLUSIVE"; reason = f"intermediate (sep_min={sep_min:.2f} sep_end={sep_end:.2f} rad={rad_end:.2f})"
    return {"vfrac": vfrac, "v": v, "v_over_c": vfrac, "dphi": dphi, "outcome": outcome, "reason": reason,
            "sep_min": sep_min, "sep_end": sep_end, "vout_rel": vout_rel, "v_in_closing": 2 * v,
            "elasticity_est": float(abs(vout_rel) / (2 * v)) if np.isfinite(vout_rel) else np.nan,
            "amp_end": amp_end, "amp_single": amp_single, "rad_frac_end": rad_end, "mass_ret_end": mass_end,
            "dE_rel_max": dE_max, "dQ_rel_max": dQ_max, "identity_ambiguous": True, "traj_tail": traj[-4:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=80); ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--c", type=float, default=0.5477); ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--a", type=float, default=0.8); ap.add_argument("--s", type=float, default=-0.5)
    ap.add_argument("--f", type=float, default=-0.1); ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--wfrac", type=float, default=0.964); ap.add_argument("--vfracs", default="0.45,0.60,0.75")
    ap.add_argument("--dphi", type=float, default=0.0, help="relative phase between the two Q-balls (pi = anti-phase / repulsive channel)")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    out = A.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C3_COLLISION_LADDER_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    op = build_kg(A.N, A.L, A.c, A.m, A.dt)
    a, s, f = A.a, A.s, A.f
    w = A.wfrac * A.m; mu = A.m ** 2 - w ** 2
    chan = "IN-PHASE (attractive)" if abs(A.dphi) < 1e-6 else f"dphi={A.dphi:.3f} ({'ANTI-PHASE/repulsive' if abs(A.dphi-np.pi)<0.3 else 'off-phase'})"
    print(f"=== C3 COLLISION LADDER | N={A.N} L={A.L} c={A.c} dt={A.dt} w={w:.3f} | {chan} head-on, sep=10 "
          f"| out={out} ===", flush=True)
    # Q-ball in this box + single-object hold sanity
    phi = None
    for sg in (1.5, 1.2, 1.8, 2.0):
        ph, pr = qball_petviashvili(op, a, s, f, mu, sig=sg)
        if ph is not None and pr["residual"] < 1e-6 and pr["occ"] < 0.5:
            phi = ph; prof = pr; break
    if phi is None:
        print(f"FAIL: no Q-ball in L={A.L} box", flush=True); return
    print(f"[T0] Q-ball: residual={prof['residual']:.2e} amp={prof['amp']:.3f} occ={prof['occ']:.4f}", flush=True)
    res = {"config": vars(A), "w": w, "dphi": A.dphi, "qball": prof, "ladder": []}
    for vf in [float(x) for x in A.vfracs.split(",")]:
        t0 = time.time()
        r = collide(phi, w, A.c, vf, op, a, s, f, out, dphi=A.dphi)
        res["ladder"].append(r)
        print(f"[v={vf:.2f}c] {r['outcome']}: {r['reason']} | mass_ret={r.get('mass_ret_end')} "
              f"rad={r.get('rad_frac_end')} dE_max={r.get('dE_rel_max'):.1e} dQ_max={r.get('dQ_rel_max'):.1e} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
        json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    ladder = [(f"{r['vfrac']:.2f}c", r["outcome"]) for r in res["ladder"]]
    outs = {r["outcome"] for r in res["ladder"]}
    vmax = f"{res['ladder'][-1]['vfrac']:.2f}" if res["ladder"] else "NA"
    chan_tag = "INPHASE" if abs(A.dphi) < 1e-6 else ("ANTIPHASE" if abs(A.dphi - np.pi) < 0.3 else f"DPHI{A.dphi:.2f}")
    if "PASS_THROUGH" in outs:
        verdict = f"C3_{chan_tag}_TRANSMISSION_FOUND"
    elif outs <= {"BOUNCE"}:
        verdict = f"C3_{chan_tag}_BOUNCE_DOMINATED_UP_TO_{vmax}c"
    elif outs == {"CAPTURE"}:
        verdict = f"C3_{chan_tag}_CAPTURE_DOMINATED_UP_TO_{vmax}c"
    else:
        verdict = f"C3_{chan_tag}_MIXED_{'_'.join(sorted(outs))}"
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} | dphi={A.dphi:.3f} | ladder {ladder} ===", flush=True)
    print(f"C3_COLLISION_LADDER_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
