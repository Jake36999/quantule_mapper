"""
ONE PRE-REGISTERED passive test (no fishing) — chiral-pair BALANCE, re-derived from the framework's
"quantule chiral pairs" (NOT the failed co-handed alignment metric).

DERIVATION (docs/PAYAN_DERIVATION_PASSIVE_DIAGNOSTIC.md / Provenance v9 "quantule chiral pairs"):
a genuine chiral pair is a BALANCED opposite-spin pair — the two bridge nodes carry axial Payan spin
of OPPOSITE sign and MATCHED magnitude, so the net topological charge CANCELS across the corridor.
The predicted failure mode is charge leak / imbalance: uncancelled net spin |s_i+s_j| = tension with
no balanced channel to vent -> instability.

PRE-REGISTERED (fixed BEFORE running):
  observable  : balance B = 1 - |s_i + s_j| / (|s_i| + |s_j| + eps)  in [0,1]
                (B=1 perfectly balanced opposite pair s_i=-s_j; B=0 aligned or one-sided/charge-leaked)
  hypothesis  : STABLE bridges have HIGHER balance B than UNSTABLE (balanced pairs vent tension).
  PASS (all): (1) mean(B_stable) - mean(B_unstable) >= 0.15
              (2) AUC(B: stable>unstable) >= 0.65
              (3) phase-randomized control gap |dB_ctrl| < 0.10  (control does NOT reproduce it)
              (4) |dB| > 0.08  (beats the FAILED alignment metric's gap of 0.08)
  verdict     : BALANCE_PREDICTS_STABILITY  (all hold)  else  BALANCE_NO_SIGNAL.
ONE test only. No solver coupling. WSL2 jax venv:
  python /mnt/f/quantule_mapper/jax_scout/afield_payan_balance_test.py --n 25
"""
import os, sys, json, glob, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout._legacy.afield_payan_diagnostic import select_population, settle, payan_observables, phase_randomized

# pre-registered thresholds
DB_MIN, AUC_MIN, CTRL_MAX, BEAT_ALIGN = 0.15, 0.65, 0.10, 0.08


def balance(ob):
    si, sj = ob["s_i"], ob["s_j"]
    return 1.0 - abs(si + sj) / (abs(si) + abs(sj) + 1e-30)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=25); args = ap.parse_args()
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "SUBSTRATE_HUNT_2026*")))[-1]
    stable, unstable = select_population(d, args.n)
    rng = np.random.default_rng(20260619)
    print(f"=== ONE PRE-REGISTERED TEST: chiral-pair BALANCE B (no coupling) — "
          f"{len(stable)} stable vs {len(unstable)} unstable ===")
    print(f"PASS if dB>={DB_MIN}, AUC>={AUC_MIN}, |dB_ctrl|<{CTRL_MAX}, |dB|>{BEAT_ALIGN}\n")
    rec = {"stable": [], "unstable": []}
    for klass, pop in (("stable", stable), ("unstable", unstable)):
        for par, g, kap, cA, h, br in pop:
            t0 = time.time(); psi = settle(par, g, kap, cA, 48)
            if psi is None:
                print(f"  [{klass} {h}] settle_nonfinite ({time.time()-t0:.0f}s)"); continue
            ob = payan_observables(psi, par, 48)
            if ob is None:
                print(f"  [{klass} {h}] <2 nodes ({time.time()-t0:.0f}s)"); continue
            ctrl = payan_observables(phase_randomized(psi, rng), par, 48)
            B, Bc = balance(ob), (balance(ctrl) if ctrl else None)
            rec[klass].append({"hash": h, "B": B, "B_ctrl": Bc, "s_i": ob["s_i"], "s_j": ob["s_j"]})
            print(f"  [{klass} {h}] B={B:.3f} (ctrl {Bc if Bc is None else round(Bc,3)}) "
                  f"s=({ob['s_i']:.0f},{ob['s_j']:.0f})  ({time.time()-t0:.0f}s)", flush=True)
    def col(k, key): return [r[key] for r in rec[k] if r.get(key) is not None]
    Bs, Bu = col("stable", "B"), col("unstable", "B")
    Bsc, Buc = col("stable", "B_ctrl"), col("unstable", "B_ctrl")
    dB = float(np.mean(Bs) - np.mean(Bu)); dB_ctrl = float(np.mean(Bsc) - np.mean(Buc))
    A = auc(Bs, Bu)
    print(f"\n--- chiral-pair balance B (1=balanced opposite pair, 0=aligned/charge-leaked) ---")
    print(f"  mean B: stable={np.mean(Bs):.3f}  unstable={np.mean(Bu):.3f}  dB={dB:+.3f}")
    print(f"  AUC(B: stable>unstable)={A:.3f} | control dB={dB_ctrl:+.3f}")
    c1, c2, c3, c4 = (dB >= DB_MIN), (A >= AUC_MIN), (abs(dB_ctrl) < CTRL_MAX), (abs(dB) > BEAT_ALIGN)
    passed = c1 and c2 and c3 and c4
    verdict = "BALANCE_PREDICTS_STABILITY" if passed else "BALANCE_NO_SIGNAL"
    print(f"  criteria: dB>={DB_MIN}:{c1}  AUC>={AUC_MIN}:{c2}  |dB_ctrl|<{CTRL_MAX}:{c3}  |dB|>{BEAT_ALIGN}:{c4}")
    out = {"n_stable": len(Bs), "n_unstable": len(Bu), "mean_B_stable": float(np.mean(Bs)),
           "mean_B_unstable": float(np.mean(Bu)), "dB": dB, "auc": A, "dB_ctrl": dB_ctrl,
           "criteria": {"dB": c1, "auc": c2, "ctrl": c3, "beats_alignment": c4},
           "verdict": verdict, "records": rec}
    od = os.path.join(d, "payan_balance_test.json")
    json.dump(out, open(od, "w"), indent=2, default=float)
    print(f"\nVERDICT: {verdict}")
    print("  -> chiral-pair balance gates stability: revisit Payan coupling RFC with THIS observable"
          if passed else
          "  -> chiral-pair balance does NOT gate stability either; Payan coupling remains NOT justified")
    print(f"wrote {od}\nNOTE: passive, pre-registered single test; NOT proof. Coupling unbuilt.")


if __name__ == "__main__":
    main()
