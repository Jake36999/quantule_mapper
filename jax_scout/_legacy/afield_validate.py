"""
Validation tier for the A-coupled hunt — the decisive web->wires test.

The hunt found (a) 6 marginal A_PERSISTENT configs (phase coupling barely above the 0.73 floor but
BRIDGE-LESS) and (b) strong-bridge + strong-A-localization configs (A_loc up to 12) with
sub-floor phase coupling. The two properties did not co-occur. The remaining decisive question,
independent of the marginal phase-coupling, is whether the strong current-coupled A-localization
actually shifts the coupling STRUCTURE from global web -> pairwise wires:
    does global_mode_fraction DROP under A-on vs gamma_A=0, with rising node_bridge selectivity?

This runs global_mode_under_A (response matrix under A-on) on the best WIRE-CANDIDATES (strong
A_loc + bridge + decent pcoup, regardless of class), each vs its gamma_A=0 control.

CAUTION: JAX scout. Not proof. CuPy has no current-coupled term -> no promotion. A drop in
global_mode_fraction is mechanistic evidence (web->wires) at scout level only.
WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/afield_validate.py
"""
import os, sys, json, glob, csv, time
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics
from jax_scout._legacy.afield_current_tune import global_mode_under_A
base = physics.SWEEP_PARAM_ORDER


def F(r, k):
    try: return float(r[k])
    except: return float("nan")


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "AF_BRIDGE_HUNT_2026*")))[-1]
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    acc = [r for r in rows if r["reject"] == "" and F(r, "pcoup") > 0.45 and F(r, "bridge") > 0.10
           and F(r, "A_loc") > 3 and 0.5 <= F(r, "er") <= 2.0]
    # rank wire-candidates: bridge + pcoup + A localization together
    acc.sort(key=lambda r: -(F(r, "bridge") + F(r, "pcoup") + 0.1*F(r, "A_loc")))
    top = acc[:3]
    print(f"A-coupled validation: web->wires (global_mode_fraction A-on vs gamma_A=0) on {len(top)} wire-candidates\n")
    report = []
    for r in top:
        par = {k: F(r, k) for k in base}
        g, kap, cA = F(r, "gamma_A"), F(r, "kappa"), F(r, "c_A")
        lbl = f"gen{r['gen']}_gA{g:.2f}_br{F(r,'bridge'):.2f}_Aloc{F(r,'A_loc'):.1f}"
        print(f"[{lbl}] pcoup={F(r,'pcoup'):.3f} er={F(r,'er'):.2f} nodes={int(F(r,'nodes'))} "
              f"bridge={F(r,'bridge'):.3f} (kap={kap:.2f} cA={cA:.2f})")
        t0 = time.time()
        gm0 = global_mode_under_A(par, 0.0, kap, cA)
        gmA = global_mode_under_A(par, g, kap, cA)
        f0 = gm0.get("global_mode_fraction"); fA = gmA.get("global_mode_fraction")
        s0 = gm0.get("structure_response_gain"); sA = gmA.get("structure_response_gain")
        drop = (f0 - fA) if (f0 is not None and fA is not None) else float("nan")
        verdict = ("A_WEB_TO_WIRES_SHIFT" if (drop is not None and drop > 0.05) else
                   "NO_STRUCTURAL_SHIFT")
        print(f"   global_mode_fraction: gamma_A=0 -> {f0}   A-on -> {fA}   drop={drop if isinstance(drop,float) else 'n/a'}")
        print(f"   structure_response_gain: gamma_A=0 -> {s0}   A-on -> {sA}")
        print(f"   -> {verdict}  ({time.time()-t0:.0f}s)\n")
        report.append({"label": lbl, "gamma_A": g, "kappa": kap, "c_A": cA, "params": par,
                       "gmf_gamma0": f0, "gmf_Aon": fA, "gmf_drop": drop,
                       "srg_gamma0": s0, "srg_Aon": sA, "verdict": verdict})
    od = os.path.join(d, "afield_validation.json")
    json.dump(report, open(od, "w"), indent=2, default=float)
    nshift = sum(1 for r in report if r["verdict"] == "A_WEB_TO_WIRES_SHIFT")
    print(f"=== {nshift}/{len(report)} show a web->wires structural shift (global_mode_fraction drop >0.05) ===")
    print("Drop => current-coupled A converts the holistic web toward pairwise/corridor structure "
          "(scout-level mechanistic evidence for the rate-of-interaction hypothesis). No drop => "
          "A localizes on the bridge but does not restructure the coupling.")
    print(f"wrote {od}")


if __name__ == "__main__":
    main()
