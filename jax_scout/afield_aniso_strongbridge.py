"""
STAGE A — focused anisotropy follow-up on STRONG-BRIDGE configs (is gen29 reproducible?).

gen29 (strongest bridge) under the conservative dispersive anisotropic metric (lam=0.1) dropped
global_mode_fraction 1.00->0.665, bounded, bridge preserved = first clean web->wires shift. But it
was 1/3. This tests whether the effect REPRODUCES on OTHER strong-bridge substrates.

Selection: strong-bridge configs (bridge>0.3, er in [0.5,2], not saturated <0.85, 2-8 nodes) from
the A-coupled hunt, + a weak-bridge control + a no-bridge negative control. For each: global_mode
at lam=0 vs lam=0.1 (conservative/dispersive form, q=stress). Promotion criterion
ANISOTROPY_STRONG_BRIDGE_REPRODUCED if >=2 strong-bridge configs show: energy bounded, bridge
preserved, global_mode drop > 0.1, pairwise rise, no runaway/saturation/node-destruction.
Else: GEN29_ANISOTROPY_SINGLETON_SIGNAL_NOT_REPRODUCED.

CAUTION: JAX scout; conservative dispersive form only (real diffusion form is dissipative, not
used). Not proof. No Hunter. WSL2 jax venv:  python jax_scout/afield_aniso_strongbridge.py
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
from jax_scout.afield_anisotropic import global_mode_aniso

order = physics.SWEEP_PARAM_ORDER
LAM, QSRC = 0.1, "stress"


def F(r, k):
    try: return float(r[k])
    except: return float("nan")


def select(d):
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    bnd = [r for r in rows if r["reject"] == "" and 0.5 <= F(r, "er") <= 2.0 and 2 <= F(r, "nodes") <= 8]
    strong = sorted([r for r in bnd if 0.3 < F(r, "bridge") < 0.85], key=lambda r: -F(r, "bridge"))[:4]
    weak = sorted([r for r in bnd if 0.1 <= F(r, "bridge") <= 0.25], key=lambda r: F(r, "bridge"))[:1]
    none = sorted([r for r in bnd if F(r, "bridge") < 0.05], key=lambda r: F(r, "bridge"))[:1]
    out = []
    for r, role in ([(x, "strong") for x in strong] + [(x, "weak") for x in weak]
                    + [(x, "no_bridge") for x in none]):
        out.append(({k: F(r, k) for k in order}, F(r, "gamma_A"), F(r, "kappa"), F(r, "c_A"),
                    f"gen{r['gen']}_br{F(r,'bridge'):.2f}", role, F(r, "bridge")))
    return out


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "AF_BRIDGE_HUNT_2026*")))[-1]
    panel = select(d)
    print(f"STAGE A anisotropy strong-bridge reproduction (lam={LAM}, q={QSRC}) — {len(panel)} configs\n")
    report = []; reproduced = 0; n_strong = 0
    for par, g, kap, cA, label, role, br0 in panel:
        t0 = time.time()
        gm0 = global_mode_aniso(par, g, kap, cA, 0.0, QSRC)
        gmA = global_mode_aniso(par, g, kap, cA, LAM, QSRC)
        f0 = gm0.get("global_mode_fraction"); fA = gmA.get("global_mode_fraction")
        drop = (f0 - fA) if (f0 is not None and fA is not None) else None
        er = gmA.get("er"); brA = gmA.get("bridge"); nA = gmA.get("n_nodes")
        bounded = er is not None and 0.5 <= er <= 2.0 and (nA or 99) <= 8 and (brA or 1) < 0.9
        bridge_preserved = brA is not None and brA > 0.3*br0 and brA > 0.05
        ok = (role == "strong" and drop is not None and drop > 0.1 and bounded and bridge_preserved
              and fA is not None and (1-fA) > (1-f0))
        if role == "strong":
            n_strong += 1
            if ok:
                reproduced += 1
        print(f"[{label}] role={role} bridge0={br0:.2f}")
        print(f"   global_mode: lam0={f0}  lam{LAM}={fA}  drop={drop if isinstance(drop,float) else 'n/a'} "
              f"| er={er} bridge_lamon={brA} nodes={nA} {'REPRODUCED' if ok else ''}  ({time.time()-t0:.0f}s)\n")
        report.append({"label": label, "role": role, "bridge0": br0, "gmf_lam0": f0, "gmf_lamon": fA,
                       "drop": drop, "er": er, "bridge_lamon": brA, "nodes": nA, "reproduced": bool(ok)})
    verdict = ("ANISOTROPY_STRONG_BRIDGE_REPRODUCED" if reproduced >= 2 else
               "GEN29_ANISOTROPY_SINGLETON_SIGNAL_NOT_REPRODUCED")
    json.dump({"lam": LAM, "q_source": QSRC, "n_strong": n_strong, "reproduced": reproduced,
               "verdict": verdict, "panel": report},
              open(os.path.join(d, "afield_aniso_strongbridge.json"), "w"), indent=2, default=float)
    print(f"=== {reproduced}/{n_strong} strong-bridge configs reproduced the bounded web->wires drop ===")
    print(f"VERDICT: {verdict}")
    print("REPRODUCED(>=2): conservative anisotropy reliably restructures strong-bridge webs -> "
          "Stage B (proper conservative tensor-geometry branch). SINGLETON: gen29 was a one-off -> "
          "do not build the tensor branch yet.")


if __name__ == "__main__":
    main()
