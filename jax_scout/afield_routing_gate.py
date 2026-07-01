"""
CORRECTED, DENOMINATOR-SAFE routing gate (validated current-coupled A machinery; NO tensor branch).

Stage B exposed that the bare bridge/void phase-kick RATIO is gameable: when the void response peak
is tiny the ratio explodes, so a NO-bridge config read 33x at lam=0. This module fixes the metric so
a routing claim requires REAL ABSOLUTE responses, not just a large ratio:

  routing gate (all must hold):
    1. bridge response amplitude  >= ABS_BRIDGE_FLOOR        (the bridge genuinely routes)
    2. void response denominator  >= VOID_DENOM_FLOOR         (else ratio meaningless -> reject)
    3. bridge/void (guarded denom) >= BRIDGE_VOID_THR
    4. bridge/node (guarded denom) >= BRIDGE_NODE_THR         (corridor beats poking a node)
    5. node count preserved (settle vs post-kick, no fragmentation)
    6. energy in [0.5,2] AND curvature bounded
    7. seed-robust: the verdict holds across multiple ICs

Plus a CONTROL SUITE: weak/no-bridge configs are run through the SAME gate and MUST be rejected; if
a control passes, the gate is leaky (CONTROL_LEAK).

Classes: BOUNDED_STRONG_BRIDGE_SUBSTRATE / ROUTING_CANDIDATE_WITH_VALID_DENOMINATOR /
CONTROL_LEAK_REJECT / DENOMINATOR_COLLAPSE_REJECT / FRAGMENTATION_REJECT / CURVATURE_RUNAWAY_REJECT /
ENERGY_DRIFT_REJECT / SEED_FRAGILE_REJECT / PROMISING_FOR_PAYAN_PHASE_ALIGNMENT.

Floors are CALIBRATED empirically by the self-test (`--selftest`) on known strong/weak/no-bridge
configs; the values below are provisional until that run prints the absolute amplitudes.
WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/afield_routing_gate.py --selftest
"""
import os, sys, csv, glob, time
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td, geometry_diag as gd
from jax_scout import afield_current_coupled as cc
from jax_scout.afield_current_coupled import multiseed_ic, L

order = physics.SWEEP_PARAM_ORDER
SEED = 20260619
SETTLE, CONT, CSNAP = 800, 400, 50
THETA = 0.4

# --- gate thresholds (CALIBRATED from the window scan, cont=2000) ---
# At cont=2000 the responsive config (gen29) lifts to bridge~0.23 while both controls stay at
# ~3e-4 noise; floors are set in that gap. A real routing candidate must ALSO be resolved (peak
# off the boundary) and bridge-specific (bridge > node) -- the window scan showed neither holds.
ABS_BRIDGE_FLOOR = 0.02    # min absolute bridge-kick response (above control noise ~3e-4)
VOID_DENOM_FLOOR = 0.005   # min void response for the ratio to be meaningful (else collapse-reject)
BRIDGE_VOID_THR = 3.0      # bridge/void (guarded) must exceed this
BRIDGE_NODE_THR = 1.5      # bridge/node (guarded) must exceed this -- corridor beats poking a node
REQUIRE_RESOLVED = True    # peak must be OFF the window boundary (not still-growing drift)
HUNT_CONT = 2000           # continuation window for the hunt (window scan: responses lift here)
ER_LO, ER_HI, CURV_MAX = 0.5, 2.0, 1.0
NODE_LO, NODE_HI = 2, 8
SEEDS = [20260619, 20260620, 20260621]


def _bump_d2(N, c):
    G = np.meshgrid(*([np.arange(N)] * 3), indexing="ij")
    return sum(np.minimum((G[a] - c[a]) % N, (c[a] - G[a]) % N).astype(float) ** 2 for a in range(3))


def _phase_kick(psi, c, N, r):
    return (psi * np.exp(1j * THETA * np.exp(-_bump_d2(N, c) / (2 * (r / 1.5) ** 2)))).astype(np.complex128)


def measure(par, g, kap, cA, N=48, seed=SEED, settle=SETTLE, cont=CONT, csnap=CSNAP):
    """One IC: settle, then phase-kick bridge/node/void and return ABSOLUTE response amplitudes
    (peak fractional energy deviation at the OTHER nodes) + bridge conductance, er, curvature,
    node-preservation. No ratios are floored here -- the gate applies the floors."""
    dx = L / N
    s0, _, fin = cc.capture_cc(par, multiseed_ic(N, seed), g, N, settle, 20, kappa=kap, c_A=cA)
    if not fin:
        return {"status": "settle_nonfinite"}
    psi0 = s0[-1]; nodes = td.detect_nodes(psi0, dx)
    if len(nodes) < 2:
        return {"status": "too_few_nodes", "n_nodes": len(nodes)}
    nodes = sorted(nodes, key=lambda n: -n["E"]); cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes]) ** (1 / 3))))
    n0 = len(nodes)
    er = float(np.sum(np.abs(psi0) ** 2) / (np.sum(np.abs(s0[0]) ** 2) + 1e-30))
    curv = float(gd.curvature_max_only(psi0, par, dx))
    geo = td.geometry_fields(psi0, par, dx); best, bp = -1.0, (0, 1)
    for i in range(n0):
        for j in range(i + 1, n0):
            c = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
            if c > best:
                best, bp = c, (i, j)
    disp = (cents[bp[1]] - cents[bp[0]]).astype(float); disp = disp - N * np.round(disp / N)
    bpt = np.round(cents[bp[0]] + 0.5 * disp).astype(int) % N
    rho = np.abs(psi0) ** 2; far = np.ones((N, N, N), bool)
    for c in cents:
        far &= _bump_d2(N, c) > (2 * node_r) ** 2
    void = np.array(np.unravel_index(np.argmin(np.where(far, rho, rho.max() + 1)), rho.shape))
    omasks = [(_bump_d2(N, c) <= node_r * node_r) for c in cents[1:]]

    def contf(p0):
        s, _, f = cc.capture_cc(par, p0, g, N, cont, csnap, kappa=kap, c_A=cA); return s, f
    sc, f0 = contf(psi0)
    if not f0:
        return {"status": "control_nonfinite"}
    T = sc.shape[0]; ctrlE = np.array([[float(np.sum(np.abs(sc[t][m]) ** 2)) for t in range(T)] for m in omasks])
    n_end = len(td.detect_nodes(sc[-1], dx))   # node preservation under free continuation

    def resp(loc):
        sb, fb = contf(_phase_kick(psi0, loc, N, node_r))
        if not fb:
            return None
        bE = np.array([[float(np.sum(np.abs(sb[t][m]) ** 2)) for t in range(T)] for m in omasks])
        mdev = (np.abs(bE - ctrlE) / (ctrlE[:, :1] + 1e-30)).mean(0)
        return {"peak": float(mdev.max()), "lag": int(np.argmax(mdev)), "boundary": int(np.argmax(mdev)) >= T - 2}
    nk = resp(cents[0]); bk = resp(bpt); vk = resp(void)
    if not (nk and bk and vk):
        return {"status": "kick_nonfinite", "n_nodes": n0, "er": er, "bridge_cond": float(best)}
    return {"status": "ok", "n_nodes": n0, "n_nodes_end": n_end, "er": er, "curv": curv,
            "bridge_cond": float(best),
            "bridge_amp": bk["peak"], "node_amp": nk["peak"], "void_amp": vk["peak"],
            "bridge_resolved": (not bk["boundary"]), "bridge_lag_frac": bk["lag"] / (T - 1)}


def gate_one(m):
    """Apply the denominator-safe gate to ONE measurement (single seed). Returns (passes, reason)."""
    if m.get("status") != "ok":
        return False, m.get("status", "no_measure")
    if not (ER_LO <= m["er"] <= ER_HI):
        return False, "ENERGY_DRIFT"
    if m["curv"] >= CURV_MAX:
        return False, "CURVATURE_RUNAWAY"
    if abs(m["n_nodes_end"] - m["n_nodes"]) > 1 or m["n_nodes_end"] > NODE_HI:
        return False, "FRAGMENTATION"
    if m["bridge_amp"] < ABS_BRIDGE_FLOOR:
        return False, "WEAK_ABS_BRIDGE"
    if m["void_amp"] < VOID_DENOM_FLOOR:
        return False, "DENOMINATOR_COLLAPSE"
    if REQUIRE_RESOLVED and not m.get("bridge_resolved", False):
        return False, "NOT_RESOLVED"        # boundary-pinned = generic drift, not a settled signal
    bv = m["bridge_amp"] / max(m["void_amp"], VOID_DENOM_FLOOR)
    bn = m["bridge_amp"] / max(m["node_amp"], VOID_DENOM_FLOOR)
    if bv < BRIDGE_VOID_THR:
        return False, "WEAK_BRIDGE_VOID"
    if bn < BRIDGE_NODE_THR:
        return False, "WEAK_BRIDGE_NODE"     # bridge must beat poking a node (bridge-specific)
    return True, "OK"


def classify(par, g, kap, cA, role="candidate", N=48, seeds=SEEDS, cont=HUNT_CONT):
    """Multi-seed routing verdict for one config. role in {candidate, weak_control, no_bridge_control}."""
    ms = [measure(par, g, kap, cA, N=N, seed=sd, cont=cont) for sd in seeds]
    oks = [m for m in ms if m.get("status") == "ok"]
    gates = [gate_one(m) for m in ms]
    n_pass = sum(1 for p, _ in gates if p)
    # substrate-level summary (median of finite measures)
    def med(key):
        vals = [m[key] for m in oks if key in m]
        return float(np.median(vals)) if vals else None
    summ = {"role": role, "n_seeds": len(seeds), "n_ok": len(oks), "n_pass": n_pass,
            "bridge_cond": med("bridge_cond"), "bridge_amp": med("bridge_amp"),
            "node_amp": med("node_amp"), "void_amp": med("void_amp"), "er": med("er"),
            "n_nodes": med("n_nodes"), "reasons": [r for _, r in gates]}
    # classification
    if len(oks) == 0:
        summ["klass"] = "ENERGY_DRIFT_REJECT" if any(m.get("status") in ("settle_nonfinite", "control_nonfinite") for m in ms) else "SUBSTRATE_UNUSABLE"
        return summ
    # dominant single-seed failure reason (for typed rejects)
    fails = [r for p, r in gates if not p]
    bc = summ["bridge_cond"] or 0.0
    substrate_ok = (bc is not None and bc >= 0.15 and bc <= 0.85 and (summ["er"] or 0) >= ER_LO
                    and (summ["er"] or 99) <= ER_HI and 2 <= (summ["n_nodes"] or 0) <= 8)
    if role != "candidate":
        # CONTROL: must NOT pass; passing => leak
        summ["klass"] = "CONTROL_LEAK_REJECT" if n_pass >= max(2, len(seeds) - 1) else "CONTROL_CLEAN"
        return summ
    if n_pass >= max(2, len(seeds) - 1):          # robust pass (>=2/3 seeds)
        summ["klass"] = "ROUTING_CANDIDATE_WITH_VALID_DENOMINATOR"
    elif n_pass >= 1:
        summ["klass"] = "SEED_FRAGILE_REJECT"
    elif "FRAGMENTATION" in fails:
        summ["klass"] = "FRAGMENTATION_REJECT"
    elif "CURVATURE_RUNAWAY" in fails:
        summ["klass"] = "CURVATURE_RUNAWAY_REJECT"
    elif "ENERGY_DRIFT" in fails:
        summ["klass"] = "ENERGY_DRIFT_REJECT"
    elif "DENOMINATOR_COLLAPSE" in fails or "WEAK_ABS_BRIDGE" in fails:
        summ["klass"] = "DENOMINATOR_COLLAPSE_REJECT"
    elif substrate_ok:
        summ["klass"] = "BOUNDED_STRONG_BRIDGE_SUBSTRATE"   # good substrate, no valid routing
    else:
        summ["klass"] = "SUBSTRATE_UNUSABLE"
    return summ


# ---- config selection for the self-test ----
def F(r, k):
    try: return float(r[k])
    except: return float("nan")


def pick_known(d, n_strong=2):
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    bnd = [r for r in rows if r["reject"] == "" and 0.5 <= F(r, "er") <= 2.0 and 2 <= F(r, "nodes") <= 8]
    strong = sorted([r for r in bnd if 0.3 < F(r, "bridge") < 0.85], key=lambda r: -F(r, "bridge"))[:n_strong]
    weak = sorted([r for r in bnd if 0.1 <= F(r, "bridge") <= 0.25], key=lambda r: F(r, "bridge"))[:1]
    none = sorted([r for r in bnd if F(r, "bridge") < 0.05], key=lambda r: F(r, "bridge"))[:1]
    out = []
    for r, role in ([(x, "candidate") for x in strong] + [(x, "weak_control") for x in weak]
                    + [(x, "no_bridge_control") for x in none]):
        out.append(({k: F(r, k) for k in order}, F(r, "gamma_A"), F(r, "kappa"), F(r, "c_A"),
                    f"gen{r['gen']}_br{F(r,'bridge'):.2f}", role))
    return out


def selftest():
    """Print ABSOLUTE bridge/node/void amplitudes (1 seed) on strong/weak/no-bridge configs to
    CALIBRATE the floors -- the no-bridge control MUST end up rejected by the absolute floors."""
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "AF_BRIDGE_HUNT_2026*")))[-1]
    panel = pick_known(d)
    print(f"=== ROUTING GATE SELF-TEST (absolute amplitudes, 1 seed, current-coupled A, no tensor) ===")
    print(f"provisional floors: ABS_BRIDGE={ABS_BRIDGE_FLOOR} VOID_DENOM={VOID_DENOM_FLOOR} "
          f"bridge/void>={BRIDGE_VOID_THR} bridge/node>={BRIDGE_NODE_THR}\n")
    for par, g, kap, cA, label, role in panel:
        t0 = time.time(); m = measure(par, g, kap, cA, seed=SEED)
        if m.get("status") == "ok":
            bv = m["bridge_amp"] / max(m["void_amp"], 1e-9); bn = m["bridge_amp"] / max(m["node_amp"], 1e-9)
            p, why = gate_one(m)
            print(f"[{label}] role={role} cond={m['bridge_cond']:.3f} nodes={m['n_nodes']}->{m['n_nodes_end']} er={m['er']:.2f}")
            print(f"    ABS amps  bridge={m['bridge_amp']:.4f} node={m['node_amp']:.4f} void={m['void_amp']:.4f}  "
                  f"| bridge/void={bv:.2f} bridge/node={bn:.2f} | gate={'PASS' if p else 'fail:'+why}  ({time.time()-t0:.0f}s)\n")
        else:
            print(f"[{label}] role={role} status={m.get('status')}  ({time.time()-t0:.0f}s)\n")
    print("CALIBRATION TARGET: set ABS_BRIDGE_FLOOR / VOID_DENOM_FLOOR so the no-bridge control FAILS "
          "(its bridge_amp/void_amp are noise) while genuine strong bridges, if any, can pass.")


def window_scan():
    """The CONT=400 self-test showed ALL configs (strong/weak/no-bridge) at ~1e-4 ABSOLUTE response
    with boundary-pinned peaks -> the window may be too short to resolve inter-node routing. Scan the
    continuation window on the strongest-settle-bridge config + the no-bridge control: does a longer
    window produce a RESOLVED (peak off the boundary), above-noise absolute bridge response that
    SEPARATES the bridge config from the no-bridge control? If yes -> set the floor there and hunt.
    If everything stays at noise even when resolved -> 'no routing' is real (not a window artifact)."""
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "AF_BRIDGE_HUNT_2026*")))[-1]
    panel = pick_known(d)
    WINDOWS = [800, 2000, 4000]
    print("=== ROUTING WINDOW-RESOLUTION SCAN (absolute amps vs continuation length) ===")
    print("question: does a longer window resolve an above-noise bridge response that separates the "
          "bridge config from the no-bridge control? (else 'no routing' is real)\n")
    for par, g, kap, cA, label, role in panel:
        print(f"[{label}] role={role}", flush=True)
        for cont in WINDOWS:
            t0 = time.time(); m = measure(par, g, kap, cA, seed=SEED, cont=cont, csnap=min(50, cont // 20))
            if m.get("status") == "ok":
                bv = m["bridge_amp"] / max(m["void_amp"], 1e-12)
                print(f"    cont={cont:<5} bridge={m['bridge_amp']:.5f} node={m['node_amp']:.5f} "
                      f"void={m['void_amp']:.5f} | b/v={bv:.1f} resolved={m['bridge_resolved']} "
                      f"lag_frac={m['bridge_lag_frac']:.2f} cond={m['bridge_cond']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"    cont={cont:<5} {m.get('status')}  ({time.time()-t0:.0f}s)", flush=True)
        print(flush=True)
    print("READ: an above-noise (>~1e-2) RESOLVED bridge response on the strong config that the "
          "no-bridge control does NOT show => routing is real, calibrate floor. Flat ~1e-4 noise at "
          "all windows, or bridge~control => no genuine routing (PROMISING_FOR_PAYAN evidence).")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--window-scan" in sys.argv:
        window_scan()
    else:
        print("use --selftest / --window-scan (calibration) or import classify() for the hunt")
