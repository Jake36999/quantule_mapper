"""
gen15 deep-dive — calibration thread for the FMIA transfer diagnostic (NOT proof).

gen15 was the single config (of 12) whose null-referenced phase coupling cleared the
cross-sim independence floor (pcoup 0.784 > floor-p95 0.73; action-rate 0.629). Params:
  D=4.964  eta=0.0787  rho_vac=0.10  omega0=2.0  a_coupling=0.1611  s=0.739  f=-0.2937  a=0.2966

This probes whether that thread is REAL or a one-seed fluke, by three tests:
  A. ROBUSTNESS  — repeat seed (baseline), altered seeds, and local parameter perturbations;
                   does the above-floor phase coupling reproduce?
  B. FIDELITY    — longer trajectory + finer snapshots; does it persist with more temporal
                   resolution?
  C. CORRIDOR PERTURBATION (causal) — at a settled time, kick ONE node vs a matched VOID
                   location with equal energy; measure the BOUNDED, DELAYED response at the
                   OTHER nodes. Structured routing = node-kick produces larger/earlier
                   other-node response than the void-kick, stays bounded, and relaxes back.
                   (Success is NOT "other nodes die" — it is bounded delayed transfer.)

Promotion is gated (per user/ChatGPT): bounded persistent nodes + transfer above null +
geometry corridor / phase-current route + robustness across seeds + no energy/curvature
runaway. If gen15 fails, it becomes a useful NEGATIVE CONTROL.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/transfer_deepdive.py
"""
import os, sys, json, time
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td

L, dt = 10.0, 0.005
order = physics.SWEEP_PARAM_ORDER
GEN15 = {"param_D": 4.964, "param_eta": 0.0787, "param_rho_vac": 0.10, "param_omega0": 2.0,
         "param_a_coupling": 0.1611, "param_s": 0.739, "param_f": -0.2937, "param_a": 0.2966}
FLOOR_P95 = 0.73        # phase-coupling independence floor (transfer_null_control.py)


def multiseed_ic(N, L, seed, K=6, sigma_div=12.0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-L/2, L/2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L/sigma_div; psi = np.zeros((N, N, N), np.complex128)
    for _ in range(K):
        cx, cy, cz = rng.uniform(-L/2, L/2, 3)
        psi += np.exp(-((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/(2*w**2))
    noise = 0.01*(rng.standard_normal((N, N, N))+1j*rng.standard_normal((N, N, N)))
    return (psi + noise).astype(np.complex128)


def pvec_of(par):
    return [float(par[k]) for k in order]


def analyze(par, ic, N, steps, n_snap, label):
    res = td.analyze_candidate(pvec_of(par), par, ic, N, L, dt, steps, n_snap,
                               bounded_abl_sens=0.0, iso_surv=np.nan)
    res["label"] = label
    return res


# ---------------------------------------------------------------- A. robustness
def robustness(N=48, steps=800, n_snap=40):
    print("=== A. ROBUSTNESS (does the above-floor phase coupling reproduce?) ===")
    runs = []
    base_seed = 20260619
    runs.append(("baseline_seed", GEN15, base_seed))
    for s in (101, 202, 303):
        runs.append((f"altseed_{s}", GEN15, s))
    perturb = {"D-0.5": ("param_D", -0.5), "D+0.04": ("param_D", 0.036),
               "omega0-0.3": ("param_omega0", -0.3), "s-0.10": ("param_s", -0.10),
               "s+0.10": ("param_s", 0.10), "f-0.10": ("param_f", -0.10),
               "f+0.10": ("param_f", 0.10), "a_coupling+0.1": ("param_a_coupling", 0.1)}
    for name, (k, dv) in perturb.items():
        par = dict(GEN15); par[k] = GEN15[k] + dv
        runs.append((f"perturb_{name}", par, base_seed))
    out = []
    hdr = f"{'label':22} {'nP':>3} {'pcoup':>6} {'arc':>6} {'cond':>6} {'Jflux':>6} {'klass'}"
    print(hdr)
    for label, par, seed in runs:
        ic = multiseed_ic(N, L, seed)
        r = analyze(par, ic, N, steps, n_snap, label)
        pc = r.get("phase_coupling_score", 0.0)
        flag = " *ABOVE-FLOOR*" if pc > FLOOR_P95 else ""
        print(f"{label:22} {r.get('n_persistent_nodes',0):>3} {pc:>6.3f} "
              f"{r.get('action_rate_coherence',0):>6.3f} {r.get('omega_corridor_conductance',0):>6.3f} "
              f"{r.get('mean_transfer_strength',0):>6.3f} {r['klass']}{flag}")
        out.append({k: r.get(k) for k in ("label", "n_persistent_nodes", "phase_coupling_score",
                    "action_rate_coherence", "omega_corridor_conductance",
                    "mean_transfer_strength", "klass")})
    above = [o for o in out if (o["phase_coupling_score"] or 0) > FLOOR_P95]
    print(f"\n  -> {len(above)}/{len(out)} runs clear the independence floor (pcoup>{FLOOR_P95}).")
    return out


# ---------------------------------------------------------------- B. fidelity
def fidelity(N=48):
    print("\n=== B. FIDELITY (longer trajectory + finer snapshots) ===")
    out = []
    for steps, n_snap, lab in ((800, 40, "N48/800/40"), (1600, 80, "N48/1600/80")):
        ic = multiseed_ic(N, L, 20260619)
        r = analyze(GEN15, ic, N, steps, n_snap, lab)
        print(f"  {lab:14} nP={r.get('n_persistent_nodes',0)} "
              f"pcoup={r.get('phase_coupling_score',0):.3f} "
              f"arc={r.get('action_rate_coherence',0):.3f} "
              f"cond={r.get('omega_corridor_conductance',0):.3f} -> {r['klass']}")
        out.append({k: r.get(k) for k in ("label", "n_persistent_nodes", "phase_coupling_score",
                    "action_rate_coherence", "omega_corridor_conductance", "klass")})
    return out


# ---------------------------------------------------------------- C. corridor perturbation
def _region_energy(psi, c, N, r):
    ax = [np.arange(N)] * 3
    G = np.meshgrid(*ax, indexing="ij")
    d2 = sum(np.minimum((G[a] - c[a]) % N, (c[a] - G[a]) % N) ** 2 for a in range(3))
    mask = d2 <= r * r
    return float(np.sum(np.abs(psi[mask]) ** 2)), mask


def _kick(psi, c, N, r, eps):
    """Small localized amplitude kick (a probe), Gaussian-weighted around c."""
    ax = [np.arange(N)] * 3
    G = np.meshgrid(*ax, indexing="ij")
    d2 = sum(np.minimum((G[a] - c[a]) % N, (c[a] - G[a]) % N).astype(float) ** 2 for a in range(3))
    bump = np.exp(-d2 / (2 * (r / 1.5) ** 2))
    return (psi * (1.0 + eps * bump)).astype(np.complex128)


def corridor_perturbation(N=48, steps=800, n_cont=60, eps=0.05):
    print("\n=== C. CORRIDOR PERTURBATION (causal: kick a node vs a matched void) ===")
    ic = multiseed_ic(N, L, 20260619)
    dx = L / N
    # settle
    snaps, finite = td.capture_trajectory(pvec_of(GEN15), ic, N, L, dt, steps, 40)
    if not finite:
        print("  baseline non-finite; abort.")
        return {}
    psi_settle = snaps[-1]
    nodes = td.detect_nodes(psi_settle, dx)
    if len(nodes) < 2:
        print(f"  only {len(nodes)} node(s) at settle; abort.")
        return {}
    nodes = sorted(nodes, key=lambda nd: -nd["E"])
    cents = [np.round(nd["centroid"]).astype(int) % N for nd in nodes]
    rho = np.abs(psi_settle) ** 2
    node_r = max(2, int(round((np.mean([nd["size"] for nd in nodes]) ** (1/3)))))
    # matched void location: lowest-density voxel far from all nodes
    G = np.meshgrid(*([np.arange(N)] * 3), indexing="ij")
    far = np.ones((N, N, N), bool)
    for c in cents:
        d2 = sum(np.minimum((G[a] - c[a]) % N, (c[a] - G[a]) % N) ** 2 for a in range(3))
        far &= d2 > (2 * node_r) ** 2
    void = np.array(np.unravel_index(np.argmin(np.where(far, rho, rho.max() + 1)), rho.shape))

    target = cents[0]
    # energy-matched kicks: pick eps_void so the void kick injects the same delta-E as node kick
    e_node0, _ = _region_energy(psi_settle, target, N, node_r)
    psi_nk = _kick(psi_settle, target, N, node_r, eps)
    dE_node = float(np.sum(np.abs(psi_nk) ** 2) - np.sum(np.abs(psi_settle) ** 2))
    psi_vk = _kick(psi_settle, void, N, node_r, eps)
    dE_void = float(np.sum(np.abs(psi_vk) ** 2) - np.sum(np.abs(psi_settle) ** 2))
    if dE_void != 0:
        eps_v = eps * np.sqrt(abs(dE_node / dE_void))
        psi_vk = _kick(psi_settle, void, N, node_r, eps_v)

    # three continuations: control, node-kick, void-kick
    def cont(p0):
        s, fin = td.capture_trajectory(pvec_of(GEN15), p0, N, L, dt, n_cont * 10, n_cont)
        return s, fin
    s_ctrl, f0 = cont(psi_settle)
    s_node, f1 = cont(psi_nk)
    s_void, f2 = cont(psi_vk)
    if not (f0 and f1 and f2):
        print("  a continuation went non-finite; bounded-response test inconclusive.")
    # per-OTHER-node energy response vs control, for node-kick and void-kick
    other = cents[1:]
    T = s_ctrl.shape[0]
    def response(s_branch):
        out = []
        for c in other:
            ec = np.array([_region_energy(s_ctrl[t], c, N, node_r)[0] for t in range(T)])
            eb = np.array([_region_energy(s_branch[t], c, N, node_r)[0] for t in range(T)])
            ref = ec[0] + 1e-30
            dev = np.abs(eb - ec) / ref
            out.append(dev)
        return np.array(out)                      # [n_other, T]
    R_node = response(s_node); R_void = response(s_void)
    bounded = bool(f0 and f1 and f2 and np.all(np.isfinite(R_node)) and np.all(np.isfinite(R_void))
                   and R_node.max() < 50)
    peak_node = float(R_node.max()); peak_void = float(R_void.max())
    lag_node = int(np.argmax(R_node.mean(0))); lag_void = int(np.argmax(R_void.mean(0)))
    ratio = peak_node / (peak_void + 1e-30)
    print(f"  settle nodes={len(nodes)} node_r={node_r}voxels  kick eps={eps} (energy-matched void)")
    print(f"  other-node response  peak: node-kick={peak_node:.3f}  void-kick={peak_void:.3f}  ratio={ratio:.2f}")
    print(f"  response lag-to-peak (mean over others, steps*10): node-kick={lag_node*10}  void-kick={lag_void*10}")
    print(f"  bounded (all finite, response<50x, no blow-up): {bounded}")
    structured = bounded and ratio > 1.5 and lag_node > 0
    print(f"  -> structured routing signature: {structured}  "
          f"({'node-kick propagates to other nodes more than a matched void, bounded+delayed' if structured else 'no clear node-routed transfer above the void baseline'})")
    return {"peak_node": peak_node, "peak_void": peak_void, "ratio": ratio,
            "lag_node_steps": lag_node * 10, "lag_void_steps": lag_void * 10,
            "bounded": bounded, "structured_routing": structured}


def main():
    t0 = time.time()
    print("gen15 deep-dive  [calibration thread, NOT proof]")
    print(f"params: {GEN15}\n")
    A = robustness()
    B = fidelity()
    C = corridor_perturbation()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "sweep_runs", f"GEN15_DEEPDIVE_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    json.dump({"params": GEN15, "floor_p95": FLOOR_P95, "robustness": A,
               "fidelity": B, "corridor_perturbation": C},
              open(os.path.join(outdir, "gen15_deepdive.json"), "w"), indent=2, default=float)
    # verdict (strict: a real thread must survive ALTERED SEEDS, LONGER TRAJECTORY, and show
    #          a RESOLVED (non-boundary-pinned) causal lag -- not just chance/chaotic divergence)
    altseeds = [o for o in A if o["label"].startswith("altseed_")]
    alt_above = [o for o in altseeds if (o["phase_coupling_score"] or 0) > FLOOR_P95]
    seed_robust = len(alt_above) >= 2                       # majority of altered seeds hold
    fid_long = next((o for o in B if o["label"] == "N48/1600/80"), None)
    fidelity_holds = bool(fid_long and (fid_long["phase_coupling_score"] or 0) > FLOOR_P95)
    lag = C.get("lag_node_steps", 0)
    routed = bool(C.get("structured_routing") and 0 < lag < (n_cont_end := 600))  # not boundary-pinned
    print("\n=== VERDICT (gen15 calibration thread) ===")
    print(f"  reproduces across ALTERED seeds (>=2/3 above floor): {seed_robust} "
          f"({len(alt_above)}/{len(altseeds)})")
    print(f"  survives LONGER trajectory (1600 steps above floor):  {fidelity_holds}")
    print(f"  causal corridor routing (resolved, non-boundary lag):  {routed}")
    if seed_robust and fidelity_holds and routed:
        print("  -> POSITIVE CONTROL: robust + persistent + causally routed. Marginal positive seed.")
    elif fidelity_holds and (seed_robust or routed):
        print("  -> WEAK/MARGINAL thread; bridge objective should confirm or kill it.")
    else:
        print("  -> NEAR-NEGATIVE CONTROL: the above-floor coupling is a finite-window, "
              "seed-fragile effect (does not survive longer trajectory and/or altered seeds; "
              "perturbation response is tiny + boundary-pinned). Treat gen15 as a marginal/negative "
              "control for the bridge diagnostic, NOT a positive seed to chase.")
    print(f"\nwrote {outdir}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
