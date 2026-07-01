"""
STAGE 2 — characterize the REAL inter-node coupling in the bridge-objective finalists.

Stage 1 established: bridged finalists propagate a node perturbation to the other nodes
(~0.002-0.003) while a no-corridor control does NOT (~0) -> the coupling is real and
bridge-associated, on a ~2000-step timescale, but it is NOT bridge-selective (bridge~=node).
Whether or not selective routing validates, this coupling is a genuine positive worth
characterizing directly. The question is NOT "can we find more" but "what IS the coupling
channel and timescale".

Probes (energy-conserving phase kicks, theta=0.4 unless scaling):
  * node->node response MATRIX M[i,j] = peak rel. response at node j when node i is kicked,
    + bridge-kick row + void-kick row + the no-corridor control;
  * GLOBAL-MODE vs PAIRWISE decomposition of M (SVD: sigma1^2/sum = global_mode_fraction; the
    rest = pairwise_coupling_fraction; plus row-uniformity = does kicking ANY node give the same
    response pattern = global driver);
  * coupling TIMESCALE (mean time-to-peak; decay time = peak -> half-peak);
  * amplitude SCALING (theta in {0.1,0.2,0.4,0.8}: log-log slope; 1=linear, >1=nonlinear);
  * phase-current and Omega^2 response gains (|J_pert - J_ctrl|/J_ctrl, same for Omega^2).

Outputs (per finalist): structure_response_gain, void_normalized_response,
response_time_to_peak, response_decay_time, node_bridge_selectivity, global_mode_fraction,
pairwise_coupling_fraction, phase_current_response_gain, omega_response_gain,
coupling_timescale_index.

CAUTION: JAX scout-level diagnosis; not proof; no new hunt.
WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/coupling_characterize.py
"""
import os, sys, json, glob, csv, time
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td

L, dt = 10.0, 0.005
order = physics.SWEEP_PARAM_ORDER
BASE_SEED = 20260619
SETTLE = 800
WIN_STEPS, WIN_NSNAP = 2800, 70          # covers the ~2000-step resolved peak (Stage 1)
THETA = 0.4
SCALE_THETAS = [0.1, 0.2, 0.4, 0.8]


def multiseed_ic(N, seed, K=6):
    rng = np.random.default_rng(seed)
    x = np.linspace(-L/2, L/2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L/12.0; psi = np.zeros((N, N, N), np.complex128)
    for _ in range(K):
        cx, cy, cz = rng.uniform(-L/2, L/2, 3)
        psi += np.exp(-((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/(2*w**2))
    noise = 0.01*(rng.standard_normal((N, N, N))+1j*rng.standard_normal((N, N, N)))
    return (psi+noise).astype(np.complex128)


def _bump(N, c, r):
    G = np.meshgrid(*([np.arange(N)]*3), indexing="ij")
    d2 = sum(np.minimum((G[a]-c[a]) % N, (c[a]-G[a]) % N).astype(float)**2 for a in range(3))
    return np.exp(-d2/(2*(r/1.5)**2)), d2


def _phase_kick(psi, c, N, r, theta):
    b, _ = _bump(N, c, r); return (psi*np.exp(1j*theta*b)).astype(np.complex128)


def _regE(psi, m): return float(np.sum(np.abs(psi[m])**2))


def _decay_time(curve, spc):
    """steps from peak down to half-peak (NaN if never recovers)."""
    ip = int(np.argmax(curve)); pk = curve[ip]
    for t in range(ip, len(curve)):
        if curve[t] <= 0.5*pk:
            return (t-ip)*spc
    return float("nan")


def characterize(par, role, label, N=48):
    pv = [par[k] for k in order]; dx = L/N
    snaps, fin = td.capture_trajectory(pv, multiseed_ic(N, BASE_SEED), N, L, dt, SETTLE, 20)
    if not fin:
        return {"status": "settle_nonfinite", "label": label, "role": role}
    psi0 = snaps[-1]; nodes = td.detect_nodes(psi0, dx)
    if len(nodes) < 2:
        return {"status": "too_few_nodes", "label": label, "role": role, "n": len(nodes)}
    nodes = sorted(nodes, key=lambda n: -n["E"]); nn = len(nodes)
    cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes])**(1/3))))
    masks = [(_bump(N, c, node_r)[1] <= node_r*node_r) for c in cents]
    geo0 = td.geometry_fields(psi0, par, dx)
    # strongest bridge
    best, bp = -1, (0, 1)
    for i in range(nn):
        for j in range(i+1, nn):
            c = td.corridor_pair_metrics(geo0, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
            if c > best:
                best, bp = c, (i, j)
    disp = (cents[bp[1]]-cents[bp[0]]).astype(float); disp = disp - N*np.round(disp/N)
    bridge_pt = np.round(cents[bp[0]]+0.5*disp).astype(int) % N
    rho = np.abs(psi0)**2; far = np.ones((N, N, N), bool)
    for c in cents:
        _, d2 = _bump(N, c, 2*node_r); far &= d2 > (2*node_r)**2
    void = np.array(np.unravel_index(np.argmin(np.where(far, rho, rho.max()+1)), rho.shape))
    spc = WIN_STEPS//WIN_NSNAP

    def cont(p0):
        s, f = td.capture_trajectory(pv, p0, N, L, dt, WIN_STEPS, WIN_NSNAP); return s, f
    s_ctrl, f0 = cont(psi0)
    if not f0:
        return {"status": "control_nonfinite", "label": label, "role": role}
    T = s_ctrl.shape[0]
    ctrlE = np.array([[_regE(s_ctrl[t], m) for t in range(T)] for m in masks])   # [nn,T]
    J_ctrl = float(np.sqrt(np.mean(sum(x**2 for x in geo0["J"]))))
    om_ctrl = float(np.sqrt(np.mean(geo0["omega_sq"]**2)))

    def response_row(kick_loc, theta=THETA):
        s_b, fb = cont(_phase_kick(psi0, kick_loc, N, node_r, theta))
        if not fb:
            return None, None, None
        bE = np.array([[_regE(s_b[t], m) for t in range(T)] for m in masks])
        dev = np.abs(bE-ctrlE)/(ctrlE[:, :1]+1e-30)            # [nn,T] per-node rel deviation
        peaks = dev.max(1)                                      # peak per node
        meancurve = dev.mean(0)
        geo_b = td.geometry_fields(s_b[-1], par, dx)
        Jg = abs(float(np.sqrt(np.mean(sum(x**2 for x in geo_b["J"])))) - J_ctrl)/(J_ctrl+1e-30)
        omg = abs(float(np.sqrt(np.mean(geo_b["omega_sq"]**2))) - om_ctrl)/(om_ctrl+1e-30)
        return peaks, meancurve, (Jg, omg)

    # response MATRIX: kick each node, peak response at each other node
    M = np.full((nn, nn), np.nan); t2peaks = []; decays = []; Jgains = []; omgains = []
    for i in range(nn):
        peaks, mc, gains = response_row(cents[i])
        if peaks is None:
            continue
        for j in range(nn):
            if j != i:
                M[i, j] = peaks[j]
        t2peaks.append(int(np.argmax(mc))*spc); decays.append(_decay_time(mc, spc))
        Jgains.append(gains[0]); omgains.append(gains[1])
    # bridge + void rows (response at all nodes)
    bpk, bmc, _ = response_row(bridge_pt); vpk, vmc, _ = response_row(void)
    bridge_peak = float(np.nanmean(bpk)) if bpk is not None else float("nan")
    void_peak = float(np.nanmean(vpk)) if vpk is not None else float("nan")

    res = {"status": "ok", "label": label, "role": role, "n_nodes": nn,
           "best_bridge_conductance": float(best), "steps_per_snap": spc}
    offdiag = M[~np.isnan(M)]
    struct_gain = float(np.mean(offdiag)) if offdiag.size else 0.0
    res["structure_response_gain"] = struct_gain
    res["void_normalized_response"] = struct_gain/(void_peak+1e-30)
    res["node_bridge_selectivity"] = bridge_peak/(struct_gain+1e-30)
    res["response_time_to_peak"] = float(np.nanmean(t2peaks)) if t2peaks else float("nan")
    res["response_decay_time"] = float(np.nanmean(decays)) if decays else float("nan")
    res["coupling_timescale_index"] = (res["response_decay_time"]/res["response_time_to_peak"]
                                       if res["response_time_to_peak"] else float("nan"))
    res["phase_current_response_gain"] = float(np.mean(Jgains)) if Jgains else float("nan")
    res["omega_response_gain"] = float(np.mean(omgains)) if omgains else float("nan")
    # GLOBAL vs PAIRWISE: SVD of the off-diagonal response matrix
    Mf = np.nan_to_num(M, nan=0.0)
    if nn >= 2 and np.any(Mf):
        sv = np.linalg.svd(Mf, compute_uv=False); tot = float(np.sum(sv**2))
        res["global_mode_fraction"] = float(sv[0]**2/tot) if tot > 0 else float("nan")
        res["pairwise_coupling_fraction"] = 1.0 - res["global_mode_fraction"]
        # row uniformity: do all kicked-nodes produce the same response pattern? (global driver)
        rows = Mf / (Mf.sum(1, keepdims=True)+1e-30)
        res["row_pattern_similarity"] = float(np.mean([
            np.dot(rows[a], rows[b])/(np.linalg.norm(rows[a])*np.linalg.norm(rows[b])+1e-30)
            for a in range(nn) for b in range(a+1, nn)])) if nn >= 2 else float("nan")
    res["response_matrix"] = M.tolist()
    return res


def scaling_test(par, N=48):
    """Peak response vs kick amplitude theta (node-0 kick) -> log-log slope."""
    pv = [par[k] for k in order]; dx = L/N
    snaps, fin = td.capture_trajectory(pv, multiseed_ic(N, BASE_SEED), N, L, dt, SETTLE, 20)
    if not fin:
        return {}
    psi0 = snaps[-1]; nodes = td.detect_nodes(psi0, dx)
    if len(nodes) < 2:
        return {}
    nodes = sorted(nodes, key=lambda n: -n["E"]); cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes])**(1/3))))
    masks = [(_bump(N, c, node_r)[1] <= node_r*node_r) for c in cents[1:]]

    def cont(p0):
        s, f = td.capture_trajectory(pv, p0, N, L, dt, WIN_STEPS, WIN_NSNAP); return s, f
    s_ctrl, f0 = cont(psi0)
    if not f0:
        return {}
    T = s_ctrl.shape[0]; ctrlE = np.array([[_regE(s_ctrl[t], m) for t in range(T)] for m in masks])
    peaks = []
    for th in SCALE_THETAS:
        s_b, fb = cont(_phase_kick(psi0, cents[0], N, node_r, th))
        if not fb:
            peaks.append(np.nan); continue
        bE = np.array([[_regE(s_b[t], m) for t in range(T)] for m in masks])
        peaks.append(float((np.abs(bE-ctrlE)/(ctrlE[:, :1]+1e-30)).mean(0).max()))
    th = np.array(SCALE_THETAS); pk = np.array(peaks); good = np.isfinite(pk) & (pk > 0)
    slope = float(np.polyfit(np.log(th[good]), np.log(pk[good]), 1)[0]) if good.sum() >= 2 else float("nan")
    return {"scale_thetas": SCALE_THETAS, "scale_peaks": [float(x) for x in peaks],
            "amplitude_scaling_exponent": slope}


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "BRIDGE_HUNT_2026*")))[-1]
    fz = json.load(open(os.path.join(d, "frozen_finalists.json")))
    targets = [(f"gen{fz['finalists'][k]['generation']}_{fz['finalists'][k]['config_hash']}",
                {kk: float(fz['finalists'][k]['params'][kk]) for kk in order}, "finalist") for k in (0, 2)]
    # no-corridor control
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    def F(r, k):
        try: return float(r[k])
        except: return float("nan")
    nc = sorted([r for r in rows if r["klass"] == "no_corridor_stable_nodes" and 2 <= F(r, "nodes") <= 8
                 and F(r, "max_cond") < 0.05 and 0.5 <= F(r, "er") <= 2.0], key=lambda r: F(r, "max_cond"))[0]
    targets.append((f"NEGCTRL_gen{nc['gen']}", {k: F(nc, k) for k in order}, "no_corridor_control"))

    print(f"STAGE 2 coupling characterization ({WIN_STEPS} steps) — {len(targets)} configs\n")
    report = []
    for label, par, role in targets:
        t0 = time.time(); r = characterize(par, role, label)
        if r.get("status") == "ok" and role == "finalist":
            r.update(scaling_test(par))
        report.append(r); _print(r, time.time()-t0)
    od = os.path.join(d, "coupling_characterization.json")
    json.dump(report, open(od, "w"), indent=2, default=float)
    print(f"wrote {od}")


def _print(r, secs):
    if r.get("status") != "ok":
        print(f"[{r['label']}] {r.get('status')}  ({secs:.0f}s)\n"); return
    print(f"[{r['label']}] role={r['role']} nNodes={r['n_nodes']} bridge_cond={r['best_bridge_conductance']:.3f}")
    print(f"   structure_response_gain={r['structure_response_gain']:.4f} "
          f"void_normalized={r['void_normalized_response']:.2f} node_bridge_selectivity={r['node_bridge_selectivity']:.2f}")
    print(f"   time_to_peak={r['response_time_to_peak']:.0f}st decay_time={r['response_decay_time']}st "
          f"timescale_index={r['coupling_timescale_index']}")
    print(f"   global_mode_fraction={r.get('global_mode_fraction')} pairwise_fraction={r.get('pairwise_coupling_fraction')} "
          f"row_similarity={r.get('row_pattern_similarity')}")
    print(f"   phase_current_gain={r['phase_current_response_gain']:.3f} omega_gain={r['omega_response_gain']:.4f} "
          f"amp_scaling_exp={r.get('amplitude_scaling_exponent')}")
    print(f"   ({secs:.0f}s)\n")


if __name__ == "__main__":
    main()
