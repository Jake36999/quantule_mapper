"""
PHASE A — characterize feb56dc7 as the current confirmed stable object:
LONG_TIME_STABLE_4_NODE_BOUND_STATE (T=6000-confirmed bounded multi-node steady state; NOT 'infinite'
/ 'ground state' / 'molecule'). Bare S-NCGL (gamma_A=0). No solver mod.

Probes: (1) node GEOMETRY (final positions, pairwise distances, arrangement, boundary proximity);
(2) per-node ANATOMY (radial rho(r)/v_r/v_t profile, core density, residual micro-circulation);
(3) INTER-NODE corridors (conductance, J flux, Omega^2 smoothness -> bond vs passive coexistence);
(4) DYNAMICS over time (node tracks, pairwise distances, er, nodes, core density, v_t/v_r through the
rotation->steady relaxation); (5) PERTURBATION / BOND tests (phase-kick, density attenuate x0.5,
density amplify x1.5 on ONE node) -> classify BOUND_STATE_RESTORING / _RECONFIGURES / _FRAGILE /
_PASSIVE_COEXISTENCE / _ARTIFACT_REJECT.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/feb_bound_state.py
"""
import os, sys, json, time
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import transfer_diag as td
from jax_scout.afield_current_coupled import multiseed_ic, L as L_
from jax_scout import core_characterize as cc   # capture (bare), shell_metrics, radial_profile, _bump, order, N

N, order = cc.N, cc.order
PAR = {"param_D": 2.7329, "param_eta": 0.0704, "param_rho_vac": 1.1866, "param_omega0": 0.0,
       "param_a_coupling": 2.3098, "param_s": 0.0129, "param_f": -0.4861, "param_a": 0.4802}
T_LONG, NSNAP, SEED = 6000, 60, 20260619
T_PERT, NSNAP_P = 2000, 20


def _pair_dists(cents):
    out = []
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            d = (np.array(cents[i], float) - np.array(cents[j], float))
            d -= N * np.round(d / N)
            out.append(((i, j), float(np.linalg.norm(d))))
    return out


def geometry(psi, dx):
    nodes = sorted(td.detect_nodes(psi, dx), key=lambda n: -n["E"])
    cents = [(np.round(n["centroid"]).astype(int) % N).tolist() for n in nodes]
    pd = _pair_dists(cents)
    ds = [d for _, d in pd]
    return {"n_nodes": len(nodes), "centroids": cents, "node_E": [float(n["E"]) for n in nodes],
            "pair_dists": [{"pair": list(p), "dist": d} for p, d in pd],
            "dist_mean": float(np.mean(ds)) if ds else 0.0, "dist_std": float(np.std(ds)) if ds else 0.0,
            "dist_min": float(min(ds)) if ds else 0.0, "dist_max": float(max(ds)) if ds else 0.0}, nodes, cents


def corridors(psi, par, nodes, dx):
    if len(nodes) < 2:
        return []
    geo = td.geometry_fields(psi, par, dx); out = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            m = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)
            out.append({"pair": [i, j], "conductance": float(m["conductance"]), "J_flux": float(m["J_flux"]),
                        "omega_smooth": float(m["omega_smooth"]), "path_align": float(m["path_align"])})
    return out


def track(prev, cents, thresh):
    used = set()
    for tr in prev:
        last = np.array(tr[-1], float); best, bi = thresh, -1
        for i, c in enumerate(cents):
            if i in used:
                continue
            d = np.array(c, float) - last; d -= N * np.round(d / N); dist = float(np.linalg.norm(d))
            if dist < best:
                best, bi = dist, i
        if bi >= 0:
            tr.append(cents[bi]); used.add(bi)
    for i, c in enumerate(cents):
        if i not in used:
            prev.append([c])
    return prev


def node_mask(c, r):
    return cc._bump(c) <= r * r


def main():
    dx = L_ / N; psi0 = multiseed_ic(N, SEED); t0 = time.time()
    snaps, fin = cc.capture(PAR, psi0, T_LONG, NSNAP)
    nfin = max((t for t in range(snaps.shape[0]) if np.all(np.isfinite(np.abs(snaps[t])))), default=0) + 1
    e0 = float(np.sum(np.abs(snaps[0]) ** 2)) + 1e-30
    # dynamics + tracks + pairwise distances
    tracks = []; dyn = []
    for t in range(nfin):
        psi = snaps[t]; g, nodes, cents = geometry(psi, dx)
        er = float(np.sum(np.abs(psi) ** 2) / e0)
        dom = (np.round(nodes[0]["centroid"]).astype(int) % N) if nodes else None
        sm = cc.shell_metrics(psi, PAR, dom, dx, 1, 4)[0] if dom is not None else {}
        track(tracks, cents, thresh=8)
        dyn.append({"t_step": t * (T_LONG // NSNAP), "er": er, "n_nodes": g["n_nodes"],
                    "dist_mean": g["dist_mean"], "dist_std": g["dist_std"],
                    "core_rho": sm.get("core_rho"), "v_t": sm.get("v_t"), "v_r": sm.get("v_r")})
    # final steady state
    psi_f = snaps[nfin - 1]; geo_f, nodes_f, cents_f = geometry(psi_f, dx)
    cor_f = corridors(psi_f, PAR, nodes_f, dx)
    profs = {}
    for k, n in enumerate(nodes_f[:4]):
        c = np.round(n["centroid"]).astype(int) % N
        profs[f"node{k}"] = {"centroid": c.tolist(), "core_rho": float(np.abs(psi_f[tuple(c)]) ** 2),
                             **cc.radial_profile(psi_f, PAR, c, dx)}
        sm = cc.shell_metrics(psi_f, PAR, c, dx, 1, 4)[0]
        profs[f"node{k}"]["residual_v_t"] = sm["v_t"]; profs[f"node{k}"]["residual_v_r"] = sm["v_r"]
    # perturbation / bond tests from the settled state
    settle = snaps[nfin - 1]
    base_d = geo_f["dist_mean"]; base_n = geo_f["n_nodes"]
    pert_results = {}
    dom_c = cents_f[0]; node_r = max(2, int(round(np.mean([n["size"] for n in nodes_f]) ** (1 / 3))))
    perts = {"phase_kick": lambda p: cc.phase_kick(p, dom_c, node_r, 0.6),
             "attenuate_x0.5": lambda p: (p * np.where(node_mask(dom_c, node_r), 0.5, 1.0)).astype(np.complex128),
             "amplify_x1.5": lambda p: (p * np.where(node_mask(dom_c, node_r), 1.5, 1.0)).astype(np.complex128)}
    for name, fn in perts.items():
        ps, _ = cc.capture(PAR, fn(settle), T_PERT, NSNAP_P)
        nf = max((t for t in range(ps.shape[0]) if np.all(np.isfinite(np.abs(ps[t])))), default=0) + 1
        ser = []
        for t in range(nf):
            g, _, _ = geometry(ps[t], dx)
            ser.append({"t_step": t * (T_PERT // NSNAP_P), "n_nodes": g["n_nodes"], "dist_mean": g["dist_mean"]})
        gf, _, _ = geometry(ps[nf - 1], dx)
        d_ret = abs(gf["dist_mean"] - base_d) / (base_d + 1e-9)
        restored = (gf["n_nodes"] == base_n) and (d_ret < 0.15)
        reconf = (gf["n_nodes"] == base_n) and (d_ret >= 0.15)
        klass = ("RESTORING" if restored else "RECONFIGURES" if reconf else
                 "FRAGILE" if gf["n_nodes"] < base_n else "GREW_OR_FRAGMENTED")
        pert_results[name] = {"final_n_nodes": gf["n_nodes"], "final_dist_mean": gf["dist_mean"],
                              "dist_return_err": d_ret, "klass": klass, "series": ser}
        print(f"  pert {name}: nodes {base_n}->{gf['n_nodes']} dist {base_d:.1f}->{gf['dist_mean']:.1f} "
              f"(err {d_ret:.2f}) -> {klass}", flush=True)
    n_restore = sum(1 for v in pert_results.values() if v["klass"] == "RESTORING")
    n_frag = sum(1 for v in pert_results.values() if v["klass"] in ("FRAGILE", "GREW_OR_FRAGMENTED"))
    verdict = ("BOUND_STATE_RESTORING" if n_restore >= 2 else
               "BOUND_STATE_FRAGILE" if n_frag >= 2 else
               "BOUND_STATE_RECONFIGURES" if any(v["klass"] == "RECONFIGURES" for v in pert_results.values()) else
               "BOUND_STATE_PASSIVE_COEXISTENCE")
    # residual circulation summary
    vt_final = [profs[k]["residual_v_t"] for k in profs]
    report = {"config": "feb56dc7 bare S-NCGL", "params": PAR, "N": N, "T_long": T_LONG,
              "result_id": "LONG_TIME_STABLE_4_NODE_BOUND_STATE",
              "final_geometry": geo_f, "corridors": cor_f, "node_profiles": profs,
              "residual_v_t_per_node": vt_final, "rotating_at_end": bool(np.mean(np.abs(vt_final)) > 0.02),
              "perturbation_tests": pert_results, "bond_verdict": verdict, "dynamics": dyn}
    outdir = os.path.join(ROOT, "sweep_runs", "SUBSTRATE_HUNT_20260621_161557", "feb56dc7_bound_state")
    os.makedirs(outdir, exist_ok=True)
    keep = np.linspace(0, nfin - 1, 12).astype(int)
    np.savez_compressed(os.path.join(outdir, "frames.npz"), psi=snaps[keep].astype(np.complex64), frames=keep)
    json.dump(report, open(os.path.join(outdir, "feb_bound_state.json"), "w"), indent=2, default=float)
    print(f"\nfinal: {geo_f['n_nodes']} nodes, pairwise dist {geo_f['dist_min']:.1f}-{geo_f['dist_max']:.1f} "
          f"(mean {geo_f['dist_mean']:.1f} std {geo_f['dist_std']:.1f}); residual v_t {np.mean(np.abs(vt_final)):.3f}")
    print(f"corridor conductances: {[round(c['conductance'],3) for c in cor_f]}")
    print(f"BOND VERDICT: {verdict}  ({(time.time()-t0)/60:.1f} min)\nwrote {outdir}")


if __name__ == "__main__":
    main()
