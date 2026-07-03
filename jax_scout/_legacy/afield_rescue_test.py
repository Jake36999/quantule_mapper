"""
A_SCALAR / VACUUM_REFERENCE RESCUE TEST  (honestly labeled — see the audit).

AUDIT FINDING: the existing A-field prototype (afield_prototype.py) is SCALAR and
DENSITY-sourced: A obeys d^2A/dt^2 = -c^2 k^2 A + rho_k (sourced by rho=|psi|^2, a scalar),
and couples back by modulating a SCALAR geometric quantity isotropically
(vacuum_ref: rho_vac_eff = rho_vac + gamma_A*A; additive: Omega^2_eff = Omega^2*exp(gamma_A*A)).
It does NOT couple to the informational current J_info = rho*grad(phi). Therefore this is a test
of SCALAR A-field rescue, NOT a test of current-coupled "rate of interaction" / FMIA wires.
Per the theory (current carries direction; scalar density does not), this is EXPECTED to be able
to modulate the web's timing/amplitude but NOT to create directed selective routing. We run it as
the disciplined empirical control before implementing the current-coupled branch.

Falsification question: does A-on keep the 800-step phase-coupling signal from collapsing below
the 0.73 floor by 1600 steps (with bounded geometry, no runaway, no A DC/k=0 runaway)?

Sweep gamma_A in {0, 0.01, 0.02, 0.05, 0.1, 0.2} (vacuum_ref) on frozen gen18/gen14/gen34.

Classifications: A_RESCUED_TRANSFER_CANDIDATE / A_STABILIZED_WEB_ONLY / A_SCALAR_NO_RESCUE /
A_CURRENT_CHANNEL_REQUIRED / A_RUNAWAY_REJECT / A_ARTIFACT_REJECT.

CAUTION: JAX scout-level; not proof; gamma_A>0 is the ACTIVE experimental branch (contract key
IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1), NOT rank-compatible with gamma_A=0, NOT for CuPy/Hunter.
WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/afield_rescue_test.py
"""
import os, sys, json, glob, time
from functools import partial
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td, afield_prototype as afp, geometry_diag as gd

L, dt = 10.0, 0.005
order = physics.SWEEP_PARAM_ORDER
BASE_SEED = 20260619
STEPS, NSNAP = 1600, 40
FLOOR = td.THR_PHASECOUP                  # 0.73
GAMMAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
TOPOLOGY = "vacuum_ref"


def multiseed_ic(N, seed, K=6):
    rng = np.random.default_rng(seed)
    x = np.linspace(-L/2, L/2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L/12.0; psi = np.zeros((N, N, N), np.complex128)
    for _ in range(K):
        cx, cy, cz = rng.uniform(-L/2, L/2, 3)
        psi += np.exp(-((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/(2*w**2))
    noise = 0.01*(rng.standard_normal((N, N, N))+1j*rng.standard_normal((N, N, N)))
    return (psi+noise).astype(np.complex128)


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9, 10))
def _capture_afield(pvec, psi0, gamma_A, N, L_, dt_, n_snap, stride, rd, cd, topology):
    """Strided real-space capture of the A-coupled trajectory (+ A snapshots)."""
    ops = physics._ops_from_vec(pvec, N, L_, dt_, rd, cd)
    psi_k = jnp.fft.fftn(psi0) * ops.dealias_mask
    A_k = jnp.zeros((N, N, N), cd); A_dot_k = jnp.zeros((N, N, N), cd)

    def inner(carry, _):
        psi_k, A_k, A_dot_k = carry
        A_k, A_dot_k, A_real = afp._update_afield(A_k, A_dot_k, psi_k, ops, dt_)
        if topology == "additive_potential":
            mod = jnp.exp(jnp.clip(gamma_A * A_real, -30.0, 30.0)); psi_k = physics.step(psi_k, ops, None, mod)
        else:
            mod = jnp.maximum(ops.rho_vac + gamma_A * A_real, afp.RHO_VAC_EFF_FLOOR); psi_k = physics.step(psi_k, ops, mod, None)
        return (psi_k, A_k, A_dot_k), None

    def outer(carry, _):
        carry, _ = lax.scan(inner, carry, None, length=stride)
        psi_k, A_k, _ = carry
        return carry, (jnp.fft.ifftn(psi_k), jnp.real(jnp.fft.ifftn(A_k)))

    carry, (snaps, Asnaps) = lax.scan(outer, (psi_k, A_k, A_dot_k), None, length=n_snap)
    finite = jnp.all(jnp.isfinite(jnp.abs(snaps[-1]))) & jnp.all(jnp.isfinite(Asnaps[-1]))
    return snaps, Asnaps, finite


def capture_afield(par, ic, gamma_A, N, steps, n_snap, topology=TOPOLOGY):
    stride = max(1, steps // n_snap)
    pv = jnp.asarray([par[k] for k in order])
    snaps, Asnaps, fin = _capture_afield(pv, jnp.asarray(ic), float(gamma_A), N, L, dt,
                                         n_snap, stride, jnp.float64, jnp.complex128, topology)
    snaps = np.concatenate([np.asarray(ic)[None], np.asarray(snaps)], axis=0)
    Asnaps = np.asarray(Asnaps)
    Asnaps = np.concatenate([np.zeros_like(Asnaps[:1]), Asnaps], axis=0)   # A(t=0)=0
    return snaps, Asnaps, bool(fin)


def phase_coupling_of(snaps, par, N):
    """Replicate analyze_candidate's null-referenced phase-coupling + exchange on given snaps."""
    dx = L / N
    snap_nodes = [td.detect_nodes(s, dx) for s in snaps]
    tracks = td.track_nodes(snap_nodes, N)
    if len(tracks) < 2:
        return {"n_persistent_nodes": len(tracks), "phase_coupling_score": 0.0,
                "energy_exchange_index": 0.0}
    Tn = len(snaps); rng = np.random.default_rng(td.SURR_SEED)
    E_tot = np.array([float(np.sum(np.abs(s) ** 2)) for s in snaps])
    gph = np.unwrap(np.array([float(np.angle(np.sum(s * np.abs(s) ** 2))) for s in snaps]))
    gi = np.arange(len(gph)); gsl, gint = np.polyfit(gi, gph, 1); dphi_glob = gph - (gsl*gi+gint)
    prepped = [td.prep_track(t, E_tot, dphi_glob) for t in tracks]
    pcs, exs = [], []
    for i in range(len(prepped)):
        for j in range(i+1, len(prepped)):
            tm = td.temporal_pair_metrics(prepped[i], prepped[j], Tn, rng)
            if tm:
                pcs.append(tm["phase_couple_excess"]); exs.append(tm["E_exchange_excess"])
    return {"n_persistent_nodes": len(tracks),
            "phase_coupling_score": float(np.mean(pcs)) if pcs else 0.0,
            "energy_exchange_index": float(np.mean(exs)) if exs else 0.0}


def bridge_of(psi_final, par, N):
    dx = L / N; nodes = td.detect_nodes(psi_final, dx)
    if len(nodes) < 2:
        return 0.0
    geo = td.geometry_fields(psi_final, par, dx); conds = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            conds.append(td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"])
    return float(np.max(conds)) if conds else 0.0


def classify(g0_pc, pc_full, er, curv, finite, amp, a_runaway):
    if not finite or amp > 1e3 or a_runaway:
        return "A_RUNAWAY_REJECT"
    if not (0.1 <= er <= 5.0) or curv >= 1.0:
        return "A_ARTIFACT_REJECT"
    if pc_full > FLOOR:
        return "A_RESCUED_TRANSFER_CANDIDATE"
    return "A_SCALAR_NO_RESCUE"


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "BRIDGE_HUNT_2026*")))[-1]
    fz = json.load(open(os.path.join(d, "frozen_finalists.json")))
    print(f"A_SCALAR_RESCUE_TEST (topology={TOPOLOGY}, density-sourced scalar A) — top 3 finalists")
    print(f"AUDIT: existing A is scalar/density-sourced, NOT current-coupled. Falsify: does A-on keep")
    print(f"phase coupling > {FLOOR} floor at {STEPS} steps?  gamma_A sweep {GAMMAS}\n")
    report = []
    for fr in fz["finalists"][:3]:
        par = {k: float(fr["params"][k]) for k in order}
        label = f"gen{fr['generation']}_{fr['config_hash']}"
        ic = multiseed_ic(48, BASE_SEED); print(f"[{label}]")
        for g in GAMMAS:
            t0 = time.time()
            snaps, Asnaps, fin = capture_afield(par, ic, g, 48, STEPS, NSNAP)
            amp = float(np.max(np.abs(snaps[-1]))) if fin else float("inf")
            er = float(np.sum(np.abs(snaps[-1])**2)/(np.sum(np.abs(snaps[0])**2)+1e-30))
            curv = gd.curvature_max_only(snaps[-1], par, L/48) if fin else float("inf")
            pc = phase_coupling_of(snaps, par, 48) if fin else {"phase_coupling_score": 0.0, "n_persistent_nodes": 0, "energy_exchange_index": 0.0}
            bridge = bridge_of(snaps[-1], par, 48) if fin else 0.0
            A_energy = float(np.sum(Asnaps[-1]**2)) if fin else float("nan")
            A_max = float(np.max(np.abs(Asnaps[-1]))) if fin else float("nan")
            a_runaway = (not np.isfinite(A_energy)) or A_max > 1e6
            kl = classify(None, pc["phase_coupling_score"], er, curv, fin, amp, a_runaway)
            print(f"   gA={g:<5} fin={int(bool(fin))} nP={pc['n_persistent_nodes']} "
                  f"pcoup@1600={pc['phase_coupling_score']:.3f}{'  >floor' if pc['phase_coupling_score']>FLOOR else ''} "
                  f"exch={pc['energy_exchange_index']:.3f} bridge={bridge:.3f} er={er:.2f} curv={curv:.2f} "
                  f"A_E={A_energy:.2g} -> {kl}  ({time.time()-t0:.0f}s)")
            report.append({"label": label, "gamma_A": g, "topology": TOPOLOGY, "finite": bool(fin),
                           "n_persistent_nodes": pc["n_persistent_nodes"], "phase_coupling_1600": pc["phase_coupling_score"],
                           "energy_exchange": pc["energy_exchange_index"], "bridge_maxCond": bridge,
                           "er": er, "curv": curv, "A_energy": A_energy, "A_max": A_max, "klass": kl})
        print()
    od = os.path.join(d, "afield_scalar_rescue.json")
    json.dump(report, open(od, "w"), indent=2, default=float)
    rescued = sum(1 for r in report if r["klass"] == "A_RESCUED_TRANSFER_CANDIDATE")
    print(f"=== {rescued}/{len(report)} (finalist x gamma_A) show A_RESCUED_TRANSFER_CANDIDATE ===")
    print("If 0 with gamma_A>0: scalar density-sourced A does NOT rescue long-lived transfer "
          "-> next required branch = CURRENT_COUPLED_A_FIELD (couple to J_info=rho*grad(phi)). "
          "See docs/AFIELD_CURRENT_COUPLED_RFC.md")
    print(f"wrote {od}")


if __name__ == "__main__":
    main()
