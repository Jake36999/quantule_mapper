"""
FMIA transfer / interaction-rate diagnostic  (PASSIVE, MONITORED — never solver feedback).

WHY THIS LAYER EXISTS
---------------------
The ablation panel ("remove one node, do the others destabilise?") tests *destructive*
mutual support. The clean doubly-gated hunt found that the bounded multi-node population
does NOT show destructive ablation-dependency (support_legs = 0 / 864). Per IRER theory
(Declaration of Intellectual Provenance v9), that is the WRONG test for the mechanism:

  * Concept 21 "Fields of Minimal Informational Action (FMIA)" — informational systems route
    along "Informational Parallels": coherent channels / manifoldic lines "characterized by
    low resistance or minimal informational dissipation/tension." "Gradient-Derived
    Informational Forces propagate along FMIA Informational Parallels." Overlapping
    trajectories form "Interference Channels" / "Informational Caustics."
  * Concept 22 "Informational Manifold Topology" — a "resonance-weighted metric tensor
    g_ij(RD,PAS) ... governs geodesics in Informational Parallels" (== our Omega^2 conformal
    metric / Delta_g).
  * Author's own proposed empirical test (provenance L2501): "represent rho peaks as nodes
    and their interactions as edges ... preferred pathways of influence ... how
    information/stress moves through the simulated substrate ... precursors to Informational
    Parallels."

So once stable nodes are set, support shows up as the *rate of interaction / transfer of
information* between them — energy transfer along paths of least resistance, phase/current
exchange, overlapping interference lattices — NOT as catastrophic failure on node removal.

This module operationalises that: time-resolved node tracking + a node-interaction graph
whose edges are measured transfer/coupling channels. It is a SCOUT diagnostic — it RE-SCORES
the existing stable population, it does NOT promote/reject, and it does NOT alter solver
physics. Omega^2 / curvature come from the single source of truth (gravity/unified_omega)
with param_skip_topology_cap=True, exactly like geometry_diag. Stamp the contract version
into provenance. Do not claim IRER evidence from the JAX scout alone; CuPy validates finalists.

Runs in the WSL2 jax venv (jax + numpy + scipy + gravity.unified_omega), same as
validate_candidates.py.
"""
import os
import sys
import numpy as np
import scipy.ndimage as ndi

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gravity"))
from jax_scout import physics                                   # noqa: E402
from unified_omega import derive_stable_conformal_factor        # noqa: E402  single source of truth

TRANSFER_DIAG_CONTRACT_VERSION = "IRER-FMIA-TRANSFER-DIAG-v2"
# v2: temporal coupling measured as EXCESS over a circular-shift surrogate null, on
#     residual fluctuations (node phase detrended by its own frequency ramp; node energy
#     residualised against the global breathing mode). v1 reported raw phase-lock / xcorr,
#     which are trivially ~1 for independent identical-frequency oscillators (global-mode
#     artifact). v2 reads ~0 for independent nodes and >0 only for genuine coupling.

# -- node detection --
NODE_SIGMA = 2.5            # density node threshold: rho > mean + NODE_SIGMA*std
MIN_NODE_VOXELS = 3         # ignore speckle components
TRACK_PERSIST_FRAC = 0.6   # a track must appear in >= this fraction of snapshots to count
# -- geometry corridor sampling --
PATH_SAMPLES = 24          # samples along an inter-node segment
BRIDGE_HALFWIDTH = 0.20    # bridge region = central +/-20% of the segment
SPEC_WIN = 12              # half-side of the local cube window for interference spectra
# -- transfer cross-correlation + surrogate null --
MAX_LAG_FRAC = 0.3         # max |lag| as fraction of #snapshots for lagged xcorr
N_SURR = 24                # circular-shift surrogates per pair for the null baseline
MIN_SHIFT_FRAC = 0.15      # surrogate shifts drawn from [MIN_SHIFT_FRAC*T, (1-...)*T]
N_INTERF_NULL = 8          # random-window baselines for interference-overlap excess
SURR_SEED = 20260620
# -- v2 classification thresholds, on the EXCESS-over-null scale --
# (re-scoring, not promotion). THR_PHASECOUP / THR_INTERF are set to the INDEPENDENCE
# FLOOR measured by transfer_null_control.py: feeding the pipeline node-phase / node-spectra
# from DIFFERENT, independent simulations (true coupling = 0) still yields phase_couple_excess
# up to p95 ~0.73, so any in-situ value below that is NOT separable from pipeline bias.
# Thresholds are set at those floors so a transfer class fires only above guaranteed-zero.
THR_JFLUX = 0.12           # mean |normalised directed bridge current|
THR_ALIGN = 0.40           # |J . axis| / |J| along bridge
THR_CONDUCT = 0.20         # density-bridge conductance (bottleneck rho / node rho)
THR_PHASECOUP = 0.73       # phase-coupling EXCESS -- set to cross-sim independence-floor p95
THR_ACTIONRATE = 0.15      # action-rate coupling EXCESS over surrogate null
THR_EXCHANGE = 0.15        # residual energy-exchange EXCESS over surrogate null
THR_INTERF = 0.15          # interference-lattice EXCESS (conservative; pending its own floor)
THR_ABL = 0.15             # destructive-ablation tag (carried from the hunt)
EDGE_COUPLE = 0.20         # graph edge present if blended coupling exceeds this


# ======================================================================
# 1. Strided real-space trajectory capture (jitted; closes over nothing host-side)
# ======================================================================
@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def _capture(pvec, psi0, N, L, dt, n_snap, stride, rd, cd):
    ops = physics._ops_from_vec(pvec, N, L, dt, rd, cd)
    psi_k = physics.initial_psi_k(psi0, ops)

    def inner(pk, _):
        return physics.step(pk, ops), None

    def outer(pk, _):
        pk, _ = lax.scan(inner, pk, None, length=stride)
        return pk, jnp.fft.ifftn(pk)

    psi_k_fin, snaps = lax.scan(outer, psi_k, None, length=n_snap)
    finite = jnp.all(jnp.isfinite(jnp.abs(snaps[-1])))
    return snaps, finite


def capture_trajectory(pvec, ic, N, L, dt, steps, n_snap):
    """Return (snaps[n_snap+1, N,N,N] complex128 incl. t=0, finite_bool)."""
    stride = max(1, steps // n_snap)
    snaps, finite = _capture(jnp.asarray(pvec), jnp.asarray(ic),
                             N, L, dt, n_snap, stride, jnp.float64, jnp.complex128)
    snaps = np.asarray(snaps)
    snaps = np.concatenate([np.asarray(ic)[None], snaps], axis=0)  # prepend IC
    return snaps, bool(finite)


# ======================================================================
# 2. Node detection (wrap-aware) + tracking
# ======================================================================
def _circ_centroid(coords, weights, N):
    """Wrap-aware weighted centroid along one axis (coords in [0,N))."""
    th = 2 * np.pi * coords / N
    c = np.average(np.cos(th), weights=weights)
    s = np.average(np.sin(th), weights=weights)
    ang = np.arctan2(s, c) % (2 * np.pi)
    return ang * N / (2 * np.pi)


def detect_nodes(psi, dx):
    """List of node dicts for one field: {centroid(vox), E, M, phase, size}."""
    rho = np.abs(psi) ** 2
    thr = rho.mean() + NODE_SIGMA * rho.std()
    mask = rho > thr
    lbl, nn = ndi.label(mask)
    nodes = []
    for i in range(1, nn + 1):
        sel = lbl == i
        sz = int(sel.sum())
        if sz < MIN_NODE_VOXELS:
            continue
        idx = np.array(np.nonzero(sel))          # (3, sz)
        w = rho[sel]
        cen = np.array([_circ_centroid(idx[a], w, rho.shape[a]) for a in range(3)])
        E = float(rho[sel].sum()) * dx ** 3        # node energy (integrated rho)
        M = float(np.sqrt(rho[sel]).sum()) * dx ** 3  # node "mass" (integrated amplitude)
        ph = float(np.angle(np.sum(psi[sel])))     # node mean phase
        nodes.append({"centroid": cen, "E": E, "M": M, "phase": ph, "size": sz})
    return nodes


def _pdist(a, b, N):
    """Periodic (minimal-image) Euclidean distance between voxel centroids."""
    d = np.abs(a - b)
    d = np.minimum(d, N - d)
    return float(np.sqrt((d ** 2).sum()))


def track_nodes(snap_nodes, N, gate_frac=0.25):
    """Greedy nearest-centroid tracking across snapshots.
    Returns persistent tracks: list of dicts with time series arrays (E,M,phase,centroid,present)."""
    T = len(snap_nodes)
    gate = gate_frac * N
    # seed tracks from the first snapshot that has nodes
    tracks = []  # each: {"cen":cen, "hist":{t:nodedict}}
    for t, nodes in enumerate(snap_nodes):
        used = set()
        # match existing tracks to this snapshot's nodes
        for tr in tracks:
            best, bd = -1, 1e9
            for k, nd in enumerate(nodes):
                if k in used:
                    continue
                d = _pdist(tr["cen"], nd["centroid"], N)
                if d < bd:
                    bd, best = d, k
            if best >= 0 and bd <= gate:
                nd = nodes[best]
                used.add(best)
                tr["hist"][t] = nd
                tr["cen"] = nd["centroid"]          # update prediction
        # unmatched nodes spawn new tracks
        for k, nd in enumerate(nodes):
            if k in used:
                continue
            tracks.append({"cen": nd["centroid"].copy(), "hist": {t: nd}})
    # keep persistent tracks; build dense time series (NaN where absent)
    out = []
    for tr in tracks:
        if len(tr["hist"]) < TRACK_PERSIST_FRAC * T:
            continue
        E = np.full(T, np.nan); M = np.full(T, np.nan); ph = np.full(T, np.nan)
        cen = np.full((T, 3), np.nan)
        for t, nd in tr["hist"].items():
            E[t] = nd["E"]; M[t] = nd["M"]; ph[t] = nd["phase"]; cen[t] = nd["centroid"]
        out.append({"E": E, "M": M, "phase": ph, "centroid": cen,
                    "present": ~np.isnan(E),
                    "cen_final": _last_valid(cen)})
    return out


def _last_valid(cen):
    for t in range(cen.shape[0] - 1, -1, -1):
        if not np.isnan(cen[t, 0]):
            return cen[t]
    return cen[0]


# ======================================================================
# 3. Temporal transfer metrics (per persistent node pair)
# ======================================================================
def _interp_nan(x):
    x = x.copy(); idx = np.arange(len(x)); good = ~np.isnan(x)
    if good.sum() < 2:
        return None
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def _lagged_xcorr(a, b, max_lag):
    """Return (best_pos_corr, best_pos_lag, most_neg_corr) over lags in [-max_lag,max_lag]."""
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0, 0, 0.0
    a /= na; b /= nb
    best_pos, best_lag, most_neg = -2.0, 0, 2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float(np.dot(a[lag:], b[:len(b) - lag])) if lag < len(a) else 0.0
        else:
            c = float(np.dot(a[:len(a) + lag], b[-lag:]))
        if c > best_pos:
            best_pos, best_lag = c, lag
        if c < most_neg:
            most_neg = c
    return best_pos, best_lag, most_neg


def _corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _regress_out(y, x):
    """Residual of y after least-squares regressing out x (both mean-removed)."""
    xc = x - x.mean(); yc = y - y.mean()
    if xc.std() == 0:
        return yc
    beta = float(np.dot(yc, xc) / np.dot(xc, xc))
    return yc - beta * xc


def prep_track(track, E_tot, dphi_glob):
    """Per-node detrended fluctuation series so coupling is measured ABOVE trivial
    common-mode behaviour:
      * energy residual = node energy with the GLOBAL breathing mode (E_tot) regressed out
        (co-settling of the whole field removed);
      * phase residual = node phase minus its own linear frequency ramp (identical-frequency
        oscillators -> constant offset -> zero residual), THEN with the global phase-wobble
        mode (dphi_glob) regressed out (a shared global driver is collective, not pairwise
        transfer -- this is the collective-vs-transfer cut the objective requires).
    Returns dict or None."""
    E = _interp_nan(track["E"]); phi = _interp_nan(track["phase"])
    if E is None or phi is None:
        return None
    Eres = _regress_out(E, E_tot)
    uph = np.unwrap(phi); idx = np.arange(len(uph))
    sl, inter = np.polyfit(idx, uph, 1)
    dphi = uph - (sl * idx + inter)
    dphi = _regress_out(dphi, dphi_glob)          # remove shared global phase mode
    return {"Eres": Eres, "dphi": dphi, "omega": float(sl)}


def _pair_raw(di, dj, max_lag):
    """Raw coupling of two prepped tracks (residual fluctuations only)."""
    bp, lag, mn = _lagged_xcorr(di["Eres"], dj["Eres"], max_lag)
    exch = max(0.0, -mn)                                  # anti-correlated residual energy = exchange
    cocorr = max(0.0, bp)                                 # co-fluctuating residual energy
    pcoup = abs(_corr(di["dphi"], dj["dphi"]))            # phase-fluctuation coupling
    wi, wj = np.diff(di["dphi"]), np.diff(dj["dphi"])     # action-rate (freq) fluctuations
    arc = abs(_corr(wi, wj))
    return {"E_exchange": exch, "E_cocorr": cocorr, "E_lag": lag,
            "phase_couple": pcoup, "action_rate": arc}


def temporal_pair_metrics(di, dj, n_snap, rng):
    """Coupling EXCESS over a circular-shift surrogate null. Independent identical
    oscillators -> excess ~ 0; genuine coupling -> excess > 0."""
    if di is None or dj is None:
        return None
    max_lag = max(1, int(MAX_LAG_FRAC * n_snap))
    sig = _pair_raw(di, dj, max_lag)
    T = len(di["Eres"]); lo = max(1, int(MIN_SHIFT_FRAC * T)); hi = T - lo
    null = {"E_exchange": [], "phase_couple": [], "action_rate": []}
    if hi > lo:
        for _ in range(N_SURR):
            k = int(rng.integers(lo, hi + 1))
            dj2 = {"Eres": np.roll(dj["Eres"], k), "dphi": np.roll(dj["dphi"], k)}
            s = _pair_raw(di, dj2, max_lag)
            for kk in null:
                null[kk].append(s[kk])
    out = dict(sig)
    for kk in null:
        nm = float(np.mean(null[kk])) if null[kk] else 0.0
        out[kk + "_excess"] = max(0.0, sig[kk] - nm)
        out[kk + "_null"] = nm
    return out


# ======================================================================
# 4. Geometric corridor metrics (final field; FMIA Informational Parallels)
# ======================================================================
def _grad(f, dx):
    return ((np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * dx),
            (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * dx),
            (np.roll(f, -1, 2) - np.roll(f, 1, 2)) / (2 * dx))


def _lap(f, dx):
    return sum((np.roll(f, -1, a) - 2 * f + np.roll(f, 1, a)) / dx ** 2 for a in range(3))


def geometry_fields(psi, params, dx):
    """Omega^2, curvature proxy R, and J_info=rho*grad(phase) on the final field."""
    rho = np.abs(psi) ** 2
    rho_safe = np.maximum(rho, 1e-7)
    geo = dict(params); geo["param_skip_topology_cap"] = True
    omega_sq = np.asarray(derive_stable_conformal_factor(rho_safe, geo), dtype=np.float64)
    lnOm = 0.5 * np.log(np.maximum(omega_sq, 1e-30))
    glx, gly, glz = _grad(lnOm, dx)
    R = -(2.0 / np.maximum(omega_sq, 1e-30)) * (_lap(lnOm, dx) + 0.5 * (glx**2 + gly**2 + glz**2))
    px, py, pz = _grad(psi, dx)
    Jx = np.imag(np.conj(psi) * px); Jy = np.imag(np.conj(psi) * py); Jz = np.imag(np.conj(psi) * pz)
    return {"rho": rho, "omega_sq": omega_sq, "R": R, "J": (Jx, Jy, Jz)}


def _sample_line(field, c0, c1, N, nsamp):
    """Sample a periodic field along the minimal-image segment c0->c1 (nearest voxel)."""
    d = c1 - c0
    d = d - N * np.round(d / N)          # minimal image displacement
    ts = np.linspace(0, 1, nsamp)
    pts = (c0[None, :] + ts[:, None] * d[None, :]) % N
    iv = np.round(pts).astype(int) % N
    return field[iv[:, 0], iv[:, 1], iv[:, 2]], d


def corridor_pair_metrics(geo, ci, cj, N, dx):
    """FMIA corridor: density-bridge conductance, Omega^2 smoothness, J_info flux + alignment."""
    rho = geo["rho"]; omega_sq = geo["omega_sq"]
    Jx, Jy, Jz = geo["J"]
    Jmag = np.sqrt(Jx**2 + Jy**2 + Jz**2)
    # node reference density
    rho_i = rho[tuple(np.round(ci).astype(int) % N)]
    rho_j = rho[tuple(np.round(cj).astype(int) % N)]
    rho_ref = max(1e-30, 0.5 * (rho_i + rho_j))
    rho_path, disp = _sample_line(rho, ci, cj, N, PATH_SAMPLES)
    om_path, _ = _sample_line(omega_sq, ci, cj, N, PATH_SAMPLES)
    # exclude the node interiors (endpoints) from the bottleneck search
    inner = rho_path[2:-2] if len(rho_path) > 4 else rho_path
    conductance = float(inner.min() / rho_ref)            # density bridge (no null gap == ~>0.2)
    log_om = np.log(np.maximum(om_path, 1e-30))
    omega_smooth = float(1.0 / (1.0 + log_om.std()))      # smooth metric corridor
    # directed current through the bridge midpoint region, projected on node-node axis
    u = disp / (np.linalg.norm(disp) + 1e-30)
    ts = np.linspace(0, 1, PATH_SAMPLES)
    mid = (np.abs(ts - 0.5) <= BRIDGE_HALFWIDTH)
    pts = (ci[None, :] + ts[:, None] * disp[None, :]) % N
    iv = np.round(pts).astype(int) % N
    Ju = (Jx[iv[:, 0], iv[:, 1], iv[:, 2]] * u[0] +
          Jy[iv[:, 0], iv[:, 1], iv[:, 2]] * u[1] +
          Jz[iv[:, 0], iv[:, 1], iv[:, 2]] * u[2])
    Jm = Jmag[iv[:, 0], iv[:, 1], iv[:, 2]]
    Jref = np.sqrt(np.mean(Jmag ** 2)) + 1e-30
    flux = float(np.mean(Ju[mid]) / Jref)                 # signed normalised directed current
    align = float(np.mean(np.abs(Ju[mid]) / (Jm[mid] + 1e-30)))  # |J.axis|/|J| on the bridge
    return {"conductance": conductance, "omega_smooth": omega_smooth,
            "J_flux": flux, "path_align": align}


def _local_spectrum(psi, c, N, w):
    """Power spectrum of a wrap-extracted cube window centred on c."""
    c = np.round(c).astype(int) % N
    sl = [np.arange(c[a] - w, c[a] + w) % N for a in range(3)]
    win = psi[np.ix_(sl[0], sl[1], sl[2])]
    P = np.abs(np.fft.fftn(win)) ** 2
    P.flat[0] = 0.0                                       # drop DC
    return P.ravel()


def _bhatt(psi, ca, cb, N, w):
    Pa = _local_spectrum(psi, ca, N, w); Pb = _local_spectrum(psi, cb, N, w)
    sa, sb = Pa.sum(), Pb.sum()
    if sa <= 0 or sb <= 0:
        return 0.0
    return float(np.sum(np.sqrt((Pa / sa) * (Pb / sb))))


def interference_overlap(psi, ci, cj, N, rng, w=SPEC_WIN):
    """Shared interference-lattice EXCESS: Bhattacharyya overlap of the two nodes' local
    power spectra, minus the baseline overlap of node i with random (non-node) windows.
    Two generic blobs share low-k spectral mass trivially; excess isolates genuinely
    shared structure (overlapping FMIA Interference Channels). Returns (excess, raw)."""
    raw = _bhatt(psi, ci, cj, N, w)
    nulls = [_bhatt(psi, ci, rng.integers(0, N, 3).astype(float), N, w)
             for _ in range(N_INTERF_NULL)]
    base = float(np.mean(nulls)) if nulls else 0.0
    return max(0.0, raw - base), raw


# ======================================================================
# 5. Full per-candidate analysis -> node-interaction graph + classification
# ======================================================================
def analyze_candidate(pvec, params, ic, N, L, dt, steps, n_snap, bounded_abl_sens=0.0,
                      iso_surv=np.nan):
    dx = L / N
    snaps, finite = capture_trajectory(pvec, ic, N, L, dt, steps, n_snap)
    amp_final = float(np.max(np.abs(snaps[-1]))) if finite else float("inf")
    result = {"contract": TRANSFER_DIAG_CONTRACT_VERSION, "finite": finite,
              "amp_final": amp_final, "iso_surv": iso_surv,
              "bounded_abl_sens": bounded_abl_sens}

    if (not finite) or amp_final > 1e3:
        result.update({"n_persistent_nodes": 0, "klass": "unstable_reject"})
        return result

    snap_nodes = [detect_nodes(s, dx) for s in snaps]
    tracks = track_nodes(snap_nodes, N)
    nP = len(tracks)
    result["n_persistent_nodes"] = nP
    result["nodes_per_snap_med"] = float(np.median([len(s) for s in snap_nodes]))

    if nP < 2:
        result["klass"] = "independent_static_condensates" if nP == 1 else "unstable_reject"
        result.update(_empty_graph())
        return result

    geo = geometry_fields(snaps[-1], params, dx)
    Tn = len(snaps)
    rng = np.random.default_rng(SURR_SEED)
    E_tot = np.array([float(np.sum(np.abs(s) ** 2)) for s in snaps])   # global breathing mode
    # global phase-wobble mode: energy-weighted whole-field phase, detrended by its ramp
    gph = np.array([float(np.angle(np.sum(s * np.abs(s) ** 2))) for s in snaps])
    gph = np.unwrap(gph); gidx = np.arange(len(gph))
    gsl, gint = np.polyfit(gidx, gph, 1)
    dphi_glob = gph - (gsl * gidx + gint)
    prepped = [prep_track(t, E_tot, dphi_glob) for t in tracks]
    pairs = []
    for i in range(nP):
        for j in range(i + 1, nP):
            tm = temporal_pair_metrics(prepped[i], prepped[j], Tn, rng)
            if tm is None:
                continue
            cm = corridor_pair_metrics(geo, tracks[i]["cen_final"], tracks[j]["cen_final"], N, dx)
            io_ex, io_raw = interference_overlap(snaps[-1], tracks[i]["cen_final"],
                                                 tracks[j]["cen_final"], N, rng)
            # blended coupling (excess channels only) for graph-edge presence
            couple = float(np.mean([
                abs(cm["J_flux"]), tm["E_exchange_excess"], tm["phase_couple_excess"],
                cm["path_align"] * cm["conductance"], io_ex,
            ]))
            pairs.append({**tm, **cm, "interference_excess": io_ex,
                          "interference_raw": io_raw, "couple": couple})

    result.update(_aggregate_graph(pairs, nP))
    result["klass"] = _classify(result)
    return result


def _empty_graph():
    keys = ["interaction_graph_density", "mean_transfer_strength", "max_transfer_strength",
            "energy_exchange_index", "phase_coupling_score", "geometric_path_alignment",
            "omega_corridor_conductance", "interference_lattice_overlap", "action_rate_coherence",
            "raw_phase_lock", "raw_E_xcorr", "raw_interference"]
    return {k: 0.0 for k in keys}


def _aggregate_graph(pairs, nP):
    if not pairs:
        return _empty_graph()
    def m(k): return float(np.mean([p[k] for p in pairs]))
    def mx(k): return float(np.max([p[k] for p in pairs]))
    n_edges = sum(1 for p in pairs if p["couple"] >= EDGE_COUPLE)
    poss = nP * (nP - 1) / 2
    jflux = [abs(p["J_flux"]) for p in pairs]
    return {
        "interaction_graph_density": float(n_edges / poss) if poss else 0.0,
        "mean_transfer_strength": float(np.mean(jflux)),
        "max_transfer_strength": float(np.max(jflux)),
        "energy_exchange_index": m("E_exchange_excess"),       # null-referenced
        "phase_coupling_score": m("phase_couple_excess"),      # null-referenced
        "geometric_path_alignment": m("path_align"),
        "omega_corridor_conductance": m("conductance"),
        "interference_lattice_overlap": m("interference_excess"),  # null-referenced
        "action_rate_coherence": m("action_rate_excess"),      # null-referenced
        # raw (global-mode-inclusive) values kept for transparency only:
        "raw_phase_lock": m("phase_couple"),
        "raw_E_xcorr": mx("E_cocorr"),
        "raw_interference": m("interference_raw"),
    }


def _classify(r):
    """Precedence re-scoring (NOT promotion/rejection). v2 thresholds on the excess scale."""
    geo_transfer = (r["mean_transfer_strength"] > THR_JFLUX and
                    r["geometric_path_alignment"] > THR_ALIGN and
                    r["omega_corridor_conductance"] > THR_CONDUCT)
    phase_current = (r["phase_coupling_score"] > THR_PHASECOUP and
                     (r["action_rate_coherence"] > THR_ACTIONRATE or
                      r["energy_exchange_index"] > THR_EXCHANGE))
    interference = (r["interference_lattice_overlap"] > THR_INTERF and
                    r["interaction_graph_density"] > 0.0)
    if r["bounded_abl_sens"] > THR_ABL:
        return "destructive_ablation_supported"
    if geo_transfer:
        return "geometric_transfer_candidate"
    if phase_current:
        return "phase_current_transfer_candidate"
    if interference:
        return "interference_lattice_candidate"
    if not np.isnan(r["iso_surv"]) and r["iso_surv"] < 0.5:
        return "collective_density_threshold"
    return "independent_static_condensates"
