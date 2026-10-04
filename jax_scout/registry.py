"""Component registry for experiment specs (IMPLEMENTATION_PLAN_2026-10 Phase E1).

A spec (irer_specs) names three kinds of component; this module maps the names to code. Every entry
WRAPS existing, gated code rather than re-implementing it: the steppers are the ones the order gates
cover (tests/test_stepper_order_jax.py), the ICs and observers are the harnesses' own helpers.

  SUBSTRATE  stepper + operator. build(grid, dt, params) -> Sim with .init(fields), .advance(n),
             .fields() -> {name: real-space array}, and .t
  IC         initial condition. ic(sim, **params) -> {field: array}; handed to sim.init
  OBSERVER   pure read. obs(sim, **params) -> {key: float}

Names are versioned: a change that alters numerical output must bump the version, so a spec that
pinned `version: 1` can tell it is no longer getting the same component.
"""
from __future__ import annotations

import os
from functools import partial

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

SUBSTRATES, ICS, OBSERVERS = {}, {}, {}


def _register(table, name, version, doc, params):
    def deco(fn):
        table[name] = {"name": name, "version": version, "fn": fn, "doc": doc.strip(), "params": params}
        return fn
    return deco


substrate = partial(_register, SUBSTRATES)
ic = partial(_register, ICS)
observer = partial(_register, OBSERVERS)


# =============================================================== substrates

class _Grid:
    def __init__(self, N, L):
        self.N, self.L = int(N), float(L)
        self.dx = self.L / self.N
        self.dV = self.dx ** 3
        x = np.linspace(-self.L / 2, self.L / 2, self.N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing="ij")


@substrate("etdrk4-sncgl", 1, """
Phase C / C1 / C2 S-NCGL on the fixed ETDRK4 (jax_scout/physics.py, e270cdc). params = the physics
param dict (param_D, param_eta, param_a, ... ; kinetic_mode='conservative' + param_geom_off for the C2
NLS branch). Fields: psi.""", {"type": "object"})
def _etdrk4(grid, dt, params):
    from jax_scout import physics

    class Sim:
        kind = "etdrk4-sncgl"

        def __init__(self):
            self.grid, self.dt, self.params, self.t = grid, dt, dict(params), 0.0
            self.ops = physics.build_operators(grid.N, grid.L, dt, self.params)
            self._adv = jax.jit(lambda pk, n: jax.lax.fori_loop(0, n, lambda i, p: physics.step(p, self.ops), pk),
                                static_argnums=1)

        def init(self, fields):
            self.psi_k = physics.initial_psi_k(jnp.asarray(fields["psi"], dtype=jnp.complex128), self.ops)

        def advance(self, n):
            self.psi_k = self._adv(self.psi_k, int(n))
            self.t += n * self.dt

        def fields(self):
            return {"psi": np.asarray(jnp.fft.ifftn(self.psi_k))}
    return Sim()


@substrate("kg-strang", 1, """
C3 complex nonlinear Klein-Gordon, Strang split with exact linear rotation
(jax_scout/phase_d_c3_wave.py:kg_evolve). params: c, m, a, s, f. Fields: psi, pi.""",
           {"type": "object", "required": ["c", "m", "a", "s", "f"]})
def _kg(grid, dt, params):
    from jax_scout.phase_d_c3_wave import build_kg, kg_evolve

    class Sim:
        kind = "kg-strang"

        def __init__(self):
            self.grid, self.dt, self.params, self.t = grid, dt, dict(params), 0.0
            p = self.params
            self.op = build_kg(grid.N, grid.L, p["c"], p["m"], dt)

        def init(self, fields):
            m = self.op["mask"]
            self.psi_k = jnp.fft.fftn(jnp.asarray(fields["psi"], dtype=jnp.complex128)) * m
            pi = fields.get("pi")
            self.pi_k = (jnp.fft.fftn(jnp.asarray(pi, dtype=jnp.complex128)) * m if pi is not None
                         else jnp.zeros_like(self.psi_k))

        def advance(self, n):
            p = self.params
            self.psi_k, self.pi_k = kg_evolve(self.psi_k, self.pi_k, self.op, p["a"], p["s"], p["f"], int(n))
            self.t += n * self.dt

        def fields(self):
            return {"psi": np.asarray(jnp.fft.ifftn(self.psi_k)), "pi": np.asarray(jnp.fft.ifftn(self.pi_k))}
    return Sim()


@substrate("tg-rk4", 1, """
TG dual substrate: KG field + temporal T + geometric G, RK4
(jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py:rk4_step). params: the TG cfg keys (c, m, a, s, f,
alpha_T, omega_T, omega_G, gamma_T, gamma_G, kappa_TG, epsilon_G, cT, cG, absorb_width,
absorb_strength, core_radius) plus optional flags [source, temporal, geometric, feedback] and refs.
Fields: phi, pi, T, G.""", {"type": "object"})
def _tg(grid, dt, params):
    from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s
    from jax_scout.phase_d_c3_wave import build_kg

    class Sim:
        kind = "tg-rk4"

        def __init__(self):
            self.grid, self.dt, self.params, self.t = grid, dt, dict(params), 0.0
            p = self.params
            cfg = {k: v for k, v in p.items() if k not in ("flags", "refs")}
            cfg.update(dt=dt, L=grid.L)
            self.cfgv = b1s.cfg_array(cfg)
            self.g = b1s.make_grid(build_kg(grid.N, grid.L, p["c"], p["m"], dt))
            self.refs = b1s.refs_array(p.get("refs") or {"reference_energy_max": 1.0, "reference_charge_max": 1.0,
                                                          "source_global_norm_S0": 1.0})
            self.flags = jnp.asarray(p.get("flags", [1.0, 1.0, 1.0, 1.0]), dtype=jnp.float64)
            self._b1s = b1s

        def init(self, fields):
            phi = jnp.asarray(fields["psi"], dtype=jnp.complex128)
            pi = jnp.asarray(fields.get("pi", jnp.zeros_like(phi)), dtype=jnp.complex128)
            z = jnp.zeros(phi.shape, dtype=jnp.float64)
            self.state = (phi, pi, z, z, z, z)

        def advance(self, n):
            self.state = self._b1s.evolve_n(self.state, self.cfgv, self.refs, self.g, self.flags, int(n))
            self.t += n * self.dt

        def fields(self):
            phi, pi, T, _, G, _ = self.state
            return {"psi": np.asarray(phi), "pi": np.asarray(pi), "T": np.asarray(T), "G": np.asarray(G)}
    return Sim()


# =============================================================== initial conditions

@ic("gaussian", 1, """A·exp(-|x-c|²/2σ²)·exp(i k·x); optional pi = -i·omega·psi (KG/TG).""", {"type": "object"})
def _gaussian(sim, A=1.0, sigma=1.0, center=(0.0, 0.0, 0.0), k=(0.0, 0.0, 0.0), omega=None):
    g = sim.grid
    r2 = (g.X - center[0]) ** 2 + (g.Y - center[1]) ** 2 + (g.Z - center[2]) ** 2
    psi = (A * np.exp(-r2 / (2 * sigma ** 2)) * np.exp(1j * (k[0] * g.X + k[1] * g.Y + k[2] * g.Z))).astype(np.complex128)
    out = {"psi": psi}
    if omega is not None:
        out["pi"] = -1j * omega * psi
    return out


@ic("multiseed", 1, """Phase C K-blob IC (core_saturation_search.build_ic, the a* harnesses' own).
Requires grid L = 10 (the harness constant).""", {"type": "object"})
def _multiseed(sim, K=6, seed=20260619, ic_norm="per_blob_fixed"):
    from jax_scout import core_saturation_search as css
    if abs(sim.grid.L - 10.0) > 1e-12:
        raise ValueError("multiseed IC is defined on L=10 (afield_current_coupled.L)")
    psi0, _ = css.build_ic(sim.grid.N, int(K), seed=int(seed), ic_norm=ic_norm)
    return {"psi": np.asarray(psi0)}


@ic("qball", 1, """Stationary Q-ball by Petviashvili (phase_d_c3_wave.qball_petviashvili) for the
substrate's (a, s, f); KG convention: omega = sqrt(m² - mu), pi = -i·omega·psi; optional configurational
kick v along x: pi -= v·∂x psi.""", {"type": "object", "required": ["mu"]})
def _qball(sim, mu, A=1.0, sig=1.2, kick_v=0.0):
    from jax_scout.phase_d_c3_wave import build_kg, qball_petviashvili
    p = sim.params
    op = build_kg(sim.grid.N, sim.grid.L, p.get("c", 1.0), p.get("m", 1.0), sim.dt)
    phi, info = qball_petviashvili(op, p["a"], p["s"], p["f"], float(mu), A=A, sig=sig)
    if phi is None:
        raise RuntimeError("Q-ball solve failed: %s" % info)
    psi = np.asarray(phi, dtype=np.complex128)
    omega = float(np.sqrt(max(p.get("m", 1.0) ** 2 - float(mu), 0.0)))
    pi = -1j * omega * psi
    if kick_v:
        kx = 2 * np.pi * np.fft.fftfreq(sim.grid.N, d=sim.grid.dx)
        dx_psi = np.fft.ifftn(1j * kx[:, None, None] * np.fft.fftn(psi))
        pi = pi - kick_v * dx_psi
    return {"psi": psi, "pi": pi}


@ic("load_npz", 1, """Fields from a saved .npz (e.g. a settled a* state). params: path, key (default psi_fin).""",
    {"type": "object", "required": ["path"]})
def _load_npz(sim, path, key="psi_fin"):
    return {"psi": np.load(path)[key].astype(np.complex128)}


# =============================================================== observers

@observer("mass", 1, """mass = ∫|psi|² dV; mass_ratio = mass / mass at t=0.""", {"type": "object"})
def _mass(sim, field="psi"):
    m = float(np.sum(np.abs(sim.fields()[field]) ** 2) * sim.grid.dV)
    sim.__dict__.setdefault("_mass0", m)
    return {"mass": m, "mass_ratio": m / sim._mass0 if sim._mass0 else float("nan")}


@observer("amp", 1, """amp = max|psi|.""", {"type": "object"})
def _amp(sim, field="psi"):
    return {"amp": float(np.max(np.abs(sim.fields()[field])))}


@observer("centroid", 1, """Periodic (circular-mean) centroid of |psi|² along x, y, z
(phase_d_c3_wave.centroid_x); unwrapped across the box so velocities can be fitted.""", {"type": "object"})
def _centroid(sim, field="psi"):
    from jax_scout.phase_d_c3_wave import centroid_x
    rho = np.abs(sim.fields()[field]) ** 2
    g = sim.grid
    raw = {"x": centroid_x(rho, g.N, g.L),
           "y": centroid_x(np.transpose(rho, (1, 0, 2)), g.N, g.L),
           "z": centroid_x(np.transpose(rho, (2, 1, 0)), g.N, g.L)}
    prev = sim.__dict__.setdefault("_centroid_prev", dict(raw))
    off = sim.__dict__.setdefault("_centroid_off", {"x": 0.0, "y": 0.0, "z": 0.0})
    for a in raw:
        d = raw[a] - prev[a]
        if d > g.L / 2:
            off[a] -= g.L
        elif d < -g.L / 2:
            off[a] += g.L
        prev[a] = raw[a]
    return {a: raw[a] + off[a] for a in raw}


@observer("nodes", 1, """Number of nodes (transfer_diag.detect_nodes: rho > mean + NODE_SIGMA·std, labelled).""",
          {"type": "object"})
def _nodes(sim, field="psi"):
    from jax_scout import transfer_diag as td
    return {"n_nodes": float(len(td.detect_nodes(sim.fields()[field], sim.grid.dx)))}


@observer("kg_invariants", 1, """KG energy E, U(1) charge Q, momentum Px, mass, amp
(phase_d_c3_wave.invariants). kg-strang only. Also Q_rel_drift / E_rel_drift vs t=0.""", {"type": "object"})
def _kg_inv(sim):
    from jax_scout.phase_d_c3_wave import invariants
    if sim.kind != "kg-strang":
        raise ValueError("kg_invariants needs the kg-strang substrate")
    p = sim.params
    inv = {k: float(v) for k, v in invariants(sim.psi_k, sim.pi_k, sim.op, p["a"], p["s"], p["f"]).items()}
    base = sim.__dict__.setdefault("_inv0", dict(inv))
    inv["Q_rel_drift"] = (inv["Q"] - base["Q"]) / (abs(base["Q"]) + 1e-300)
    inv["E_rel_drift"] = (inv["E"] - base["E"]) / (abs(base["E"]) + 1e-300)
    return {"kg_" + k if k in ("mass", "amp") else k: v for k, v in inv.items()}


@observer("tg_diagnostics", 1, """All TG-B1S diagnostics (energies, charge, node position/width, T/G at the
node, source/damping/boundary rates). tg-rk4 only.""", {"type": "object"})
def _tg_diag(sim):
    if sim.kind != "tg-rk4":
        raise ValueError("tg_diagnostics needs the tg-rk4 substrate")
    d = sim._b1s.diagnostics(sim.state, sim.cfgv, sim.refs, sim.g)
    return {"tg_" + k: float(np.asarray(v)) for k, v in d.items()}


@observer("energy_ratio", 1, """er = Σ|psi|² / Σ|psi_0|² on raw grid sums (the a* harnesses' er(t),
core_saturation_search.evaluate_candidate).""", {"type": "object"})
def _er(sim, field="psi"):
    e = float(np.sum(np.abs(sim.fields()[field]) ** 2))
    sim.__dict__.setdefault("_e0", e)
    return {"er": e / sim._e0}


# =============================================================== spec checks / description

def check_spec(spec) -> list:
    errs = []

    def chk(table, comp, where):
        name = comp.get("name")
        if name not in table:
            errs.append("%s: unknown component '%s' (known: %s)" % (where, name, ", ".join(sorted(table))))
            return
        want = comp.get("version")
        if want is not None and want != table[name]["version"]:
            errs.append("%s: '%s' is version %d, spec pins %d" % (where, name, table[name]["version"], want))
        req = (table[name]["params"] or {}).get("required", [])
        missing = [r for r in req if r not in (comp.get("params") or {})]
        if missing and where != "substrate":
            errs.append("%s: '%s' needs params %s" % (where, name, missing))

    chk(SUBSTRATES, spec["substrate"], "substrate")
    chk(ICS, spec["protocol"]["ic"], "protocol.ic")
    for i, o in enumerate(spec["observers"]):
        chk(OBSERVERS, o, "observers[%d]" % i)
    return errs


def describe() -> dict:
    """Machine-readable component list (used by the MCP list_components tool and the spec UI)."""
    def rows(t):
        return [{"name": v["name"], "version": v["version"], "doc": v["doc"], "params": v["params"]}
                for v in sorted(t.values(), key=lambda r: r["name"])]
    return {"substrates": rows(SUBSTRATES), "ics": rows(ICS), "observers": rows(OBSERVERS)}


def build(spec):
    """Concrete spec -> (sim, observers list). Does not run anything."""
    pr = spec["protocol"]
    grid = _Grid(pr["grid"]["N"], pr["grid"]["L"])
    sim = SUBSTRATES[spec["substrate"]["name"]]["fn"](grid, float(pr["dt"]), spec["substrate"].get("params", {}))
    fields = ICS[pr["ic"]["name"]]["fn"](sim, **(pr["ic"].get("params") or {}))
    sim.init(fields)
    # `slopes` is an executor-level option (tools/run_spec.py fits d(key)/dt); it is not an observer argument
    obs = [(o["name"], OBSERVERS[o["name"]]["fn"],
            {k: v for k, v in (o.get("params") or {}).items() if k != "slopes"}) for o in spec["observers"]]
    return sim, obs


@ic("nls_soliton", 1, """Exact conservative-branch soliton by Petviashvili (phase_d_c2_3_exact_soliton.petviashvili,
the C2.3/C2.7 harnesses' own), optionally Galilean-boosted by exp(i k x) with k = 2*pi*boost_n / L.
etdrk4-sncgl with kinetic_mode='conservative' only; grid L must be 10 (the harness constant).""",
    {"type": "object", "required": ["mu"]})
def _nls_soliton(sim, mu, A=1.0, sig=0.08, boost_n=0, max_residual=1e-6):
    from jax_scout.phase_d_c2_3_exact_soliton import petviashvili
    if getattr(sim, "kind", "") != "etdrk4-sncgl" or sim.params.get("kinetic_mode") != "conservative":
        raise ValueError("nls_soliton needs etdrk4-sncgl with kinetic_mode='conservative'")
    if abs(sim.grid.L - 10.0) > 1e-12:
        raise ValueError("nls_soliton is defined on L=10 (core_saturation_search.L_)")
    psi, prof = petviashvili(float(A), float(sig), sim.ops, sim.grid.N, float(mu))
    if psi is None or prof["residual"] > max_residual:
        raise RuntimeError("Petviashvili failed: %s" % (prof,))
    k = 2 * np.pi * int(boost_n) / sim.grid.L
    sim.__dict__["_ic_info"] = {"petviashvili_residual": float(prof["residual"]), "boost_k": k}
    return {"psi": (np.asarray(psi) * np.exp(1j * k * sim.grid.X)).astype(np.complex128)}
