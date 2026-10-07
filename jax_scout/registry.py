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

# protocol.precision -> (real dtype, complex dtype). fp64 is the default and the only precision any
# catalogued result uses. fp32 is for SCREENING only (docs/research_infrastructure/BATCHED_RUNS.md:
# 3.7-4.5x faster on the GTX 1080, ~0.1-0.5% drift over an a* replay) and is offered only by substrates
# listed in PRECISIONS; everything else is fp64-only.
DTYPES = {"fp64": (jnp.float64, jnp.complex128), "fp32": (jnp.float32, jnp.complex64)}
PRECISIONS = {"etdrk4-sncgl": ("fp64", "fp32")}


def precision_of(spec):
    return spec["protocol"].get("precision", "fp64")


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
def _etdrk4(grid, dt, params, precision="fp64"):
    from jax_scout import physics

    class Sim:
        kind = "etdrk4-sncgl"

        def __init__(self):
            self.grid, self.dt, self.params, self.t = grid, dt, dict(params), 0.0
            self.precision = precision
            self.rd, self.cd = DTYPES[precision]
            self.ops = physics.build_operators(grid.N, grid.L, dt, self.params, self.rd, self.cd)
            self._adv = jax.jit(lambda pk, n: jax.lax.fori_loop(0, n, lambda i, p: physics.step(p, self.ops), pk),
                                static_argnums=1)

        def init(self, fields):
            self.psi_k = physics.initial_psi_k(jnp.asarray(fields["psi"], dtype=self.cd), self.ops)

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
    prec, name = precision_of(spec), spec["substrate"].get("name")
    if prec not in PRECISIONS.get(name, ("fp64",)):
        errs.append("protocol.precision: '%s' does not support %s (supports: %s)"
                    % (name, prec, ", ".join(PRECISIONS.get(name, ("fp64",)))))
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
    name, prec = spec["substrate"]["name"], precision_of(spec)
    if prec not in PRECISIONS.get(name, ("fp64",)):
        raise ValueError("substrate '%s' does not support precision %s" % (name, prec))
    extra = {"precision": prec} if name in PRECISIONS else {}
    sim = SUBSTRATES[name]["fn"](grid, float(pr["dt"]), spec["substrate"].get("params", {}), **extra)
    fields = ICS[pr["ic"]["name"]]["fn"](sim, **(pr["ic"].get("params") or {}))
    sim.init(fields)
    # `slopes` is an executor-level option (tools/run_spec.py fits d(key)/dt); it is not an observer argument
    # `slopes` and `final_only` are executor-level options (tools/run_spec.py), not observer arguments
    obs = [(o["name"], OBSERVERS[o["name"]]["fn"],
            {k: v for k, v in (o.get("params") or {}).items() if k not in ("slopes", "final_only")})
           for o in spec["observers"]]
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


def export(path=None):
    """Write docs/registry/components.json so JAX-less tools (the MCP server in .venv) can list components."""
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = path or os.path.join(root, "docs", "registry", "components.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"_about": "Generated by `python -m jax_scout.registry --export`; do not edit.", **describe()},
                  fh, indent=2)
        fh.write("\n")
    return path


if __name__ == "__main__":
    import sys
    if "--export" in sys.argv:
        print(export())
    else:
        import json
        print(json.dumps(describe(), indent=2))


@observer("state_descriptors", 2, """Final-state descriptor vector for basin clustering (Phase F).
v2 (2026-10-04): adds d_contrast = max(rho)/mean(rho); node-based descriptors are zeroed when the field
is not localised (contrast < contrast_min, default 20), because a dispersed field always has speckle
above the node threshold and its "nodes" are noise (seen in the first KG ensemble demo).
Permutation-invariant by construction (no node ordering): mass, amp, n_nodes, mean/std node size,
periodic node-node distance mean/std/min, periodic radius of gyration, power-weighted spectral |k|,
and the anisotropy of the density inertia tensor (min/max eigenvalue). Use with params
{"final_only": true}: it is too expensive to sample every chunk on large grids.""", {"type": "object"})
def _descriptors(sim, field="psi", contrast_min=20.0):
    from jax_scout import transfer_diag as td
    g = sim.grid
    psi = sim.fields()[field]
    rho = np.abs(psi) ** 2
    mass = float(rho.sum() * g.dV)
    contrast = float(rho.max() / (rho.mean() + 1e-300))
    nodes = td.detect_nodes(psi, g.dx) if contrast >= contrast_min else []
    out = {"d_mass": mass, "d_amp": float(np.sqrt(rho.max())), "d_contrast": contrast,
           "d_n_nodes": float(len(nodes))}
    sizes = [float(n.get("size", 0)) for n in nodes]
    out["d_node_size_mean"] = float(np.mean(sizes)) if sizes else 0.0
    out["d_node_size_std"] = float(np.std(sizes)) if sizes else 0.0
    cents = np.array([np.asarray(n["centroid"], dtype=float) * g.dx for n in nodes]) if nodes else np.zeros((0, 3))
    if len(cents) >= 2:
        d = cents[:, None, :] - cents[None, :, :]
        d -= g.L * np.round(d / g.L)                      # minimum image
        r = np.sqrt((d ** 2).sum(-1))[np.triu_indices(len(cents), 1)]
        out.update(d_pair_mean=float(r.mean()), d_pair_std=float(r.std()), d_pair_min=float(r.min()))
    else:
        out.update(d_pair_mean=0.0, d_pair_std=0.0, d_pair_min=0.0)
    w = rho / (rho.sum() + 1e-300)
    rg2 = 0.0
    for ax, X in enumerate((g.X, g.Y, g.Z)):
        th = 2 * np.pi * (X + g.L / 2) / g.L
        R = abs(np.sum(w * np.exp(1j * th)))              # circular concentration: 1 = point, 0 = uniform
        rg2 += (g.L / (2 * np.pi)) ** 2 * (-2.0 * np.log(max(R, 1e-12)))
    out["d_rgyr"] = float(np.sqrt(rg2))
    P = np.abs(np.fft.fftn(psi)) ** 2
    k1 = 2 * np.pi * np.fft.fftfreq(g.N, d=g.dx)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    out["d_k_mean"] = float(np.sum(P * np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)) / (P.sum() + 1e-300))
    if len(cents) >= 3:
        c = cents - cents.mean(0)
        c -= g.L * np.round(c / g.L)
        ev = np.linalg.eigvalsh(c.T @ c / len(c))
        out["d_aniso"] = float(ev[0] / (ev[-1] + 1e-300))
    else:
        out["d_aniso"] = 0.0
    return out


# =============================================================== batched (vmapped) ETDRK4

def batch_key(spec):
    """Specs with equal keys can share one vmapped batch: same substrate, grid, dt and all STATIC operator
    arguments. Only physics.BATCHABLE_PARAMS and the initial condition may differ between members."""
    from jax_scout import physics
    if spec["substrate"]["name"] != "etdrk4-sncgl":
        return None
    pr = spec["protocol"]
    _, static = physics.operator_args(spec["substrate"].get("params", {}))
    return (pr["grid"]["N"], pr["grid"]["L"], pr["dt"], precision_of(spec), tuple(sorted(static.items())))


class BatchedETDRK4:
    """Advance B members of the ETDRK4 substrate together with jax.vmap over (parameters, state).

    Each member is an ordinary Sim built by build() -- so its IC, observers and recording are exactly the
    single-run ones -- and this object only replaces the time stepping: advance(n) runs n steps for all
    members in one compiled call and writes each member's state back into its Sim. Members must share
    batch_key(); their BATCHABLE_PARAMS may differ.
    """

    def __init__(self, sims, specs):
        from jax_scout import physics
        self.sims = sims
        s0 = specs[0]
        pr = s0["protocol"]
        N, L, dt = pr["grid"]["N"], pr["grid"]["L"], pr["dt"]
        dyn = [physics.operator_args(s["substrate"].get("params", {}))[0] for s in specs]
        _, static = physics.operator_args(s0["substrate"].get("params", {}))
        self.pvec = jnp.asarray([[d[k] for k in physics.BATCHABLE_PARAMS] for d in dyn], dtype=jnp.float64)
        self.psi_k = jnp.stack([s.psi_k for s in sims])
        self.dt = dt
        rd, cd = sims[0].rd, sims[0].cd

        def one(p, pk, n):
            ops = physics.ops_from_args(N, L, dt, dict(zip(physics.BATCHABLE_PARAMS, p)), static, rd, cd)
            return jax.lax.fori_loop(0, n, lambda i, q: physics.step(q, ops), pk)

        self._adv = jax.jit(jax.vmap(one, in_axes=(0, 0, None)), static_argnums=2)

    def advance(self, n):
        self.psi_k = self._adv(self.pvec, self.psi_k, int(n))
        for i, s in enumerate(self.sims):
            s.psi_k = self.psi_k[i]
            s.t += n * self.dt


def max_batch(N, budget_gb=5.0, precision="fp64"):
    """Members that fit on the GPU at once. ETDRK4 holds roughly 20 complex N^3 arrays per member
    (state, 4 stage nonlinearities, geometry buffers, FFT workspace); budget leaves headroom on 8 GB.
    complex128 is 16 bytes per point, complex64 is 8."""
    itemsize = 16 if precision == "fp64" else 8
    return max(1, int(budget_gb * 1e9 // (20 * itemsize * N ** 3)))
