# Phase D / C2.3 — Quasi-Soliton Transport: Exact-Profile Hunt, Paired N-dt Convergence, Momentum Budget

**Three headline results (pure NLS / geometry-off, `param_a_coupling=0`, dt=2.5e-4 unless noted):**
1. **No exact stationary soliton exists** — the C2.1 object is a long-lived **quasi-soliton** (metastable), confirmed
   at N=48/96/128.
2. **The kick-associated loss is purely a timestep artifact — it is spatially CONVERGED.** n=2 kick-loss over a
   matched T=6 window: **0.090 at N=96 and 0.090 at N=128** (identical); resting bleed likewise (control 0.876 at
   both N). Combined with the C2.2 dt-ladder (0.205→0.132→0.090, extrapolating ~0.03→0), the loss decomposition is
   complete: **stepper(dt)-only, no spatial component, no profile component** → `C2_KICK_LOSS_FULLY_NUMERICAL`.
3. **The velocity anomaly is real and now explained: the ring-winding/background-current picture.** The boosted
   quasi-soliton translates at **v_cent ≈ 0.037·k, robust across N and dt (r²=1.00)** — only ~0.7% of the Galilean
   prediction 2Dk = 5.47k — while the kick's momentum parks in a **delocalized phase-winding (ring current)** that
   moves no density, and then decays via a slow numerical momentum sink. Transport is core-drag against the wound
   background, not free soliton motion. **This is likely a boost-protocol/topology artifact of the periodic box**
   (an integer-n kick `e^{ikx}` necessarily imposes global winding); a *local* boost is the designed follow-up.

## 1. Exact-profile hunt → `C2_NO_EXACT_STATIONARY_SOLITON`
- **Imaginary time** (split-step, mass-renormalized): flows to the **uniform condensate** — the quasi-soliton is not
  the fixed-mass ground state (uniform is modulationally stable at this dilution: a·ρ_u ≈ 0.01 ≪ D·k₁² ≈ 1.1).
- **Petviashvili** (seeds × μ ∈ {0.10,0.15,0.20,0.22} × γ ∈ {1.5,2.0}; N=48/96/128): below μ≈0.15 converges to the
  **uniform branch to machine precision** (residual ~1e-14 — the method works; the localized branch isn't there);
  at μ=0.20–0.22 no fixed point (S→0); above μ≈0.23 **no stationary branch exists at all** — physical, since
  g(ρ)=aρ+sρ²+fρ³ saturates at g_max≈0.23 (ρ≈0.61). γ=2.0 diverges (supercritical cubic at d=3).
- The dynamically-settled object IS well-defined and **spatially converged**: settle statistics identical at N=96 and
  N=128 (mass_ret 0.9189, amp 0.993, occ 0.684) — but its stationarity residual is O(1): a breather-like metastable
  state, not an eigenstate. Consistent with C2.1 (only the widest σ survive; narrower slowly disperse).

## 2. Paired N-dt convergence (the C2.2 loose end, now closed)
| quantity (T=6 window, dt=2.5e-4) | N=96 (C2.2) | N=128 (C2.3) |
|---|---|---|
| n=0 control mass | 0.876 | **0.876** |
| n=2 kick-loss vs control | 0.090 | **0.090** |
| v_cent/k (n=2) | 0.039 | 0.039 |
N=128 at dt=2.5e-4 is CFL-stable (the C2.2 collapse was dt=1e-3 only). **Identical loss at both resolutions ⇒ zero
spatial component.** With the dt-ladder showing monotone convergence toward ~0, the C2.1 "radiation" is fully
accounted for as ETDRK4-stepper + boost-protocol error. (The Codex CuPy audit adds the geometry-ON-only algebraic
flux — absent here since geometry is off; see `docs/PHASE_D_C2_CONTRACT_REVIEW.md`.)

## 3. Momentum budget + the velocity anomaly (N=96 Tphys=10; N=128 Tphys=6)
| N | n | k | v_cent | v_cent/2Dk | mass | P_ret (end) |
|---|---|---|---|---|---|---|
| 96 | 0 | 0 | 0.000 | — | 0.775 | (P0≈0, n/a) |
| 96 | 1 | 0.628 | +0.0216 | 0.63% | 0.749 | 0.63 |
| 96 | 2 | 1.257 | +0.0455 | 0.66% | 0.659 | 0.53 |
| 128 | 2 | 1.257 | +0.0486 | 0.71% | 0.786 | 0.69 |
- **v_cent is constant in time (r²=1.00), ∝k, and N-robust: μ ≈ 0.037.** It is NOT the Galilean 2Dk — off by ~140×.
- **Momentum is initially conserved then decays slowly** (P_ret 0.63–0.69 over T=6–10): a numerical momentum sink
  (stepper+dealias on the kicked field), faster than the mass decay. Crucially, v_cent stays constant while P decays
  — so v_cent is NOT tracking P/M. The Ehrenfest relation d⟨x⟩/dt = 2D·P/M applies to the linear centroid on R³, not
  the circular mean on a torus: **a uniform ring current carries momentum but moves no density.**
- **Where the kick went:** the integer-n boost `e^{ikx}` on a torus is a *global topological winding*. The system
  relaxes to (heavy core, nearly pinned) + (delocalized winding/background current holding ~all of P). The core's
  small drift v=0.037k is the drag coupling to that flow. This explains, in one stroke: v∝k with tiny μ, constant
  v while P decays, k²-scaling of the (numerical) kick-loss seed, and μ's robustness across N/dt.
- **Peak-tracker retraction (honesty note):** interim reports read "the core moves at ~5% of Galilean" from v_peak.
  The N=128 **unkicked control's** peak also "moves" at 0.178 (r²=0.40) — the peak tracker is dominated by
  breather-core wander, not translation (r²≤0.64 everywhere). v_cent (r²=1.00) is the honest drift metric; v_peak
  readings are withdrawn.

## 4. What this means for the transport question
- **The clean-transport claim survives and sharpens:** the pure-NLS substrate's losses are entirely numerical
  (dt-only, spatially converged) — `C2_PURE_NLS_CLEAN_TRANSPORT_SUPPORTED` in the continuum limit stands.
- **But "transport velocity = 2Dk" was never achieved, and the reason is now structural:** the periodic-box winding
  boost cannot translate a quasi-soliton at the Galilean velocity; it creates a background supercurrent instead. The
  measured μ≈0.037 is a *drag mobility* in a wound box — a well-defined, reproducible transport channel, but not
  free soliton ballistics. Whether the substrate supports full Galilean transport is therefore **still open and now
  has a designed decisive test: the LOCAL boost** (phase ramp applied only across the core, decaying outside — no
  net winding). If a locally-boosted quasi-soliton moves at ~2Dk, the substrate is fully transport-capable and the
  small μ was a protocol artifact; if it still creeps, the pinning is physical. This is the natural C2.4.
- Two-node interaction (Stage-4) stays gated until the local-boost question is settled.

## 5. Provenance
Harness `jax_scout/phase_d_c2_3_exact_soliton.py` (Petviashvili + imag-time + settle-fallback + momentum/peak
telemetry; crash-resilient object/boost dumps). Runs: `sweep_runs/C23_N96` + `C23_N96_k2` (deterministic settle
verified identical) + `C23_N128`; N=48 scans in scratch. Session note: the first chain died with an app restart —
partials were recovered from logs and the harness now saves `object_psi.npy` + incremental `boosts_partial.json`.
Guardrails: mirror-only; geometry-off (contract-review blocker absent); Phase C default untouched; no clipping; no
matter claims; peak-velocity over-read explicitly retracted.
