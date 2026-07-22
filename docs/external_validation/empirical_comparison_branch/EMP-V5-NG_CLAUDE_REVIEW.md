# Claude Review — EMP-V5-NG (Nguyen et al. 2014, matter-wave soliton collisions)

**Verdict:** `DIRECTION_CONSISTENT_MECHANISTICALLY_MEANINGFUL_BUT_COARSE`
**Status:** review complete; comparison stays `PROVISIONAL` context — no verdict/gate/framing change.
**Match level:** `CLOSE ANALOGUE` (ceiling; not raised).

## What the source is / what Codex did

Nguyen, Dyke, Luo, Malomed, Hulet, *Nat. Phys.* 10, 918 (2014) — collisions of bright matter-wave solitons in an
attractive BEC. Codex extracted the **phase-dependent collision outcome** from the arXiv source package (Fig. 2a/b
in-phase, Fig. 2c + Fig. 3 anti-phase), rendering figures from the source PDFs. Execution was again diligent and
honest: arXiv PDF + source + rendered figures archived off-git with checksums; only the 4-row derived CSV +
provenance + comparator output committed (`9fc7785`); guardrails attested and protected diff empty. It flagged that
**Fig. 3's anti-phase class was inferred** from the density-minimum/out-of-phase context rather than a stated phase,
and recorded Fig. 2c as `NO_COLLAPSE` (robust survival, identities not tagged) vs Fig. 3 `PASS_THROUGH` (tagged 2:1
trajectory, text says the solitons pass through) — a careful, defensible distinction.

## The result, and why it is genuinely meaningful

`DIRECTION_CONSISTENT`: in-phase transmission 0/2, anti-phase 2/2 — transmission concentrated at anti-phase, the
same direction as the C3 grid.

This is **stronger than the V1 overlay**, because on the anti-phase side the correspondence is *mechanistic, not
just phenomenological*. In the BEC experiment, out-of-phase solitons survive collisions because destructive
interference suppresses the density spike that would otherwise cause collapse. In our C3 result, transmission
occurs only at exact anti-phase because a **destructive-interference node** forms at the collision midplane and
forbids the merge. **Same physical cause — an anti-phase interference node protecting colliding solitons — in a
real experiment and in the simulation.** That is a real, if qualitative, corroboration of the anti-phase
transmission channel.

## Why it is nonetheless coarse — and what it does NOT corroborate

The dataset is **2 phase points (0, π), 4 rows, no off-phase, no speed axis**. It tests one bit of the C3 grid —
in-phase sticks / anti-phase transmits — and leaves the two *distinctive* C3 findings untested:
- **Narrowness of the anti-phase channel.** Our C3 result's non-trivial content is that transmission is a *narrow*
  feature at *exact* anti-phase — off-phase (3π/4, 7π/8) still captures. This experiment has no off-phase data, so
  it cannot corroborate the narrowness, only the endpoint.
- **Speed dependence.** C3 finds anti-phase transmits only below ~0.5c. No collision speed was extracted, so this
  is untested.

The comparator now reports this automatically (`coverage.scope = coarse_inphase_vs_antiphase_only`, with the
untested features listed), so the result cannot be overread as validating the full phase×speed structure.

## The one asymmetry to keep honest

The **anti-phase** side is the deep correspondence (interference node in both). The **in-phase** side is
*phenomenological only*: the experiment collapses (density-spike destruction / atom loss), our C3 captures (binds
into a bound state). Both are "not transmission," but by different mechanisms. So the clean 0/2 vs 2/2 is carried
by a genuine anti-phase mechanism match plus a looser in-phase "both fail to pass through" grouping — not two
equally-deep correspondences.

## Statement to carry forward

> An independent matter-wave soliton collision experiment (Nguyen 2014) shows the same qualitative direction as the
> C3 phase×speed grid — in-phase collisions do not transmit (they collapse/merge), out-of-phase collisions survive
> and pass through — and on the anti-phase side this shares the C3 mechanism (a destructive-interference node
> protecting the solitons). A `CLOSE ANALOGUE`. It corroborates the coarse in-phase/anti-phase direction only; it
> does not test the narrowness of the anti-phase channel or the speed dependence, and the in-phase correspondence
> is phenomenological (collapse ≠ binding).

## What this does and does not establish

- **Does:** qualitative, mechanistically-motivated external support for the anti-phase transmission channel — the
  strongest external corroboration in the branch so far.
- **Does not:** establish physical correspondence, corroborate the anti-phase *narrowness* or the speed boundary,
  or equate BEC collapse with C3 capture; touches nothing about matter/gravity/unification; changes no verdict.

## Next steps (optional)

1. Corroborating the **narrowness** would need *off-phase* collision data (rare experimentally) — likely not
   available; flag as a known gap rather than a task.
2. Corroborating the **speed dependence** would need velocity-resolved collision outcomes; check whether Nguyen's
   supplementary movies or a follow-up paper report collision-velocity-resolved survival.
3. Branch status after V1 + V5: two qualitative `CLOSE ANALOGUE` overlays banked (V1 sign law weak-branch; V5
   anti-phase channel, mechanistic). This is a reasonable point to **bank the empirical branch** unless a
   velocity-resolved or off-phase dataset surfaces.
