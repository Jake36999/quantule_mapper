---
tags: [experiment, SECTOR]
date: YYYY-MM-DD
branch: Branch - NAME - Index
status: planned
---

# EXPERIMENT NAME

## Summary

One paragraph: what this experiment asks, why it matters now, and how it connects to the wider
project. Written so a third party who has read nothing else can follow what comes next.

> [!abstract] Question
> The single falsifiable question, stated so that both outcomes are describable in advance.

> [!info] Preregistered outcomes
> - **If X** → interpretation A
> - **If Y** → interpretation B
> - **If neither** → what that would mean
>
> Preregister *before* running. A result that can only be read one way was not a test.

## Why now

What made this the next thing to do. Which finding, gap, or blocker it follows from. Link the
document that motivated it.

## Method

- **Harness:** `jax_scout/...`
- **Observable:** what is measured, and in what units
- **Controls / nulls:** the arm that must come out flat or reverse
- **Gates:** the preregistered pass conditions

> [!warning] Known limitations of the instrument
> Anything about the observable that constrains how the result may be read. If the observable has a
> weighting, a normalization, or a domain choice, name it here — this is where interpretation errors
> get caught early.

## Result

Fill after the run. Link the run note rather than restating its numbers:

- Run: [[RUN_ID]]
- Verdict: `VERDICT_STRING`

![[runs/_plots/RUN_ID/figure.png]]

What this figure shows, what to look at, and what it means. **Every figure gets a sentence.**

## Interpretation

What we now believe, stated no more strongly than the evidence supports. Separate:

- **What this establishes**
- **What it does not establish**
- **What it downgrades or retracts**

## Caveats

Bulleted, honest, and specific. A caveat that does not say how it could change the conclusion is
decoration.

---

## Previous experiments

[[Previous experiment]]

## Associated docs

[[Associated docs]]

## Next experiment

[[Next experiment and or doc]]

## Branches

- [[Branch - NAME - Index]]
