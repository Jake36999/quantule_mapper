---
tags: [record, SECTOR]
date: YYYY-MM-DD
branch: Branch - NAME - Index
status: complete
---

# TITLE

Author, date, scope. State whether this is an audit, a design contract (an *intention*, possibly never
executed), a bug report, or a synthesis — the reader needs to know what kind of claim to expect.

> [!info] Sources
> Every file, run directory and code path this document draws on. A document of record that cannot be
> traced back to artifacts is an opinion.

## Summary

The finding, up front, in a paragraph. If this document changes a previously published conclusion, say
so in the first three sentences.

## Detail

Sections as needed. Quote code with file:line so claims are checkable:

```python
# jax_scout/module.py:60
...
```

## What this changes

| status | item |
|---|---|
| **Unchanged** | |
| **Downgraded** | |
| **Retracted** | |
| **Newly explained** | |

## Recommendations

| id | change | why |
|---|---|---|
| R-a | | |

---

## What changed as a result

*The single most important section for a future reader. Fill it in when the consequences land,
not when the document is written.*

- **Code / model changes:** what was altered in the harness or the model because of this, with
  commit or file references. "Nothing" is a valid and useful answer.
- **Verdicts changed:** anything confirmed, retracted, downgraded or superseded by this.
- **What was done next, and why:** the decision this document caused.
- **If nothing changed:** say so explicitly, and say whether that was a decision or a drift.

## Issues raised

*Track each issue to a status. An issue with no status is the thing that makes a document
impossible to review later.*

| # | issue | status | resolved by |
|---|---|---|---|
| 1 | | OPEN / RESOLVED / SUPERSEDED / WONTFIX | |

---

## Previous experiments

[[Previous experiment]]

## Associated docs

[[Associated docs]]

## Next experiment

[[Next experiment and or doc]]

## Branches

- [[Branch - NAME - Index]]
