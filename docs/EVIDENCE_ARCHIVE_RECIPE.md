# Evidence Archive Recipe (H3)

**Purpose:** an **off-box** backup recipe for the load-bearing evidence, so the closed Phase C claims stay
reproducible-from-data even if the 26 GB `sweep_runs/` is pruned. **Git stays lightweight** (backup/portfolio for
code, not the artifact archive) — the local disk + this off-box copy are the canonical evidence archive, now that
`EVIDENCE_INVENTORY.md` gives path/hash/claim traceability. **Do not push generated artifacts into git.**

## What to archive
The **22 load-bearing runs** from `EVIDENCE_INVENTORY.md` (stability arc §A: 14 runs; mobility+a\* arc §B: 8
runs). Two archive tiers:

### Tier 1 — Minimal (metadata-only, ~a few MB)
Every claim's *numbers* live in the `*.csv` + `*_summary.json`. This tier reproduces every table/verdict.
```bash
# from F:\quantule_mapper (Git Bash / WSL)
DEST=/e/irer_archive/evidence        # <- external disk / cloud-synced folder
mkdir -p "$DEST"
while read r; do
  mkdir -p "$DEST/$r"
  find "sweep_runs/$r" -maxdepth 1 -type f \( -name '*.csv' -o -name '*.json' \) -exec cp {} "$DEST/$r/" \;
done < <(cat docs/_evidence_loadbearing.txt 2>/dev/null || sed -n 's/^| `\([A-Z_0-9]*\)`.*/\1/p' docs/EVIDENCE_INVENTORY.md)
```

### Tier 2 — Recommended (metadata + key states, ~1–2 GB)
Adds the `psi_fin` fields needed to *re-derive* (not just re-read) the headline claims: the a\*-arc states and a
few basin exemplars.
```bash
KEY="FEB_GAIN_LADDER_LONGT_T72000_20260701_175708 FEB_ASTAR_CONFIRM_20260702_003055 \
FEB_BREATHING_LONGT_T72000_20260628_003032 FEB_CENTER_RESOLUTION_N128_20260626_195449 \
FEB_CORE_DELINEATION_T24000_20260627_175050"
for r in $KEY; do mkdir -p "$DEST/$r"; cp sweep_runs/$r/*.npz "$DEST/$r/" 2>/dev/null; done
# (skip the huge param/joint-basin .npz — 600+ MB each — unless doing a full archive)
```

## Checksum verification (matches the manifest)
```bash
# recompute the metadata sha256 and eyeball against EVIDENCE_INVENTORY.md (first 16 hex)
for r in $(sed -n 's/^| `\([A-Z_0-9]*\)`.*/\1/p' docs/EVIDENCE_INVENTORY.md); do
  find "$DEST/$r" -type f \( -name '*.csv' -o -name '*.json' \) -print0 2>/dev/null | sort -z \
    | xargs -0 cat 2>/dev/null | sha256sum | cut -c1-16 | sed "s|^|$r  |"
done
# full integrity manifest for the archive itself:
( cd "$DEST" && find . -type f -exec sha256sum {} \; > MANIFEST.sha256 )
```

## Suggested folder structure (external disk / cloud)
```
irer_archive/
  evidence/
    <RUN_ID>/            one dir per load-bearing run (csv + summary.json [+ npz])
    MANIFEST.sha256      full checksum manifest of the archive
  docs_snapshot/         copy of EVIDENCE_INVENTORY.md + BASELINE_AUDIT*.md at archive time
  README.txt             date, git commit of the code, what tier was archived
```

## Restore
```bash
# verify integrity first
( cd /e/irer_archive/evidence && sha256sum -c MANIFEST.sha256 | grep -v ': OK$' || echo "all OK" )
# restore a run back into the repo tree (results stay gitignored)
cp -r /e/irer_archive/evidence/<RUN_ID> sweep_runs/
```
Re-derivation, if a state is missing: re-run the script named in `EVIDENCE_INVENTORY.md` for that run at the
frozen geometry `e8d6a78ea` (see `BASELINE_REPRODUCTION_RUNBOOK.md`).

## Notes
- This is a **recipe**, not an automated push — the actual copy is a deliberate operator action to a location you
  control. Tier 1 is enough for claim traceability; Tier 2 for full re-derivation.
- Keep it lightweight in git: only this recipe + `EVIDENCE_INVENTORY.md` are versioned; the artifacts are not.
