# datasets/ — off-git raw storage pointer

**Raw digitized data does NOT live in git.** Like the other external data (`../../external_data/`, stored on the
E: drive), raw papers, figure images, and large extracted arrays are stored **off-git** to keep the repository
small and to avoid redistributing copyrighted figures.

## Convention

- Raw source → `E:\quantule_mapper_external_data\empirical_comparison\<EMP-id>\` (mirrors the existing external-data
  layout, with `metadata/` + checksums).
- **Only** small derived comparison tables (CSV/JSON) and the completed provenance record are committed, and they
  live in the parent `empirical_comparison_branch/` directory, not here.
- This `datasets/` folder holds only this pointer; it exists so the layout is explicit.

## Do not

- Do not commit downloaded papers, figure images, or large arrays.
- Do not point `validation_config.yaml`, the `external_data.py` manifest, or any Hunter/production config at
  anything under here. This path is comparison-branch-only and opt-in (see `../README.md`).
