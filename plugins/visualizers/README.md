# Deprecated Visualizers

`plugins/visualizers` is now a compatibility surface only.

Active read-only visualization logic lives in `F:/quantule_mapper/quantule_viz`.

Boundaries for the current refactor:

- `quantule_viz` reads saved JSON/CSV/NPZ artifacts and writes reports.
- `jax_scout` remains the producer of simulation and characterization artifacts.
- legacy plugin scripts in this folder should be treated as wrappers or historical utilities, not the active visualization system.

Preferred commands:

```powershell
python -m quantule_viz phase-c <run_dir>
python -m quantule_viz feb-bound-state <run_dir>
python -m quantule_viz core-basin <run_dir>
python -m quantule_viz core-characterize <run_dir>
python -m quantule_viz frames <npz_path> --outdir <dir>
```
