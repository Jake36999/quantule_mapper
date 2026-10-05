# Field Visual Analysis Suite

`quantule_viz field-suite` is the standard read-only visual pass for saved field artifacts. It is intended for Colab fast-lane capsules that can afford larger RAM, GPU memory, disk, and HDF5 outputs than the local workstation.

## Command

```powershell
python -m quantule_viz field-suite <artifact.h5|artifact.hdf5|artifact.npz|run_dir> --outdir <visual_outdir> --overwrite
```

Useful options:

```text
--fps 8
--rho-percentile 99.7
--topology-quantile 0.995
--max-gif-frames 160
```

## Outputs

The suite writes:

```text
density_slices.png
density_evolution.gif
vector_current.png, or vector_current.md if no complex psi field exists
topology_active_set.png
topology_components.csv
energy_timeseries.png
visual_suite_manifest.json
```

## Supported Inputs

HDF5 datasets are resolved in this order:

```text
psi_history, psi_frames, psi, frames, field_history
rho_history, rho_frames, rho
psi_final
rho_final
telemetry/{step,time,energy,energy_total,C_invariant,max_amplitude}
```

NPZ datasets are resolved similarly:

```text
psi, psi_history, frames, field_history
rho, rho_history, density
times, time, t, steps
energy, energy_total, C_invariant, max_amplitude
```

## Colab Fast-Lane Guidance

For A100 jobs, prefer saving larger HDF5/NPZ field snapshots during the simulation and rendering this suite as a separate post-processing stage. This uses Colab's extra RAM and disk for higher sample size without forcing the live solver loop to spend time on PNG/GIF rendering.

The visual suite is diagnostic only. It does not modify solver source, scientific equations, acceptance gates, or conclusions. If stored energy telemetry is missing, `energy_timeseries.png` is labelled as a `rho_integral_proxy`.
