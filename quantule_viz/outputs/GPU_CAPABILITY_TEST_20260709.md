# GPU Capability Test - 2026-07-09

## Summary

Two GPUs are visible to Windows:

- `NVIDIA GeForce GTX 1080`
- `Radeon RX 5500 XT`

The current Quantule Mapper Python runtime remains CUDA-only for GPU compute:

```text
F:\quantule_mapper\.venv\Scripts\python.exe
```

CuPy sees only the GTX 1080. The Radeon RX 5500 XT is visible through OpenCL and Vulkan, but the current project venv does not have a Python OpenCL, Vulkan, DirectML, or GPU rendering binding installed.

## Windows Device Visibility

`Win32_VideoController` reports:

- NVIDIA GeForce GTX 1080: status `OK`
- Radeon RX 5500 XT: status `OK`

`nvidia-smi` reports:

```text
GPU 0: NVIDIA GeForce GTX 1080
memory.total: 8192 MiB
memory.free: about 8059 MiB at initial query
driver: 581.80
display_active: Disabled
```

Interpretation: the GTX 1080 is no longer acting as the active display GPU, which is good for preserving it as the CUDA worker.

## Python Runtime Library Probe

Available in the active venv:

- `cupy 14.0.1`
- `numpy 2.4.3`
- `h5py 3.16.0`
- `imageio 2.37.3`

Not currently available in the active venv:

- `pyopencl`
- `torch`
- `onnxruntime`
- `tensorflow`
- `numba`
- `moderngl`
- `glfw`
- `plotly`
- `cv2`

Implication: Python can currently communicate with the GTX 1080 via CuPy, but cannot directly compute on the AMD card from the project venv without adding a new dependency path.

## CuPy / CUDA GTX 1080 Probe

CuPy result:

```json
{
  "cuda_device_count": 1,
  "device": "NVIDIA GeForce GTX 1080",
  "cupy_version": "14.0.1"
}
```

Small bounded benchmark:

```text
FP64 vector triad: about 95.1 GiB/s
FFT N=96 complex128 forward+inverse pairs: about 334 pairs/s
FFT roundtrip max error: about 1.20e-15
```

Interpretation: the GTX 1080 remains the only working project GPU compute backend and is healthy for the current CuPy solver path.

## OpenCL Visibility

`clinfo` reports two OpenCL platforms:

1. AMD Accelerated Parallel Processing
2. NVIDIA CUDA

AMD device:

```text
Board name: Radeon RX 5500 XT
OpenCL C: 2.0
Global memory: 8573157376 bytes
Max allocation: 7059013632 bytes
Compute units: 11
Max clock: 1737 MHz
Extensions include cl_khr_fp64 and cl_khr_fp16
Compiler available: Yes
```

NVIDIA OpenCL device:

```text
Name: NVIDIA GeForce GTX 1080
OpenCL C: 1.2
Global memory: 8589803520 bytes
Max allocation: 2147450880 bytes
Compute units: 20
Max clock: 1809 MHz
Extensions include cl_khr_fp64
```

Interpretation: both devices have OpenCL visibility, and the AMD OpenCL stack appears installed. However, the active Python venv currently lacks `pyopencl`, so this is not yet usable from project diagnostics without adding a dependency.

## Vulkan Visibility

`vulkaninfo --summary` reports:

- GPU0: `Radeon RX 5500 XT`
- GPU1: `NVIDIA GeForce GTX 1080`

Interpretation: the AMD card is visible as a Vulkan device and appears first in Vulkan enumeration. This is promising for external visual/render tools or browser/WebGPU/Vulkan-oriented tooling, but the current project Python venv does not have a Vulkan binding installed.

## Missing Tooling

These commands were not found in PATH:

- `ffmpeg`
- `clpeak`
- `amd-smi`
- `rocm-smi`
- `rocminfo`

Implication: no video encoder benchmark, OpenCL kernel benchmark, or ROCm-style telemetry was available from current PATH.

## Integration Assessment

Recommended current split:

- GTX 1080: keep as the CUDA/CuPy simulation worker.
- Radeon RX 5500 XT: use as display / desktop / visualization render GPU.
- CPU: continue handling HDF5/NPZ loading, downsampling, reports, CSV/markdown summaries.

Near-term safe integration:

- Run solver and diagnostic math on the GTX 1080 only.
- Keep rendering and visual inspection read-only against saved artifacts.
- Use AMD for desktop/browser/visual tools where the OS chooses it.

Potential future integration, optional:

- Add a dedicated read-only visualization worker that consumes saved artifacts and renders PNG/HTML/video outputs.
- If AMD compute is desired later, evaluate a separate OpenCL path with `pyopencl` or an external OpenCL tool. This should not touch solver physics or production paths.
- If video rendering becomes important, install/test `ffmpeg` with AMD AMF and NVIDIA NVENC encoders separately, then decide which encoder should own MP4/GIF generation.

## Recommendation

Do not attempt to make the Radeon RX 5500 XT a simulation GPU for the current CuPy solver path.

Use it first as a render/display offload device. The useful next step is a small read-only visualization/render queue that can run while the GTX 1080 remains dedicated to CUDA simulation work.
