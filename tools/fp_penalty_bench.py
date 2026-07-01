"""Quantify the FP64 vs FP32 cost on this GPU for the solver's hot ops (FFT + elementwise),
at 128^3. This is the core accuracy-vs-speed tradeoff: CuPy ETDRK4 runs in complex128."""
import time
import cupy as cp

N = 128
iters = 60


def bench(dtype, label):
    a = (cp.random.random((N, N, N)) + 1j * cp.random.random((N, N, N))).astype(dtype)
    k = cp.random.random((N, N, N)).astype(cp.float64).astype(a.real.dtype)
    # warmup + plan cache
    for _ in range(5):
        b = cp.fft.fftn(a)
        b *= k
        a = cp.fft.ifftn(b)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b = cp.fft.fftn(a)
        b *= k            # elementwise (dealias-like)
        a = cp.fft.ifftn(b)
    cp.cuda.Device().synchronize()
    dt = (time.perf_counter() - t0) / iters * 1e3
    print(f"{label:22s} {str(dtype):16s} {dt:7.2f} ms / (fft+mul+ifft)")
    return dt


print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  | N={N}^3, {iters} iters\n")
t32 = bench(cp.complex64, "FP32 (complex64)")
t64 = bench(cp.complex128, "FP64 (complex128)")
print(f"\nFP64 is {t64 / t32:.1f}x slower than FP32 on this card for the FFT-bound inner loop.")
