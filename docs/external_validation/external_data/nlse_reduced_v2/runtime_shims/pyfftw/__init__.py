import numpy as np

simd_alignment = 16

class _Config:
    NUM_THREADS = 1
    PLANNER_EFFORT = "FFTW_MEASURE"

config = _Config()

class _Cache:
    @staticmethod
    def enable():
        return None

class _Interfaces:
    cache = _Cache()

interfaces = _Interfaces()

def zeros_aligned(shape, dtype=complex, n=None):
    return np.zeros(shape, dtype=dtype)

def empty_aligned(shape, dtype=complex, n=None):
    return np.empty(shape, dtype=dtype)

def import_wisdom(wisdom):
    return None

def export_wisdom():
    return ()

class FFTW:
    def __init__(self, input_array, output_array, direction, threads=1, axes=None):
        self.direction = direction
        self.axes = axes

    def __call__(self, input_array=None, output_array=None, normalise_idft=False):
        if self.direction == "FFTW_FORWARD":
            out = np.fft.fftn(input_array, axes=self.axes)
        else:
            out = np.fft.ifftn(input_array, axes=self.axes)
        output_array[...] = out
        return output_array
