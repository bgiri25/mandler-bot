"""
Mandelbrot Set Generator
Author : Bikash Giri
Course : Numerical Scientific Computing 2026
"""

from numba import njit
import numpy as np, time
import matplotlib.pyplot as plt



@njit
def mandelbrot_point(c: complex, max_iter: int = 80) -> int:
    """Iteration count for a single complex number."""
    z = 0 + 0j
    for n in range(max_iter):
        
        if (z.real * z.real + z.imag * z.imag) > 2.0:
            return n
        z = z * z + c
    return max_iter

@njit
def mandelbrot_point_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    """Fully JIT-compiled Mandelbrot (Numba).

    Parameters are the same as the pure-Python version. Returns a
    (height, width) int32 NumPy array with iteration counts.
    """
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):  # compiled loop
        for j in range(width):  # compiled loop
            c = x[j] + 1j * y[i]
            z = 0 + 0j
            n = 0
            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
                z = z * z + c
                n += 1
            result[i, j] = n

    return result


@njit
def mandelbrot_numba_typed(xmin, xmax, ymin, ymax,
                           width, height, max_iter=100, dtype=np.float64):
    x = np.linspace(xmin, xmax, width).astype(dtype)
    y = np.linspace(ymin, ymax, height).astype(dtype)
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            result[i, j] = mandelbrot_point(c, max_iter)
    return result


for dtype in [np.float32, np.float64]:
    t0 = time.perf_counter()
    mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=dtype)
    print(f"{dtype.__name__}: {time.perf_counter() - t0:.3f} s")
r32 = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float32)
r64 = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float64)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, result, title in zip(axes, [ r32, r64], ['float32', 'float64 (ref)']):
    ax.imshow(result, cmap='hot')
    ax.set_title(title)
    ax.axis('off')

plt.savefig('precision_comparison.png', dpi=150)
print(f"Max diff float32 vs float64 : {np.abs(r32 - r64).max()}")


