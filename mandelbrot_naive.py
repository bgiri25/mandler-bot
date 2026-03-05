"""
Mandelbrot Set Generator
Author : Bikash Giri
Course : Numerical Scientific Computing 2026
"""

"""
Naive Mandelbrot set generator without using NumPy.

This mirrors the functionality of mandelbrot.py, but uses only
pure Python types (lists of lists) instead of NumPy arrays.
"""

import time
import matplotlib.pyplot as plt
from memory_profiler import profile
from numba import jit, njit 
import numpy as np

import statistics





@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
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



def mandelbrot_point(c: complex, max_iter: int = 80) -> int:
    """Iteration count for a single complex number."""
    z = 0 + 0j
    for n in range(max_iter):
        
        z = z*z + c
        if (abs(z) > 2):
            return n
    return max_iter


def linspace(start: float, stop: float, num: int) -> list[float]:
    """Simple replacement for numpy.linspace using pure Python."""
    if num == 1:
        return [float(start)]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def mandelbrot_grid(
    xmin: float = -2.0,
    xmax: float = 1.0,
    ymin: float = -1.5,
    ymax: float = 1.5,
    width: int = 100,
    height: int = 100,
    max_iter: int = 80,
) -> list[list[int]]:
    """
    Compute Mandelbrot set on a grid using nested loops (no NumPy).

    Returns
    -------
    list[list[int]]
        2D list of iteration counts with shape [height][width].
    """
    xs = linspace(xmin, xmax, width)
    ys = linspace(ymin, ymax, height)

    # 2D list for iteration counts (rows = height, cols = width)
    iters: list[list[int]] = [[0 for _ in range(width)] for _ in range(height)]

    for j in range(height):
        for i in range(width):
            c = xs[i] + 1j * ys[j]
            iters[j][i] = mandelbrot_point(c, max_iter=max_iter)

    return iters


def plot_mandelbrot(
    grid: list[list[int]],
    xmin: float = -2.0,
    xmax: float = 1.0,
    ymin: float = -1.5,
    ymax: float = 1.5,
    cmap: str = "hot",
    filename: str | None = None,
) -> None:
    """
    Visualize the Mandelbrot iterations using imshow.

    Note: matplotlib will internally convert the list-of-lists to an array,
    but this file itself does not import or use NumPy.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(
        grid,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
    )
    plt.colorbar(label="Iteration count")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.title(f"Mandelbrot set (cmap={cmap})")

    plt.show()

    
@profile
def benchmark_naive(width,height,max_itr):
        print(f"Computing Mandelbrot grid {width}x{height} (no NumPy) ...")
        t0 = time.perf_counter()
        grid = mandelbrot_grid(width=width, height=height, max_iter=max_itr)
        perfomance_list = []
        # Compute max iteration by scanning the list-of-lists
        for _ in range(3):
            t1 = time.perf_counter()
            elapsed = (t1 - t0) 
            perfomance_list.append(elapsed)
        median_time = statistics.median(perfomance_list)
        print(f"Computation took {elapsed:.3f} seconds for naive implementation")     
        return median_time,grid


if __name__ == "__main__":
    max_iter = 80
    width, height = 600, 600

    t,grid = benchmark_naive(width,height,max_iter)
    _ = mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 64, 64)



    for cmap in ["hot"]:
        plot_mandelbrot(
            grid,
            xmin=-2.0,
            xmax=1.0,
            ymin=-1.5,
            ymax=1.5,
            cmap=cmap,
            filename=f"mandelbrot_no_numpy_{cmap}.png",
        )

    

