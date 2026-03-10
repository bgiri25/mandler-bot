"""
Mandelbrot Set Generator
Author : Bikash Giri
Course : Numerical Scientific Computing 2026
"""

import cProfile
import pstats

from mandelbrot_naive import benchmark_naive
from mandelbrot_numpy import benchmark_numpy
import numpy as np


def profile_benchmarks():
    """Run profilers for the benchmark functions and print top cumulative stats.

    Uses small grid sizes so this runs quickly when executed.
    """
    cProfile.run("benchmark_naive(80,80,80)", "naive_profile.prof")
    cProfile.run("benchmark_numpy(80,80,80)", "numpy_profile.prof")

    for name in ("naive_profile.prof", "numpy_profile.prof"):
        stats = pstats.Stats(name)
        print(f"Profile results for {name}:")

        stats.sort_stats("cumulative")
        stats.print_stats(10)


if __name__ == "__main__":
    profile_benchmarks()


def mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    """Generate a Mandelbrot set using a straightforward numpy-backed loop.

    This is kept here only for convenience; the project also provides
    `benchmark_naive` in `mandelbrot_naive.py`.
    """
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=int)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0 + 0j
            for n in range(max_iter):
                if abs(z) > 2:
                    result[i, j] = n
                    break
                z = z * z + c
            else:
                result[i, j] = max_iter

    # Example usage (uncomment to run):
    # img = mandelbrot_naive(-2.0, 1.0, -1.0, 1.0, 400, 400)
    # print(img.shape)
    return result