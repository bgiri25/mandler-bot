# mandelbrot_parallel.py (Tasks 1-3 are one continuous script)
import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics
import matplotlib.pyplot as plt

@njit(cache = True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0
    for i in range(max_iter):
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag
        if z_real_sq + z_imag_sq > 4.0:
            return i
        # compute new values using previous z_real and z_imag
        z_imag_new = 2.0 * z_real * z_imag + c_imag
        z_real_new = z_real_sq - z_imag_sq + c_real
        z_real = z_real_new
        z_imag = z_imag_new
    return max_iter

@njit(cache = True)
def mandelbrot_chunk(row_start, row_end, N,
                     x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    # Use (N-1) so the grid includes both endpoints at col=0 and col=N-1
    dx = (x_max - x_min) / (N - 1) if N > 1 else 0.0
    dy = (y_max - y_min) / (N - 1) if N > 1 else 0.0
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)
    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)



def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max,
                        max_iter=100, n_workers=4,n_chunks=None):

    chunk_size = n_chunks if n_chunks is not None else max(1, N // n_workers)
    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)  # un-timed warm-up: Numba JIT in workers
        parts = pool.map(_worker, chunks)
    return np.vstack(parts)


def plot_mandelbrot(grid: np.ndarray, x_min: float, x_max: float,
                    y_min: float, y_max: float, cmap: str = "hot",
                    filename: str | None = None) -> None:
    """Display (and optionally save) the Mandelbrot iteration grid.

    Parameters
    - grid: 2D array with shape (Nrows, Ncols) containing iteration counts
    - x_min, x_max, y_min, y_max: plot extent in the complex plane
    - cmap: matplotlib colormap name
    - filename: if provided, save the figure to this path
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, extent=[x_min, x_max, y_min, y_max], origin="lower",
               cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Iteration count")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.title("Mandelbrot set")
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    chunk_size = 32
    t0 = time.perf_counter() 
    result = mandelbrot_parallel(1024, -2.5, 1.0, -1.25, 1.25, n_workers=8, n_chunks=chunk_size)
    t1 = time.perf_counter() - t0
    print(f"Computed Mandelbrot set in {t1:.3f} for {chunk_size} chunks")   
    # show and save a quick plot of the computed Mandelbrot set
    # plot_mandelbrot(result, -2.5, 1.0, -1.25, 1.25, filename="mandelbrot_parallel.png")
    # --- MP2 M3: benchmark (in __main__ block) ---
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    # Serial baseline (Numba already warm after M1 warm-up)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    for n_workers in range(1, os.cpu_count() + 1):
        chunk_size = max(1, N // n_workers)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks)  # warm-up: Numba JIT in all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
    t_par = statistics.median(times)
    speedup = t_serial / t_par
    print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")
