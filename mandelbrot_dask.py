from dask import delayed
from dask.distributed import Client, LocalCluster
import dask, numpy as np, time, statistics
from numba import njit

from mandelbrot_parallel import mandelbrot_chunk

@njit(cache=True)
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

@njit(cache=True)
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
# mandelbrot_chunk: your @njit(cache=True) function from L04/L05
def mandelbrot_dask(N, x_min, x_max, y_min, y_max,
max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(
            row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)

if __name__ == "__main__":
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, # warm up all workers
    Y_MIN, Y_MAX, 10))
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    print(f"Dask local (n_chunks=32): {statistics.median(times):.3f} s")
    client.close(); cluster.close()