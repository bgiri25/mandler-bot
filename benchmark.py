"""
Mandelbrot Set Generator
Author : Bikash Giri
Course : Numerical Scientific Computing 2026
"""
import statistics
import time

from mandelbrot_dask import mandelbrot_dask
from mandelbrot_dask_strato import mandelbrot_dask_strato
from mandelbrot_naive import mandelbrot_grid, mandelbrot_naive_numba
from mandelbrot_naive import benchmark_naive
from mandelbrot_numba import mandelbrot_point_numba
from mandelbrot_numpy import mandelbrot_hybrid, mandelbrot_numpy
from mandelbrot_parallel import mandelbrot_parallel



def bench ( fn , * args , runs =5) :
    fn (* args ) # extra warm - up
    times = []
    for _ in range ( runs ) :
        t0 = time . perf_counter ()
        fn (* args )
        times . append ( time . perf_counter () - t0 )
    return statistics . median ( times )

# # Warm up ( triggers JIT c o m p i l a t i o n -- exclude from timing )
# # grid= mandelbrot_grid(width=64, height=64, max_iter=100)


# t_full = bench ( mandelbrot_point_numba , -2 , 1 , -1.5 , 1.5 , 1024 , 1024)
# args_parallel = (1024, -2, 1, -1.5, 1.5, 100, 8,16)

# args = ( -2 , 1 , -1.5 , 1.5 , 1024 , 1024)
# t_naive = bench ( mandelbrot_grid , * args )
# t_numpy = bench ( mandelbrot_numpy , * args )
# t_numba = bench ( mandelbrot_point_numba , * args )
# t_hybrid = bench ( mandelbrot_hybrid ,  -2 , 1 , -1.5 , 1.5 ,1024,1024)
# t_parallel = bench ( mandelbrot_parallel , * args_parallel )

# t_parallel = bench ( mandelbrot_parallel( 1024, -2, 1.0, -1.5, 1.5,100, n_workers=8, n_chunks=32)
# )

# t_parallel = bench ( mandelbrot_parallel , t_parallel )


# print(f"Hybrid : {t_hybrid:.3f} s")
# print(f"Fully compiled : {t_full:.3f} s")
# print(f"Ratio : {t_hybrid / t_full:.1f} x")


# print(f"Naive : {t_naive:.3f} s")
# print(f"NumPy : {t_numpy:.3f} s ({t_naive / t_numpy:.1f} x)")
# print(f"Numba : {t_numba:.3f} s ({t_naive / t_numba:.1f} x)")
# print(f"Parallel : {t_parallel:.3f} s ({t_naive / t_parallel:.1f} x)")


if __name__ == "__main__":
    n_workers = 8
    resolutions = [1024, 4096]
    results = {}

    for res in resolutions:
        print(f"\n--- Benchmarking {res}x{res} ---")
        width, height = res, res

        args = (-2, 1, -1.5, 1.5, width, height)
        args_parallel = (width, -2, 1, -1.5, 1.5, 100, n_workers, n_workers * 2)
        args_dask_local = (width, -2, 1, -1.5, 1.5, 100, 16) # 16 is the best n_chunks for dask local with 8 workers
        args_dask_strato = (width, -2, 1, -1.5, 1.5, 100, 32) # 32 is the best n_chunks for dask strato

        t_naive = bench ( mandelbrot_grid , * args )
        t_numpy = bench ( mandelbrot_numpy , * args )
        t_numba = bench ( mandelbrot_point_numba , * args )
        # t_hybrid = bench ( mandelbrot_hybrid ,  -2 , 1 , -1.5 , 1.5 ,1024,1024)
        t_parallel = bench ( mandelbrot_parallel , * args_parallel )
        t_dask = bench(mandelbrot_dask, *args_dask_strato)
        t_dask_strato = bench(mandelbrot_dask_strato, *args_dask_strato)



        results[res] = {
            "Naive": t_naive,
            "NumPy": t_numpy,
            # "Hybrid": t_hybrid,
            "Numba": t_numba,
            "Numba + Parallel": t_parallel,
            "Dask local": t_dask,
            "Dask strato": t_dask_strato,
        }

    # Build output string
    methods = list(next(iter(results.values())).keys())
    col_w = 18

    lines = []
    lines.append("===== BENCHMARK RESULTS (median seconds) =====")
    header = f"{'Method':<20}" + "".join(f"{f'{r}x{r}':>{col_w}}" for r in resolutions)
    lines.append(header)
    lines.append("-" * len(header))
    for method in methods:
        row = f"{method:<20}" + "".join(
            f"{results[r][method]:>{col_w}.3f}" for r in resolutions
        )
        lines.append(row)

    lines.append("\n===== SPEEDUP (relative to Naive) =====")
    lines.append(header)
    lines.append("-" * len(header))
    for method in methods:
        row = f"{method:<20}" + "".join(
            f"{results[r]['Naive'] / results[r][method]:>{col_w}.2f}x"
            for r in resolutions
        )
        lines.append(row)

    output = "\n".join(lines)

    # Print to console
    print("\n\n" + output)

    # Save to file
    with open("benchmark_results.txt", "w") as f:
        f.write(output)

    print("\nResults saved to benchmark_results.txt")



