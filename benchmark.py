
import statistics
import time

from mandelbrot_naive import mandelbrot_grid, mandelbrot_naive_numba
from mandelbrot_naive import benchmark_naive
from mandelbrot_numpy import mandelbrot_numpy



def bench ( fn , * args , runs =5) :
    fn (* args ) # extra warm - up
    times = []
    for _ in range ( runs ) :
        t0 = time . perf_counter ()
        fn (* args )
        times . append ( time . perf_counter () - t0 )
    return statistics . median ( times )

# Warm up ( triggers JIT c o m p i l a t i o n -- exclude from timing )
grid= mandelbrot_grid(width=64, height=64, max_iter=100)

# t_numba = mandelbrot_naive_numba ( -2 , 1 , -1.5 , 1.5 , 64 , 64)
# t_naive = mandelbrot_grid( -2 , 1 , -1.5 , 1.5 , 64 , 64)
# t_numpy = mandelbrot_numpy( -2 , 1 , -1.5 , 1.5 , 64 , 64)

t_full = bench ( mandelbrot_naive_numba , -2 , 1 , -1.5 , 1.5 , 1024 , 1024)
t_hybrid = bench ( mandelbrot_grid ,  -2 , 1 , -1.5 , 1.5 ,1024,1024)

args = ( -2 , 1 , -1.5 , 1.5 , 1024 , 1024)
t_naive = bench ( mandelbrot_grid , * args )
t_numpy = bench ( mandelbrot_numpy , * args )
t_numba = bench ( mandelbrot_naive_numba , * args )


# print(f"Hybrid : {t_hybrid:.3f} s")
# print(f"Fully compiled : {t_full:.3f} s")
# print(f"Ratio : {t_hybrid / t_full:.1f} x")

print(f"Naive : {t_naive:.3f} s")
print(f"NumPy : {t_numpy:.3f} s ({t_naive / t_numpy:.1f} x)")
print(f"Numba : {t_numba:.3f} s ({t_naive / t_numba:.1f} x)")





