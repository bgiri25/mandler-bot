
import statistics
import time

from mandelbrot_naive import mandelbrot_grid, mandelbrot_naive_numba
from mandelbrot_naive import benchmark_naive



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

_ = mandelbrot_naive_numba ( -2 , 1 , -1.5 , 1.5 , 64 , 64)
t,grid_ = benchmark_naive(width=64, height=64, max_iter=100)

t_full = bench ( mandelbrot_naive_numba , -2 , 1 , -1.5 , 1.5 , 1024 , 1024)
# t_hybrid = bench ( benchmark_naive , width=1024, height=1024, max_iter=100)


# print ( f " Hybrid : { t_hybrid :.3 f } s " )
# print ( f " Fully compiled : { t_full :.3 f } s " )
# print ( f " Ratio : { t_hybrid / t_full :.1 f } x " )





