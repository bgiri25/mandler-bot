
from mandelbrot_naive import mandelbrot_naive_numba


# _ = mandelbrot_(-2, 1, -1.5, 1.5, 64, 64)
_ = mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 64, 64)