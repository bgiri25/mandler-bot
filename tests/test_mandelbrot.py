import numpy as np
import numpy.testing as npt
import pytest

from mandelbrot_naive import mandelbrot_point, mandelbrot_grid, linspace
from mandelbrot_numpy import mandelbrot_numpy


def test_mandelbrot_point_zero_returns_max_iter():
    """c == 0 is inside the set; iterations should reach max_iter."""
    max_it = 50
    assert mandelbrot_point(0 + 0j, max_iter=max_it) == max_it



def test_mandelbrot_grid_shape_and_values():
    """mandelbrot_grid should return the expected shape and values computed
    by mandelbrot_point for the same coordinates."""
    xmin, xmax, ymin, ymax = -1.0, 0.0, -0.5, 0.5
    width, height, max_iter = 5, 4, 30
    grid = mandelbrot_grid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                           width=width, height=height, max_iter=max_iter)

    # shape checks
    assert len(grid) == height
    assert len(grid[0]) == width


    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)
    for j in (0, height // 2, height - 1):
        for i in (0, width // 2, width - 1):
            c = xs[i] + 1j * ys[j]
            expected = mandelbrot_point(c, max_iter=max_iter)
            assert grid[j][i] == expected


def test_mandelbrot_numpy_matches_naive():
    """Compare small-grid outputs from the NumPy implementation and the
    pure-Python naive implementation (values should match)."""
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.0, 1.0
    width, height, max_iter = 10, 8, 40

    naive = mandelbrot_grid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                            width=width, height=height, max_iter=max_iter)
    np_res = mandelbrot_numpy(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                              width=width, height=height, max_iter=max_iter)

    # Convert naive (list-of-lists) to numpy array for direct comparison
    naive_arr = np.array(naive, dtype=np.int32)
    npt.assert_array_equal(naive_arr, np_res)


@pytest.mark.parametrize(
    "width,height,max_iter",
    [
        (10, 8, 40),
        (6, 6, 20),
        (4, 3, 10),
    ],
)
def test_numpy_matches_naive_param(width, height, max_iter):
    """Parametrized: compare numpy and naive implementations across sizes."""
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.0, 1.0

    naive = mandelbrot_grid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                            width=width, height=height, max_iter=max_iter)
    np_res = mandelbrot_numpy(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                              width=width, height=height, max_iter=max_iter)

    naive_arr = np.array(naive, dtype=np.int32)
    npt.assert_array_equal(naive_arr, np_res)