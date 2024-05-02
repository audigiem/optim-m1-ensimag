from lib.gd import gd_stepsize_start, gd_stepsize, gd_step
import pytest
import numpy as np


@pytest.mark.parametrize('n', [0, 10, 100])
@pytest.mark.parametrize('mu', [10., .1, 1.])
@pytest.mark.parametrize('L', [10., .1, 1.])
def test_start_lr(n, mu, L):
    assert 1e-6 < gd_stepsize_start(n, mu, L) < 1e5


@pytest.mark.parametrize("it", [0, 10, 10000])
@pytest.mark.parametrize("start", [1., 10., .1])
def test_lr(it, start):
    assert 0. < gd_stepsize(it, start) <= start


@pytest.mark.parametrize("start_point", [np.array(0.), np.array(1.), np.array(-2.)])
@pytest.mark.parametrize("grad", [
    # All functions have min in 0
    lambda x, _: 2.*x,  # gradient of x^2
    lambda x, _: np.exp(x) - np.exp(-x),  # gradient of e^x + e^-x
    lambda x, _: np.sign(x)  # ONE OF THE subgradients of |x|
])
@pytest.mark.parametrize('lr', [1e-4, 1e-6])
def test_step(start_point, grad, lr):
    assert np.abs(gd_step(start_point, 1, grad, lambda x, _: x, lr)) <= np.abs(start_point)
