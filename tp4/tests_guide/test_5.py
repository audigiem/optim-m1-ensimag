from lib.saga import saga_stepsize_start, saga_stepsize, saga_step
import pytest
import numpy as np


@pytest.mark.parametrize('n', [0, 10, 100])
@pytest.mark.parametrize('mu', [10., .1, 1.])
@pytest.mark.parametrize('L', [10., .1, 1.])
def test_start_lr(n, mu, L):
    assert 1e-6 < saga_stepsize_start(n, mu, L) < 1e5


@pytest.mark.parametrize("it", [0, 10, 10000])
@pytest.mark.parametrize("start", [1., 10., .1])
def test_lr(it, start):
    assert 0. < saga_stepsize(it, start) <= start


@pytest.mark.parametrize("start_point", [
    np.array([[0.]]),
    np.array([[1.]]),
    np.array([[-2.]])
])
@pytest.mark.parametrize("grad", [
    # All functions have min in 0
    # SAGA should work on deterministic problems by
    # always going down as well.
    lambda x, i: 2.*x[i],  # gradient of x^2
    lambda x, i: np.exp(x[i]) - np.exp(-x[i]),  # gradient of e^x + e^-x
    lambda x, i: np.sign(x[i])  # ONE OF THE subgradients of |x|
])
@pytest.mark.parametrize('lr', [1e-4, 1e-6])
def test_step(start_point, grad, lr):
    gb = np.zeros((1, 1))
    dist_begin = np.abs(start_point).mean()
    dist_end = np.abs(
        saga_step(start_point, 1, grad, lambda x, _: x, lr, gb)[0]
    ).mean()
    assert dist_end <= dist_begin
