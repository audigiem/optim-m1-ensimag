import pytest
import numpy as np

from lib.linear_model import LogisticRegression


@pytest.mark.parametrize('l1', np.logspace(-3, -1, 4))
@pytest.mark.parametrize('lr', np.logspace(-4, -1, 5))
def test_prox_down(l1, lr):
    model = LogisticRegression('gd', l1, 0., 0)
    test_point = np.array([100 + l1 + lr])
    assert np.abs(model.prox(test_point, lr)) < np.abs(test_point)
    test_point = -test_point
    assert np.abs(model.prox(test_point, lr)) < np.abs(test_point)


@pytest.mark.parametrize('l1', np.logspace(-4, -1, 4))
@pytest.mark.parametrize('lr', np.logspace(-4, -1, 4))
def test_prox_squeeze(l1, lr):
    model = LogisticRegression('gd', l1, 0., 0)
    test_point = np.array([lr**2 + l1**2 - (l1 - lr)**2]) / 3.
    assert np.abs(model.prox(test_point, lr)) < 1e-10
    test_point = -test_point
    assert np.abs(model.prox(test_point, lr)) < 1e-10
