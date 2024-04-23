import pytest
import numpy as np

from lib.linear_model import LogisticRegression


@pytest.mark.parametrize('l1', np.logspace(-3, -1, 10))
def test_l1_term(l1):
    model = LogisticRegression('gd', l1, .0, 0)
    # Fit:
    #    class(2*x) = -1
    #    class(-2*x) = 1
    # Leave no iteration to get loss
    model.fit(np.array([[2.], [-2.]]), np.array([-1, 1]))
    without_reg = np.log(1. + np.exp(2.))  # pen and paper
    assert model._empirical_risk(np.ones((1,))) > without_reg
