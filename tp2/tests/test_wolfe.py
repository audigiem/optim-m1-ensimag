import pytest
import numpy as np
from src.sim_f1 import SimF1
from src.sim_f2 import SimF2
from src.wolfe import WolfeLineSearch


GAMMA = 0.001
PREC = 1e-6

@pytest.mark.parametrize("n", range(1, 4))
def test_descent(n: int):
    start = np.zeros(n) + 1.0

    f1 = SimF1(n, 0)
    gd = WolfeLineSearch(1, f1, start, GAMMA)
    f, _, _ = gd.step()
    assert f1.primal(start) >= f

    if n == 2:
        f2 = SimF2(0)
        gd = WolfeLineSearch(1, f2, start, GAMMA)
        f, _, _ = gd.step()
        assert f2.primal(start) >= f

def test_cv():
    start = np.array([-1., 1.2])

    f2 = SimF2(0)
    gd = WolfeLineSearch(10000, f2, start, GAMMA, prec=PREC)
    ok, _ = gd.run()
    assert ok
    assert gd.vals.pop() < 1e-4
