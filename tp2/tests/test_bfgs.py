import pytest
import numpy as np
from src.sim_f1 import SimF1
from src.sim_f2 import SimF2
from src.bfgs import BFGSDescent

PREC = 1e-8

@pytest.mark.parametrize("n", range(1, 4))
def test_descent(n: int):
    start = np.zeros(n) + 1

    f1 = SimF1(n, 0)
    gd = BFGSDescent(1, f1, start)
    f, _, _ = gd.step()
    assert f1.primal(start) >= f

    if n == 2:
        f2 = SimF2(0)
        gd = BFGSDescent(1, f2, start)
        f, _, _ = gd.step()
        assert f2.primal(start) >= f

def test_cv():
    start = np.array([-1., 1.2])

    f2 = SimF2(0)
    gd = BFGSDescent(10000, f2, start, prec=PREC)
    ok, _ = gd.run()
    assert ok
    assert gd.vals.pop() < 1e-6
