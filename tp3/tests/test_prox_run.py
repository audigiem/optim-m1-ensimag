import numpy as np

from src.logistic import SimulatorStudentsDataset
from src.prox import Proximal


def test_run_cvg():
    sim = SimulatorStudentsDataset("")
    alg = Proximal(10000, sim, np.zeros(28), prec=1e-5, lr=0.03)

    retcode, _ = alg.run()
    assert retcode
    assert alg.stop(1000., np.zeros(28), None, 1)

def test_run_no_cvg():
    sim = SimulatorStudentsDataset("")
    alg = Proximal(10, sim, np.zeros(28), prec=1e-12, lr=0.03)

    retcode, _ = alg.run()
    assert not retcode
    alg.update(*sim.sim(alg.x))
    assert not alg.stop(1000., np.zeros(28), None, 1)

def test_run_stationarity():
    sim = SimulatorStudentsDataset("")
    alg = Proximal(10000, sim, np.zeros(28), prec=1e-5, lr=0.03)

    _, _ = alg.run()
    assert alg.last_x is not None
    assert np.allclose(alg.x, alg.last_x)

def test_run_finish_early():
    sim = SimulatorStudentsDataset("")
    alg = Proximal(10000, sim, np.zeros(28), prec=1e-3, lr=0.03)

    _, x_tab = alg.run()
    assert x_tab.shape[0] < 5000
