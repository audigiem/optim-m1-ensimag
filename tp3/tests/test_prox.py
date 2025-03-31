import numpy as np

from src.logistic import SimulatorStudentsDataset
from src.prox import Proximal


def test_stop_at_small_grad():
    sim = SimulatorStudentsDataset("")
    alg = Proximal(1, sim, np.zeros(28))
    g = np.zeros(28)
    alg.last_x = np.zeros(28)
    # Fake gradient step, no prox
    alg.x = alg.last_x - alg.lr * g

    assert alg.stop(1000., g, -np.eye(28), 1)

def test_stop_at_small_step():
    sim = SimulatorStudentsDataset("", lam1=1.0)
    alg = Proximal(1, sim, np.zeros(28), lr=1.0)
    g = np.ones(28)
    alg.last_x = np.zeros(28)
    alg.update(1000., g, -np.eye(28))

    assert alg.stop(1000., g, -np.eye(28), 1)

def test_no_stop_at_big_grad():
    sim = SimulatorStudentsDataset("")
    alg = Proximal(1, sim, np.zeros(28))
    g = 1000 * np.ones(28)
    alg.last_x = np.zeros(28)
    # Fake gradient step, no prox
    alg.x = alg.last_x - alg.lr * g

    assert not alg.stop(-1., g, 0. * np.eye(28), 10000)

def test_no_stop_at_big_step():
    sim = SimulatorStudentsDataset("", lam1=1.0)
    alg = Proximal(1, sim, np.zeros(28), lr=1.0)
    g = 1000 * np.ones(28)
    alg.last_x = np.ones(28)
    # Fake gradient step, no prox
    alg.x = alg.last_x - alg.lr * g

    assert not alg.stop(1000., g, -np.eye(28), 1)
