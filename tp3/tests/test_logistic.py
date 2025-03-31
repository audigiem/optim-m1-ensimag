import numpy as np

from src.logistic import SimulatorStudentsDataset


def test_l1_reg_rtype():
    sim = SimulatorStudentsDataset("")
    assert isinstance(sim.l1_regularization(np.zeros(28)), float)

def test_l1_reg_values():
    sim = SimulatorStudentsDataset("")
    assert sim.l1_regularization(np.zeros(28)) == 0.
    assert sim.l1_regularization(np.ones(28)) == .84

    sim = SimulatorStudentsDataset("", lam1=1.)
    assert sim.l1_regularization(np.ones(28)) == 28.

def test_prox_rtype():
    sim = SimulatorStudentsDataset("", lam1=1.)
    assert isinstance(sim.g_prox(np.ones(28), 1.0), np.ndarray)

def test_prox_values():
    sim = SimulatorStudentsDataset("", lam1=1.)
    assert np.allclose(sim.g_prox(np.ones(28), 1.0), np.zeros(28))
    assert np.allclose(sim.g_prox(np.ones(28), 2.0), np.zeros(28))
    assert np.allclose(sim.g_prox(2*np.ones(28), 1.0), np.ones(28))

    sim = SimulatorStudentsDataset("", lam1=.5)
    assert np.allclose(sim.g_prox(np.ones(28), 2.0), np.zeros(28))
    assert np.allclose(sim.g_prox(2*np.ones(28), 1.0), 1.5*np.ones(28))
