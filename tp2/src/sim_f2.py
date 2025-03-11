import numpy as np
from src.sim import Simulator


class SimF2(Simulator):
    def __init__(self, npts: int):
        super().__init__(
            2,
            npts,
            [(-1.5, 1.5), (-.2, 1.5)],
            0, 200,
            [1,2,5,10,20,30,50,100,200],
            'Rosenbrock',
            np.ones(2)
        )

    def sim(self, x: np.ndarray):
        assert x.shape == (self.n,)
        # ==== PUT CODE HERE ====
        raise NotImplementedError("=== put code here ===")
        f = ...
        g = ...
        return f, g, None

    def primal(self, x: np.ndarray):
        assert x.shape == (self.n,)
        raise NotImplementedError("=== put code here ===")
        return

    def gradient(self, x: np.ndarray):
        assert x.shape == (self.n,)
        raise NotImplementedError("=== put code here ===")
        return
