import numpy as np
from src.sim import Simulator


class SimF1(Simulator):
    def __init__(self, n: int, npts: int):
        super().__init__(
            n,
            npts,
            [(-5, 5)]*2,
            0, 30,
            [0.25,1,2,5,10,15],
            'f1',
            np.zeros(n)
        )

    def sim(self, x: np.ndarray):
        """
        Simuler f1
        """
        assert x.shape == (self.n,)

        # ==== PUT CODE HERE ====
        raise NotImplementedError("=== put code here ===")
        f = ...
        g = ...
        # Pour Newton, calculer en plus H la hessienne
        # H = ???
        # return f, g, h
        return f, g, None

    def primal(self, x: np.ndarray):
        assert x.shape == (self.n,)
        raise NotImplementedError("=== put code here ===")
        return

    def gradient(self, x: np.ndarray):
        assert x.shape == (self.n,)
        raise NotImplementedError("=== put code here ===")
        return

    def hessian(self, x: np.ndarray):
        assert x.shape == (self.n,)
        raise NotImplementedError("=== put code here ===")
        return
