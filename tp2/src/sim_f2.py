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
        """
        Simuler f2 = 100*(x2-x1^2)^2 + (1-x1)^2
        :param x: np.ndarray valeur varaible pour évaluer f2
        :return: float f2(x), np.ndarray gradient de f2(x)
        """
        assert x.shape == (self.n,)
        # f(x) = 100*(x2-x1^2)^2 + (1-x1)^2
        f = 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
        # gradient de f en x
        g = np.array([
            -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]),
            200*(x[1]-x[0]**2)
        ])
        
        return f, g, None

    def primal(self, x: np.ndarray):
        assert x.shape == (self.n,)
        return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

    def gradient(self, x: np.ndarray):
        assert x.shape == (self.n,)
        return np.array([
            -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]),
            200*(x[1]-x[0]**2)
        ])
