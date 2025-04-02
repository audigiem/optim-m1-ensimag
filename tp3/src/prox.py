import numpy as np
from src.simprox import SimulatorProx
from src.algos import Descent


class Proximal(Descent):
    def __init__(
        self,
        nsteps: int,
        oracle: SimulatorProx,
        start: np.ndarray,
        lr: float=1e-3,
        prec: float=1e-6
    ):
        super().__init__(nsteps, oracle, start)
        self.lr = lr
        self.prec = prec
        self.last_x = None

    def stop(self, f, g, h, it: int):
        del f, g, h, it
        # ===== INSERT CODE HERE =======
        raise NotImplementedError()

    def update(self, f, g, h):
        del f, h
        # ===== INSERT CODE HERE =======
        raise NotImplementedError()
