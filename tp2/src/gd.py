import numpy as np
from src.algos import Descent
from src.sim import Simulator


class GradDescent(Descent):
    def __init__(
        self,
        nsteps: int,
        oracle: Simulator,
        start: np.ndarray,
        lr: float=1e-3,
        prec: float=1e-6
    ):
        super().__init__(nsteps, oracle, start)
        self.lr = lr
        self.prec = prec

    def stop(self, f, g, h, it: int):
        # ==== PUT CODE HERE ====
        # Decider si l'algo doit s'arrêter
        raise NotImplementedError("=== put code here ===")
        return True

    def update(self, f, g, h):
        # ==== PUT CODE HERE ====
        raise NotImplementedError("=== put code here ===")
        # Mettre à jour le vecteur x selon le gradient.
        return self.x
