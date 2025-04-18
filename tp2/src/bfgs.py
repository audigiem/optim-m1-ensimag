from typing import Optional
import numpy as np
from src.sim import Simulator
from src.wolfe import WolfeLineSearch


class BFGSDescent(WolfeLineSearch):
    def __init__(
        self,
        nsteps: int,
        oracle: Simulator,
        start: np.ndarray,
        w0: Optional[np.ndarray]=None,
        m1: float=1e-4, m2: float=0.9,
        ls_max=50,
        prec: float=1e-6
    ):
        super().__init__(nsteps, oracle, start, m1, m2, ls_max, prec)
        self.w = np.eye(start.shape[0]) if w0 is None else w0

        self.setup()

    def setup(self):
        """
        Mise en place de l'algorithme: initialisation de s et y par un pas de
        gradient classique avec recherche linéaire.
        """
        x = np.copy(self.x)
        f, g1, h = self.oracle.sim(self.x)
        super().update(f, g1, h)

        _, g2, _ = self.oracle(self.x)
        self.y = g2 - g1
        self.s = self.x - x
        self.last_g = g1

        self.nsteps -= 1

    def update(self, f, g, h):
        """
        Cette fonction de maj n'est pas aussi simple que pour les autres
        algorithmes, on se permet de calculer le prochain gradient
        deux fois à l'avance.
        """
        del f, h
        x = np.copy(self.x)
        # Calculer la direction de recherche
        d = -self.w @ g
        # Calculer la taille du pas
        alpha = self.l_search(d)
        # Maj de x
        self.x = x + alpha * d

        self.last_g = g

        # Calculer le prochain gradient par un appel au simulateur
        _, g1, _ = self.oracle.sim(self.x)

        # Maj de s et y
        self.s = self.x - x
        self.y = g1 - self.last_g
        

        # ==== PUT CODE HERE ====
        # Mettre à jour W_k
        self.w = self.w + np.outer(self.s, self.s) / np.dot(self.s, self.y) - np.outer(self.w @ self.y, self.y @ self.w) / np.dot(self.y, self.w @ self.y)

        return self.x
