import numpy as np
from src.gd import GradDescent
from src.sim import Simulator
from scipy.optimize import line_search


class WolfeLineSearch(GradDescent):
    def __init__(
        self,
        nsteps: int,
        oracle: Simulator,
        start: np.ndarray,
        m1: float=1e-4, m2: float=0.9,
        ls_max=50,
        prec: float=1e-6
    ):
        super().__init__(nsteps, oracle, start)
        self.m1 = m1
        self.m2 = m2
        self.ls_max = ls_max
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

    def l_search(self, dir):
        """
        Fonction de recherche linéaire de Wolfe-Armijo

        Args
        ----
        dir: ndarray
            direction de recherche
        """
        # ==== PUT CODE HERE ====
        raise NotImplementedError("=== put code here ===")
        gamma = line_search(
            # ==== PUT CODE HERE ====
            # Premier argument: la référence de la fonction primale que
            #    l'on veut évaluer en un point
            None,
            # ==== PUT CODE HERE ====
            # Deuxième argument: la fonction qui calcule le gradient en
            #    un point
            None,
            # ==== PUT CODE HERE ====
            # Troisème argument: le point d'évaluation de départ de la
            #    recherche linéaire.
            None,
            # ==== PUT CODE HERE ====
            # Quatrième argument: la direction de recherche
            None,
            # Le reste: ne pas toucher.
            gfk=None, old_fval=None, old_old_fval=None, args=(),
            # Les hyperparamètres de Wolfe-Armijo.
            c1=self.m1,
            c2=self.m2,
            amax=self.ls_max
        )[0]
        return gamma
