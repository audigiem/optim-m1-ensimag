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
        """
        Vérifie si l'algorithme doit s'arrêter.
        - Arrêt si la norme du gradient est inférieure à un seuil `prec`
        - Arrêt également si on a atteint le nombre d'itérations maximales
        """
        # Si la norme du gradient est trop petite, on arrête
        if np.linalg.norm(g) < self.prec:
            print(f"Convergence atteinte à l'itération {it} : norme du gradient < {self.prec}")
            return True

        # Si le nombre d'itérations maximum est atteint, on arrête
        if it >= self.nsteps:
            print(f"Nombre d'itérations maximum atteint ({self.nsteps})")
            return True

        return False

    def update(self, f, g, h):
        """
        Met à jour le vecteur `x` selon la règle de la descente de gradient.
        x_{k+1} = x_k - alpha * g_k
        """
        # Mise à jour du vecteur x
        self.x = self.x - self.lr * g
        return self.x

    def l_search(self, dir):
        """
        Fonction de recherche linéaire de Wolfe-Armijo

        Args
        ----
        dir: ndarray
            direction de recherche
        """
        gamma = line_search(
            # ==== PUT CODE HERE ====
            # Premier argument: la référence de la fonction primale que
            #    l'on veut évaluer en un point
            self.oracle.primal,
            
            
            # ==== PUT CODE HERE ====
            # Deuxième argument: la fonction qui calcule le gradient en
            #    un point
            self.oracle.gradient,
            # ==== PUT CODE HERE ====
            # Troisème argument: le point d'évaluation de départ de la
            #    recherche linéaire.
            self.x,
            # ==== PUT CODE HERE ====
            # Quatrième argument: la direction de recherche
            dir,
            # Le reste: ne pas toucher.
            gfk=None, old_fval=None, old_old_fval=None, args=(),
            # Les hyperparamètres de Wolfe-Armijo.
            c1=self.m1,
            c2=self.m2,
            amax=self.ls_max
        )[0]
        return gamma
