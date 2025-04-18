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
