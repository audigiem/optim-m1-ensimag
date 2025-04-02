import numpy as np
from src.sim import Simulator
from collections import deque


class Descent:
    """
    Classe abstraite faisant office de squelette pour vos solvers.
    """
    def __init__(
        self,
        nsteps: int,
        oracle: Simulator,
        start: np.ndarray
    ):
        self.nsteps = nsteps
        self.oracle = oracle
        x0 = start
        self.steps = deque(maxlen=(nsteps + 1))
        self.steps.append(x0)
        self.vals = deque(maxlen=nsteps)
        self.x = np.copy(x0)

    def update(self, f, g, h):
        """
        Une fonction qui prend en paramètres les sorties du simulateur à
        l'itération courante et met à jour ``self.x``

        Args
        ----
        f: float
            Sortie primale du simulateur
        g: ndarray
            Gradient
        h: ndarray|None
            Optionellement une hessienne

        Returns
        -------
        x: ndarray
            Nouvel état du vecteur
        """
        del f, g, h
        raise NotImplementedError(
            "Implement the update method for this algorithm"
        )

    def stop(
        self,
        f: float,
        g: np.ndarray,
        h: np.ndarray,
        it: int
    ):
        """
        Une fonction qui prend en paramètres les sorties du simulateur à
        l'itération courante et décide si l'algorithme doit s'arrêter.

        Args
        ----
        f: float
            Sortie primale du simulateur
        g: ndarray
            Gradient
        h: ndarray|None
            Optionellement une hessienne

        Returns
        -------
        do_stop: bool
            Valeur logique de l'arrêt
        """
        del f, g, h, it
        raise NotImplementedError(
            "Implement the stopping criterion for this algorithm"
        )

    def step(self):
        """
        Lance le simulateur et met à jour le vecteur ``x``.

        Returns
        -------
        (f, g, h): tuple[float, ndarray, ndarray|None]
            Sorties du simulateur
        """
        # Appelle le simulateur
        f, g, h = self.oracle.sim(self.x)

        # Met à jour le vecteur
        x = self.update(f, g, h)
        self.steps.append(x)

        return f, g, h

    def run(self):
        """
        Réalise ``nsteps`` étapes de l'algorithme, et l'arrête si besoin.

        Returns
        -------
        (cvg, steps): tuple[bool, ndarray]
            convergence logical value and the stacked iterations
        """
        for it in range(self.nsteps):
            # Fait un pas de descente
            f, g, h = self.step()

            # Sauvegarde la valeur de la fonction pour monitorer sa descente
            self.vals.append(f)

            # Teste la condition d'arrêt
            if self.stop(f, g, h, it):
                msg = "FINISHED -- {} iterations -- final value: {}\n\n"
                print(msg.format(it, f))
                break
        else:
            # Si l'algo n'a pas convergé et arrive au bout
            #   des itérations, on affiche un message
            #   précisant qu'on l'a arrêté.
            print("STOPPED -- final value: {}\n\n".format(
                self.oracle.primal(self.x))
            )
            return False, np.vstack(list(self.steps))
        return True, np.vstack(list(self.steps))
