"""
Functions specific to
Stochastic Gradient Descent
"""

from typing import Optional, Tuple, Callable
import numpy as np


def sgd_stepsize_start(
    n: int,
    mu: float,
    L: float
) -> float:
    """
    A function to choose the starting step-size of the algorithm, i.e. its step-size for the first iteration.

    Parameters
    ----------
    n: int
        number of samples of the data distribution
    mu: float
        strong-convexity constant of the objective function
    L: float
        (Approximate ?) Lipschitz constant of the gradient of the objective.
    """
    # ####### TODO (4) ########
    raise NotImplementedError("TODO (4)")


def sgd_stepsize(it: int, start: float) -> float:
    """
    A function to choose the step-size of the algorithm depending on the current iteration number.

    Parameters
    ----------
    it: int
        current iteration number
    start: float
        first-iteration step-size chosen
    """
    # ####### TODO (4) ########
    raise NotImplementedError("TODO (4)")


def sgd_step(
    x: np.ndarray,
    n: int,
    grad: Callable[[np.ndarray, Optional[int]], np.ndarray],
    prox: Callable[[np.ndarray, float], np.ndarray],
    stepsize: float,
) -> Tuple[np.ndarray]:
    """
    This function performs the step of this Stochastic Gradient Descent algorithm.
    Starting at ``x``, it outputs the next state of the algorithm as a 1-uple containing the next state.
    """
    # ####### TODO (4) ########
    raise NotImplementedError("TODO (4)")
