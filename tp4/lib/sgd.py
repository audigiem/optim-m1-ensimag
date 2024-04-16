"""
Functions specific to
Stochastic Gradient Descent
"""

from typing import Optional, Tuple, Callable
import numpy as np


def sgd_stepsize_start(n: int, mu: float, L: float) -> float:
    # ####### TODO (4) ########
    raise NotImplementedError("TODO (4)")


def sgd_stepsize(it: int, start: float) -> float:
    # ####### TODO (4) ########
    raise NotImplementedError("TODO (4)")


def sgd_step(
    x: np.ndarray,
    grad: Callable[[np.ndarray, Optional[int]], np.ndarray],
    prox: Callable[[np.ndarray, float], np.ndarray],
    stepsize: float,
) -> Tuple[np.ndarray]:
    # ####### TODO (4) ########
    raise NotImplementedError("TODO (4)")
