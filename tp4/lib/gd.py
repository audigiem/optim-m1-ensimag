"""
Functions specific to Gradient Descent
"""

from typing import Optional, Tuple, Callable
import numpy as np


def gd_stepsize_start(n: int, mu: float, L: float) -> float:
    # ####### TODO (2) ########
    raise NotImplementedError("TODO (2)")


def gd_stepsize(it: int, start: float) -> float:
    # ####### TODO (2) ########
    raise NotImplementedError("TODO (2)")


def gd_step(
    x: np.ndarray,
    grad: Callable[[np.ndarray, Optional[int]], np.ndarray],
    prox: Callable[[np.ndarray, float], np.ndarray],
    stepsize: float,
) -> Tuple[np.ndarray]:
    # ####### TODO (3) ########
    raise NotImplementedError("TODO (3)")
