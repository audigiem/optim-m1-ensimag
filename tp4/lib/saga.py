"""
Functions specific to SAGA
TODO (5)
"""

from typing import Optional, Tuple, Callable
import numpy as np


def initialize_gradients_buffer(n: int, d: int) -> np.ndarray:
    # ####### TODO (5) ########
    raise NotImplementedError("TODO (5)")


def saga_stepsize_start(n: int, mu: float, L: float) -> float:
    # ####### TODO (5) ########
    raise NotImplementedError("TODO (5)")


def saga_stepsize(it: int, start: float) -> float:
    # ####### TODO (5) ########
    raise NotImplementedError("TODO (5)")


def saga_step(
    x: np.ndarray,
    grad: Callable[[np.ndarray, Optional[int]], np.ndarray],
    prox: Callable[[np.ndarray, float], np.ndarray],
    stepsize: float,
    gradients_buffer: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # ####### TODO (5) ########
    raise NotImplementedError("TODO (5)")
