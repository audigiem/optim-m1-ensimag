from typing import Callable, Tuple, Optional
import numpy as np

from .gd import gd_stepsize_start, gd_stepsize, gd_step
from .sgd import sgd_stepsize_start, sgd_stepsize, sgd_step
from .saga import (
    initialize_gradients_buffer,
    saga_stepsize_start,
    saga_stepsize,
    saga_step
)


def GD(
    x0: np.ndarray,
    grad: Callable[[np.ndarray, Optional[int]], np.ndarray],
    prox: Callable[[np.ndarray, float], np.ndarray],
    max_iter: int,
    n: int,
    L: float,
    mu: float
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Parameters
    ----------
    x0: array, shape (nb_features,)
        Initialisation of the solver
    grad: function
        Gradient of the objective function
    max_iter: int
        Number of iterations (i.e. number of descent steps). Note that for GD or SVRG,
        one iteration is one epoch i.e, one pass through the data, while for SGD, SAG and SAGA,
        one iteration uses only one data point.
    n: int
        Dataset size
    L: float
        Smoothness constant of the objective function
    mu: float
        Strong convexity constant of the objective function

    Returns
    -------
    x: array, shape (nb_features,)
        final iterate of the solver
    x_tab: array, shape (nb_features, max_iter)
        table of all the iterates
    '''
    step_start = gd_stepsize_start(n, mu, L)
    x = x0
    x_tab = [x]

    for k in range(max_iter):
        x, = gd_step(x, grad, prox, gd_stepsize(k, step_start))
        x_tab.append(x)

    return x, np.vstack(x_tab)


def SGD(
    x0: np.ndarray,
    grad: Callable[[np.ndarray, Optional[int]], np.ndarray],
    prox: Callable[[np.ndarray, float], np.ndarray],
    max_iter: int,
    n: int,
    L: float,
    mu: float
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Parameters
    ----------
    x0: array, shape (nb_features,)
        Initialisation of the solver
    grad: function
        Gradient of the objective function
    max_iter: int
        Number of iterations (i.e. number of descent steps). Note that for GD or SVRG,
        one iteration is one epoch i.e, one pass through the data, while for SGD, SAG and SAGA,
        one iteration uses only one data point.
    n: int
        Dataset size
    L: float
        Smoothness constant of the objective function
    mu: float
        Strong convexity constant of the objective function

    Returns
    -------
    x: array, shape (nb_features,)
        final iterate of the solver
    x_tab: array, shape (nb_features, max_iter)
        table of all the iterates
    '''
    stepsize_0 = sgd_stepsize_start(n, mu, L)
    x = x0
    x_tab = [x]

    for k in range(max_iter):
        stepsize = sgd_stepsize(k, stepsize_0)
        x, = sgd_step(x, grad, prox, stepsize)
        if k % n == 0:  # each completed epoch
            x_tab.append(x)

    return x, np.vstack(x_tab)


def SAGA(
    x0: np.ndarray,
    grad: Callable[[np.ndarray, Optional[int]], np.ndarray],
    prox: Callable[[np.ndarray, float], np.ndarray],
    max_iter: int,
    n: int,
    L: float,
    mu: float
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Parameters
    ----------
    x0: array, shape (nb_features,)
        Initialisation of the solver
    grad: function
        Gradient of the objective function
    max_iter: int
        Number of iterations (i.e. number of descent steps). Note that for GD or SVRG,
        one iteration is one epoch i.e, one pass through the data, while for SGD, SAG and SAGA,
        one iteration uses only one data point.
    n: int
        Dataset size
    L: float
        Smoothness constant of the objective function
    mu: float
        Strong convexity constant of the objective function

    Returns
    -------
    x: array, shape (nb_features,)
        final iterate of the solver
    x_tab: array, shape (nb_features, max_iter)
        table of all the iterates
    '''
    d = len(x0)
    alpha = initialize_gradients_buffer(n, d)
    stepsize_0 = saga_stepsize_start(n, mu, L)
    x = x0
    x_tab = [x]

    for k in range(max_iter):
        stepsize = saga_stepsize(k, stepsize_0)
        x, alpha = saga_step(x, grad, prox, stepsize, alpha)
        if k % n == 0:  # each completed epoch
            x_tab.append(x)

    return x, np.vstack(x_tab)
