"""
Regularization functions
"""

import numpy as np


def reg_l1(
    x: np.ndarray,
    const: float
):
    """
    Computes the L1 regularization that one may add to the objective function.

    Arguments
    ---------
    x: np.ndarray
        current parameters
    const: float
        :math:`\\lambda_1` multiplicative constant.

    Returns
    -------
    l1: float
        the regularizing term
    """
    # ######## TODO (6) ########
    return 0.
    raise NotImplementedError("TODO (6)")


def reg_l2(
    x: np.ndarray,
    const: float
):
    """
    Computes the L2 regularization that one may add to the objective function.

    Arguments
    ---------
    x: np.ndarray
        current parameters
    const: float
        :math:`\\lambda_2` multiplicative constant.

    Returns
    -------
    l2: float
        the regularizing term
    """
    return .5 * const * np.sum(x * x)
