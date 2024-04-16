"""
Regularization functions
"""

import numpy as np


def reg_l1(x, const):
    # ######## TODO (6) ########
    raise NotImplementedError("TODO (6)")


def reg_l2(x, const):
    return .5 * const * np.sum(x * x)
