from typing import Union

import numpy as np
from lp_solver.eta_matrix import EtaMatrix


def backward_transformation(B: Union[EtaMatrix, np.ndarray], Cb):
    if isinstance(B, EtaMatrix):
        B_inverted = B.invert()
        y = np.array(Cb)

        b_inv = np.reshape(B_inverted.column, (1, len(B_inverted.column)))
        Cb = np.reshape((1, len(Cb)))

        y[B_inverted.column_idx] = np.multiply(b_inv, Cb.T)

    # Temporary, until we'll implement LU decomposition
    else:
        y = np.multiply(Cb, np.linalg.inv(B))

    return y


def forward_transformation(B: Union[EtaMatrix, np.ndarray], a):
    if isinstance(B, EtaMatrix):
        B_inverted = B.invert()
        d = np.array([a[i] + B_inverted.column[i] * a[B.column_idx] for i in range(len(a))])
        d[B.column_idx] = B_inverted.column[B.column_idx] * a[B.column_idx]

    # Temporary, until we'll implement LU decomposition
    else:
        d = np.linalg.solve(B, a)

    return d
