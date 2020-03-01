import functools
from typing import Union, List, Callable

import numpy as np

from lp_solver.eta_matrix import EtaMatrix
from lp_solver.lp_program import LpProgram


def extract_legal_coefficients(rule: Callable[[Union[np.ndarray, List[float]], Union[np.ndarray, List[int]]], int]):
    @functools.wraps(rule)
    def _wrapper(coefficients: Union[np.ndarray, List[float]], variables: Union[np.ndarray, List[int]]) -> int:
        if len(coefficients) != len(variables):
            raise Exception('Non matching vars/coef size')

        if np.max(coefficients) <= 0:
            return -1

        filtered_coeffs, filtered_vars = zip(*(filter(lambda unit: unit[0] > 0, zip(coefficients, variables))))

        chosen_var = rule(filtered_coeffs, filtered_vars)

        if type(variables) == np.ndarray:
            variables = variables.tolist()

        return variables.index(chosen_var)

    return _wrapper


@extract_legal_coefficients
def blands_rule(_: Union[np.ndarray, List[float]], variables: Union[np.ndarray, List[int]]) -> int:
    return int(np.min(variables))


@extract_legal_coefficients
def dantzig_rule(coefficients: Union[np.ndarray, List[float]], variables: Union[np.ndarray, List[int]]) -> int:
    return variables[np.argmax(coefficients)]


def backward_transformation(B: Union[EtaMatrix, np.ndarray], Cb):
    if isinstance(B, EtaMatrix):
        B_inverted = B.invert()
        y = np.array(Cb)

        y[B_inverted.column_idx] = np.dot(B_inverted.column, Cb)

    # Temporary, until we'll implement LU decomposition
    else:
        y = np.dot(Cb, np.linalg.inv(B))

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


def get_entering_variable_idx(lp_program: LpProgram,
                              rule: Callable[[Union[np.ndarray, List[float]], Union[np.ndarray, List[int]]], int]):
    y = backward_transformation(lp_program.B, lp_program.Cb)

    coefs = lp_program.Cn - np.dot(y, lp_program.An)
    variables = lp_program.Xn

    return rule(coefs, variables)


def get_leaving_variable_idx(lp_program: LpProgram, entring_variable_idx: int):
    a = lp_program.An[:, entring_variable_idx]
    d = forward_transformation(lp_program.B, a)

    b = lp_program.b

    return np.argmin(b / d)
