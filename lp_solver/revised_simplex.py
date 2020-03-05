import functools
from typing import Union, List, Callable, TYPE_CHECKING, Tuple, Set

import numpy as np

from lp_solver.eta_matrix import EtaMatrix
from lp_solver.UnboundedException import UnboundedException

EPSILON = 10 ** -4
if TYPE_CHECKING:
    pass


def extract_legal_coefficients(rule: Callable[[Union[np.ndarray, List[float]], Union[np.ndarray, List[int]]], int]):
    @functools.wraps(rule)
    def _wrapper(coefficients: Union[np.ndarray, List[float]], variables: Union[np.ndarray, List[int]],
                 bad_vars: Set[int]) -> int:

        if len(coefficients) != len(variables):
            raise Exception('Non matching vars/coef size')

        # Zero tolerance
        if np.max(coefficients) <= EPSILON and set(variables) == bad_vars:
            return -1

        # Remove bad variables: i.e. given or with negative coefficient
        # TODO: Sorry Yoni the zip(*filter()) does not work
        # filtered_coeffs, filtered_vars = zip(
        #     *(filter(lambda unit: unit[0] > 0 and unit[1] not in bad_vars, zip(coefficients, variables))))
        filtered_coeffs, filtered_vars = [], []
        for c, v in zip(coefficients, variables):
            if c > 0 and v not in bad_vars:
                filtered_coeffs.append(c)
                filtered_vars.append(v)
        if len(filtered_coeffs) == 0:
            return -1

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
    # TODO: Break tie in coefficients using smaller index (from variables)
    # TODO: add test
    return variables[np.argmax(coefficients)]


def backward_transformation(B: Union[EtaMatrix, np.ndarray], Cb: np.ndarray) -> np.ndarray:
    if isinstance(B, EtaMatrix):
        B_inverted = B.invert()
        y = np.array(Cb)

        y[B_inverted.column_idx] = np.dot(B_inverted.column, Cb)

    # Temporary, until we'll implement LU decomposition
    else:
        y = np.dot(Cb, np.linalg.inv(B))

    return y


def forward_transformation(B: Union[EtaMatrix, np.ndarray], a: np.ndarray) -> np.ndarray:
    if isinstance(B, EtaMatrix):
        B_inverted = B.invert()
        d = np.array([a[i] + B_inverted.column[i] * a[B.column_idx] for i in range(len(a))])
        d[B.column_idx] = B_inverted.column[B.column_idx] * a[B.column_idx]

    # Temporary, until we'll implement LU decomposition
    else:
        d = np.linalg.solve(B, a)

    return d


def get_entering_variable_idx(lp_program, bad_vars: Set[int]) -> int:
    y = lp_program.Cb
    for eta in lp_program.etas[::-1]:
        y = backward_transformation(eta, y)

    y_tag = backward_transformation(lp_program.B, lp_program.Cb)
    # np.testing.assert_almost_equal(y_tag, y)
    coefs = lp_program.Cn - np.dot(y, lp_program.An)
    variables = lp_program.Xn

    return lp_program.rule(coefs, variables, bad_vars)


def is_unbounded(leaving_var_coefficient: np.ndarray) -> bool:
    return all(leaving_var_coefficient < 0)


def get_leaving_variable_idx(lp_program, entering_variable_idx: int) -> Tuple[int, float, EtaMatrix]:
    a = lp_program.An[:, entering_variable_idx]

    d = np.copy(a)
    for eta in lp_program.etas:
        d = forward_transformation(eta, d)
    d_tag = forward_transformation(lp_program.B, a)
    # assert np.array_equal(d, d_tag)
    np.testing.assert_almost_equal(d, d_tag)
    if is_unbounded(d):
        raise UnboundedException()

    b = lp_program.b

    assert any(d != 0)
    d_copy = np.copy(d)
    d[d <= 0] = 1 / np.inf
    # (lecture 12 slide 21)
    t = b / d
    leaving_var = int(np.argmin(t))
    eta_d = EtaMatrix(d_copy, leaving_var)
    return leaving_var, t[leaving_var], eta_d

# def refactorization(B: np.ndarray, etas: List[EtaMatrix]) -> np.ndarray:
#     '''
#     Create a fresh basis out of the current basis and a list of transformations
#     :param B: previous basis
#     :param etas: list of transformations
#     :return: fresh basis
#     '''
#     for e in etas:
#         B = np.dot(B, e)
#
