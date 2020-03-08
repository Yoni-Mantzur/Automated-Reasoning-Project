import functools
from typing import Union, List, Callable, Tuple, Set

import numpy as np

from lp_solver.UnboundedException import UnboundedException, InfeasibleException
from lp_solver.eta_matrix import EtaMatrix

EPSILON = 10 ** -4


def extract_legal_coefficients(rule: Callable[[Union[np.ndarray, List[float]], Union[np.ndarray, List[int]]], int]):
    @functools.wraps(rule)
    def _wrapper(coefficients: Union[np.ndarray, List[float]], variables: Union[np.ndarray, List[int]],
                 bad_vars: Set[int]) -> int:

        if len(coefficients) != len(variables):
            raise Exception('Non matching vars/coef size')

        # Zero tolerance
        if np.max(coefficients) <= EPSILON and set(variables) == bad_vars:
            # TODO: Remove
            print("OOPS")
            return -1

        # Remove bad variables: i.e. given or with negative coefficient
        # TODO: Sorry Yoni the zip(*filter()) does not work
        # filtered_coeffs, filtered_vars = zip(
        #     *(filter(lambda unit: unit[0] > 0 and unit[1] not in bad_vars, zip(coefficients, variables))))
        op = lambda x: x > 0
        filtered_coeffs, filtered_vars = [], []
        for c, v in zip(coefficients, variables):
            if op(c) and v not in bad_vars:
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
def blands_rule(_: Union[np.ndarray, List[float]],
                variables: Union[np.ndarray, List[int]]) -> int:
    return int(np.min(variables))


@extract_legal_coefficients
def dantzig_rule(coefficients: Union[np.ndarray, List[float]],
                 variables: Union[np.ndarray, List[int]]) -> int:
    if not isinstance(variables, np.ndarray):
        variables = np.array(variables)
    c = np.abs(coefficients)

    return int(np.min(variables[np.where(c == np.amax(c))]))


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
    for eta in lp_program.u_etas:
        y = forward_transformation(eta, y)

    for eta in lp_program.l_etas[::-1]:
        y = backward_transformation(eta, y)

    y = lp_program.permute_matrix(y, lp_program.p_inv)

    y_tag = backward_transformation(lp_program.B, lp_program.Cb)
    np.testing.assert_almost_equal(y_tag, y)

    coefs = lp_program.Cn - np.dot(y, lp_program.An)
    variables = lp_program.Xn

    return lp_program.rule(coefs, variables, bad_vars)


def is_unbounded(leaving_var_coefficient: np.ndarray, is_max: bool) -> bool:
    if is_max:
        return all(leaving_var_coefficient < 0)
    else:
        return all(leaving_var_coefficient > 0)


def calculate_entering_coefficient(lp_program, entering_idx):
    a = lp_program.An[:, entering_idx]
    d = np.copy(a)

    d = lp_program.permute_matrix(d, lp_program.p)
    for eta in lp_program.l_etas:
        d = forward_transformation(eta, d)
    for eta in lp_program.u_etas[::-1]:
        d = backward_transformation(eta, d)

    for eta in lp_program.etas:
        d = forward_transformation(eta, d)

    np.testing.assert_almost_equal(d, forward_transformation(lp_program.B, a))
    return d


def calculate_entering_bounds(lp, d) -> Tuple[float, int]:
    '''
    :param d: entering variable coefficients
    '''
    b = lp.b

    min_bound = 0
    max_bound = np.inf
    leaving_var = [-1, -1]

    assert len(d) == len(b)
    for i, (d1, b1) in enumerate(zip(d, b)):
        if d1 == 0:
            continue
        t = b1 / d1
        if d1 > 0 and b1 < 0:
            # The assignment must be less then zero, not possible
            raise InfeasibleException()
        elif d1 > 0 and b1 >= 0:
            if t < max_bound:
                leaving_var[1] = i
                max_bound = t
        elif d1 < 0 and b1 <= 0:
            if t > min_bound:
                leaving_var[0] = i
                min_bound = t
        elif d1 < 0 < b1:
            assert t < min_bound

    assert lp.is_aux or np.all(b >= 0)

    if min_bound > max_bound:
        raise InfeasibleException()
    elif lp.is_aux:
        # We are in a max problem but the coefficient of the objective is negative
        return min_bound, leaving_var[0]
    elif max_bound is np.inf:
        raise UnboundedException()
    return max_bound, leaving_var[1]


def get_leaving_variable_idx(lp_program, entering_idx: int) -> Tuple[int, float, EtaMatrix]:
    d = calculate_entering_coefficient(lp_program, entering_idx)

    # if is_unbounded(d, is_max=not lp_program.is_aux):
    #     raise UnboundedException()

    # b = lp_program.b
    # assert any(d != 0)
    # d_copy = np.copy(d)
    # d[d == 0] = 1 / np.inf
    #
    # # lecture 12 slide 21 (point 4): max t s.t. t * d <= b (i.e. if d <= 0 things get messy)
    # t = b / d
    # # d is minus so we negated the equation --> got opposite operation(<=/>=) need to re negate it
    # t[d < 0] = t[d < 0] * -1
    # leaving_var = int(np.argmin(t))
    # # t = b / d

    t, leaving_var = calculate_entering_bounds(lp_program, d)
    eta_d = EtaMatrix(d, leaving_var)
    return leaving_var, t, eta_d
