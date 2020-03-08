from typing import NamedTuple

import numpy as np
import pytest
from lp_solver.UnboundedException import UnboundedException, InfeasibleException
from lp_solver.eta_matrix import EtaMatrix
from lp_solver.lp_program import LpProgram, calculate_entering_bounds
from lp_solver.revised_simplex import backward_transformation, forward_transformation, blands_rule, dantzig_rule, \
    get_entering_variable_idx, get_leaving_variable_idx

test_cases_btran = [[EtaMatrix([-4, 3, 2], 1), [1, 2, 3], [1, 0, 3]]]
test_cases_ftran = [[EtaMatrix([-4, 3, 2], 1), [1, 2, 3], [1 + 8 / 3, 2 / 3, 3 - 4 / 3]]]


@pytest.mark.parametrize(['B', 'c', 'expected_result'], test_cases_btran)
def test_backward_transformation(B, c, expected_result):
    assert_transformation(B, c, expected_result, backward_transformation)


@pytest.mark.parametrize(['B', 'a', 'expected_result'], test_cases_ftran)
def test_forward_transformation(B, a, expected_result):
    assert_transformation(B, a, expected_result, forward_transformation)


def assert_transformation(B, x, expected_result, transformation):
    actual_result_with_eta_matrix = transformation(B, x)
    actual_result_with_matrix = transformation(B.get_matrix(), x)

    assert np.allclose(actual_result_with_eta_matrix, actual_result_with_matrix)
    assert np.allclose(actual_result_with_eta_matrix, expected_result)


rules_test_cases = [[[0.5, 4, 3], [1, 4, 2], (0, 1)],
                    [[0.5, 0.5], [1, 5], (0, 0)],
                    [[0, 0.5], [5, 1], (1, 1)],
                    [[-0.5, -4, -3], [1, 4, 2], (-1, -1)]]


@pytest.mark.parametrize(['coefs', 'variables', 'expected_chosen_var'], rules_test_cases)
def test_blands_rule(coefs, variables, expected_chosen_var):
    assert expected_chosen_var[0] == blands_rule(coefs, variables, set())


@pytest.mark.parametrize(['coefs', 'variables', 'expected_chosen_var'], rules_test_cases)
def test_dantzig_rule(coefs, variables, expected_chosen_var):
    assert expected_chosen_var[1] == dantzig_rule(coefs, variables, set())


@pytest.mark.parametrize(['rule'], [['dantzig'], ['bland']])
def test_get_entering_variable_idx(rule):
    lp_program = LpProgram(rule=rule)

    # lp_program.etas = [EtaMatrix([3, 1, 0], 0), EtaMatrix([1, 1, 0], 1), EtaMatrix([4, 3, 1], 2)]

    lp_program.B = np.array([[3, 1, 0], [1, 1, 0], [4, 3, 1]])

    # In the entering_variable we iterate over the etas, for this test performance is neglected (i.e. using EtaMatrix)
    lp_program.l_etas = [lp_program.B]
    # lp_program.l_etas = lu(lp_program.B)

    lp_program.An = np.array([[2, 2, 1, 0], [1, 1, 0, 1], [3, 4, 0, 0]])
    lp_program.Xb = np.array([1, 3, 7])
    lp_program.Xn = np.array([2, 4, 5, 6])
    lp_program.Cb = np.array([19, 12, 0])
    lp_program.Cn = np.array([13, 17, 0, 0])

    lp_program.b = np.array([54, 63, 15])
    expected_var = 1

    assert expected_var == get_entering_variable_idx(lp_program, set())
    # assert expected_var == get_entering_variable_idx(lp_program, rule=blands_rule, set())


@pytest.mark.parametrize(['rule'], [['dantzig'], ['bland']])
def test_get_entering_variable_idx_refactorize(rule):
    lp_program = LpProgram(rule=rule)

    # lp_program.etas = [EtaMatrix([3, 1, 0], 0), EtaMatrix([1, 1, 0], 1), EtaMatrix([4, 3, 1], 2)]

    lp_program.B = np.array([[3, 1, 0], [1, 1, 0], [4, 3, 1]])

    # Create the eta matrices
    lp_program.refactorize()

    lp_program.An = np.array([[2, 2, 1, 0], [1, 1, 0, 1], [3, 4, 0, 0]])
    lp_program.Xb = np.array([1, 3, 7])
    lp_program.Xn = np.array([2, 4, 5, 6])
    lp_program.Cb = np.array([19, 12, 0])
    lp_program.Cn = np.array([13, 17, 0, 0])

    lp_program.b = np.array([54, 63, 15])
    expected_var = 1

    assert expected_var == get_entering_variable_idx(lp_program, set())
    # assert expected_var == get_entering_variable_idx(lp_program, rule=blands_rule, set())


@pytest.mark.parametrize(['rule'], [['dantzig'], ['bland']])
def test_get_leaving_variable_idx(rule):
    lp_program = LpProgram(rule=rule)

    # lp_program.etas = [EtaMatrix([3, 1, 0], 0), EtaMatrix([1, 1, 0], 1), EtaMatrix([4, 3, 1], 2)]

    lp_program.B = np.array([[3, 1, 0], [1, 1, 0], [4, 3, 1]])

    # In the entering_variable we iterate over the etas, for this test performance is neglected (i.e. using EtaMatrix)
    lp_program.l_etas = [lp_program.B]

    lp_program.An = np.array([[2, 2, 1, 0], [1, 1, 0, 1], [3, 4, 0, 0]])
    lp_program.Xb = np.array([1, 3, 7])
    lp_program.Xn = np.array([2, 4, 5, 6])
    lp_program.Cb = np.array([19, 12, 0])
    lp_program.Cn = np.array([13, 17, 0, 0])

    lp_program.b = np.array([54, 63, 15])
    expected_var = 2

    v_idx, t, d = get_leaving_variable_idx(lp_program, 1)
    assert expected_var == v_idx
    assert np.allclose(d.column, np.array([0.5, 0.5, 0.5]))
    assert t == 30


@pytest.mark.parametrize(['rule'], [['dantzig'], ['bland']])
def test_get_leaving_variable_idx_refactorize(rule):
    lp_program = LpProgram(rule=rule)

    lp_program.B = np.array([[3, 1, 0], [1, 1, 0], [4, 3, 1]])

    # Create the eta matrices
    lp_program.refactorize()

    lp_program.An = np.array([[2, 2, 1, 0], [1, 1, 0, 1], [3, 4, 0, 0]])
    lp_program.Xb = np.array([1, 3, 7])
    lp_program.Xn = np.array([2, 4, 5, 6])
    lp_program.Cb = np.array([19, 12, 0])
    lp_program.Cn = np.array([13, 17, 0, 0])

    lp_program.b = np.array([54, 63, 15])
    expected_var = 2

    v_idx, t, d = get_leaving_variable_idx(lp_program, 1)
    assert expected_var == v_idx
    assert np.allclose(d.column, np.array([0.5, 0.5, 0.5]))
    assert t == 30


def test_entering_variable_refactorize_and_eta():
    lp_program = LpProgram(rule='dantzig')

    lp_program.B = np.array([[3, 1, 0], [1, 1, 0], [4, 3, 1]])

    # Create the eta matrices
    lp_program.refactorize()

    lp_program.An = np.array([[2, 2, 1, 0], [1, 1, 0, 1], [3, 4, 0, 0]])
    lp_program.Xb = np.array([1, 3, 7])
    lp_program.Xn = np.array([2, 4, 5, 6])
    lp_program.Cb = np.array([19, 12, 0])
    lp_program.Cn = np.array([13, 17, 0, 0])

    lp_program.b = np.array([54, 63, 15])
    entering_idx = 1
    expected_var = 2

    v_idx, t, d = get_leaving_variable_idx(lp_program, entering_idx)
    assert expected_var == v_idx
    assert np.allclose(d.column, np.array([0.5, 0.5, 0.5]))
    assert t == 30
    lp_program.swap_basis(entering_idx, v_idx, d)
    lp_program.b -= (t * d.column).astype(np.int64)
    lp_program.b[v_idx] = t

    expected_var = -1
    assert expected_var == get_entering_variable_idx(lp_program, set())


def test_leaving_variable_refactorize_and_eta():
    lp_program = LpProgram(rule='dantzig')

    lp_program.B = np.array([[3, 1, 0], [1, 1, 0], [4, 3, 1]])

    # Create the eta matrices
    lp_program.refactorize()

    lp_program.An = np.array([[2, 2, 1, 0], [1, 1, 0, 1], [3, 4, 0, 0]])
    lp_program.Xb = np.array([1, 3, 7])
    lp_program.Xn = np.array([2, 4, 5, 6])
    lp_program.Cb = np.array([19, 12, 0])
    lp_program.Cn = np.array([13, 17, 0, 0])

    lp_program.b = np.array([54, 63, 15])
    entering_idx = 1
    expected_var = 2

    v_idx, t, d = get_leaving_variable_idx(lp_program, entering_idx)
    assert expected_var == v_idx
    assert np.allclose(d.column, np.array([0.5, 0.5, 0.5]))
    assert t == 30
    lp_program.swap_basis(entering_idx, v_idx, d)
    lp_program.b -= (t * d.column).astype(np.int64)
    lp_program.b[v_idx] = t

    expected_var = 2
    expected_d = np.array([-1., -1., 2.])
    expected_t = 15
    # We don't really care about the entering variable for the test, just use 4
    v_idx, t, d = get_leaving_variable_idx(lp_program, 1)
    assert expected_var == v_idx
    assert np.allclose(expected_d, d.column)
    assert expected_t == t


@pytest.mark.parametrize(['b', 'd', 'is_aux', 'e_t', 'e_var'],
                         [[np.array([1, 2]), np.array([1, 4]), False, 0.5, 1],
                          [np.array([1, 2]), np.array([-1, -4]), False, UnboundedException(), None],
                          [np.array([1, 2]), np.array([1, -4]), False, 1, 0],
                          [np.array([-1, -2]), np.array([-1, -1]), True, 2, 1],
                          [np.array([10, -2]), np.array([-1, -1]), True, 2, 1],
                          [np.array([2, -5]), np.array([1, -1]), True, InfeasibleException(), None],
                          [np.array([-2, -5]), np.array([1, 1]), True, InfeasibleException(), None],
                          [np.array([1, 2]), np.array([0, 4]), False, 0.5, 1],
                          [np.array([1, 2]), np.array([0, 0]), False, UnboundedException(), None],
                          ])
def test_calculate_entering_bounds(b, d, is_aux, e_t, e_var):
    lp = NamedTuple('lp', [('b', np.ndarray), ('is_aux', bool)])
    lp.b = b
    lp.is_aux = is_aux

    if isinstance(e_t, Exception):
        with pytest.raises(type(e_t)):
            calculate_entering_bounds(lp, d)
    else:
        t, v_idx = calculate_entering_bounds(lp, d)
        assert t == e_t
        assert v_idx == e_var
