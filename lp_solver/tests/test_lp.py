from functools import reduce

import numpy as np
import pytest

from lp_solver.lp_program import LpProgram, EtaMatrix


def test_initialize():
    # matrix_str = ["4x1,5x2,-6x3>=1", "4x3,5x0>=2", "3.1x1,1.1x2>=0"]
    matrix_str = ["4x1,5x2,-6x3<=1", "4x3,5x0<=2", "3.1x1,1.1x2<=0"]
    objective_str = "2x1,-3x2,0x3"
    non_basic_variables = [0, 1, 2, 3]
    matrix = np.array([
        [0, 4, 5, -6],
        [5, 0, 0, 4],
        [0, 3.1, 1.1, 0],
    ])

    lp = LpProgram(matrix_str, objective_str)
    assert np.array_equal(lp.An, matrix)
    assert np.array_equal(lp.b, [1, 2, 0])
    assert sorted(lp.Xn) == sorted(non_basic_variables)

    basic_variables = [4, 5, 6]
    basic_matrix = np.eye(len(basic_variables))
    assert sorted(lp.Xb) == sorted(basic_variables)
    assert np.array_equal(lp.B, basic_matrix)

    objective_non_basic = np.array([0, 2, -3, 0])
    objective_basic = np.zeros(shape=len(basic_variables))
    assert np.array_equal(lp.Cb, objective_basic)
    assert np.array_equal(lp.Cn, objective_non_basic)
    # lp.dump()
    # print(lp)


# def test_simple_lp():
#     objective = '5x1,4x2,3x3'
#     constraints = ['2x1,3x2,x3<=5', '4x1,x2,2x3<=11', '3x1,4x2,2x3<=8']
#     lp = LpProgram(constraints, objective, rule='Dantzig')
#     assert lp.solve == 13

@pytest.mark.parametrize('rule', ['dantzig', 'bland'])
def test_auxiliry_lp(rule):
    # HW3 Q1
    objective = 'x1,3x2'
    constraints = ['-1x1,x2<=-1', '-2x1,-2x2<=-6', '-1x1,4x2<=2']
    lp = LpProgram(constraints, objective, rule=rule)
    assert lp.solve() == np.inf


@pytest.mark.parametrize('rule', ['bland', 'dantzig'])
def test_simple_lp(rule):
    # Lecture 11 Slide 19
    objective = '5x1,4x2,3x3'
    constraints = ['0x0,2x1,3x2,x3<=5', '4x1,x2,2x3<=11', '3x1,4x2,2x3<=8']
    lp = LpProgram(constraints, objective, rule=rule)
    assert lp.solve() == 13


def test_simple_lp3():
    # Lecture 11 Slide 19 + some
    objective = '5x1,4x2,3x3,1x4'
    constraints = ['0x0,2x1,3x2,x3,0.1x4<=5', '4x1,x2,2x3<=11', '3x1,4x2,2x3<=8']
    lp = LpProgram(constraints, objective, rule='bland')
    assert lp.solve() > 13


def test_simple_lp2():
    objective = '5x1,4x2,3.1x3'
    constraints = ['0x0,2x1,3x2,x3<=5', '4x1,x2,2x3<=11', '3x1,4x2,2x3<=8']
    lp = LpProgram(constraints, objective, rule='bland')
    lp.solve()


Bs = [np.array([[2, 0, 0], [4, 1, 0], [3, 0, 1]])]
Ps = [[2, 1, 0]]
etas = [[EtaMatrix([1, 0.75, 0.5], 0), EtaMatrix([0, 1, 2 / 3], 1), EtaMatrix([4, 1, 0], 0),\
         EtaMatrix([0, -0.75, 1], 1), EtaMatrix([0, 0, -2 / 3], 2)]]

test_cases = [[b, p, e] for b, p, e in zip(Bs, Ps, etas)]


@pytest.mark.parametrize(['B', 'P', 'expected_etas'], test_cases)
def test_refactorize(B, P, expected_etas):
    lp = LpProgram()

    l_etas, u_etas, p = lp.refactorize(B)

    assert any((eta == expected_etas[i] for i, eta in enumerate(l_etas + u_etas)))

    l_etas[0] = l_etas[0].get_matrix()
    u_etas[0] = u_etas[0].get_matrix()
    l = reduce(lambda a, b: np.dot(a, b.get_matrix()), l_etas)
    u = reduce(lambda a, b: np.dot(a, b.get_matrix()), u_etas)

    from scipy.linalg import lu
    P, L, U = lu(B)

    np.testing.assert_array_almost_equal(L, l)
    np.testing.assert_array_almost_equal(U, u.T)

    p_mat = np.eye(len(p))
    for i, pi in enumerate(p):
        p_mat[pi, :] = np.eye(len(p))[i, :]

    np.testing.assert_array_almost_equal(P, p_mat)

    np.testing.assert_almost_equal(np.dot(p_mat, np.dot(l, u.T)), B)
