import numpy as np
import pytest

from lp_solver.lp_program import LpProgram


def test_initialize():
    matrix_str = ["4x1,5x2,-6x3>=1", "4x3,5x0>=2", "3.1x1,1.1x2>=0"]
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


@pytest.mark.parametrize('rule', ['bland', 'dantzig'])
def test_simple_lp(rule):
    objective = '5x1,4x2,3x3'
    constraints = ['0x0,2x1,3x2,x3<=5', '4x1,x2,2x3<=11', '3x1,4x2,2x3<=8']
    lp = LpProgram(constraints, objective, rule=rule)
    assert lp.solve() == 13


def test_simple_lp2():
    objective = '5x1,4x2,3.1x3'
    constraints = ['0x0,2x1,3x2,x3<=5', '4x1,x2,2x3<=11', '3x1,4x2,2x3<=8']
    lp = LpProgram(constraints, objective, rule='bland')
    lp.solve()

