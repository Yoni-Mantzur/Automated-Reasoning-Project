import numpy as np

from lp_solver.lp_program import LpProgram


def test_initialize():
    matrix_str = ["4x1,5x2,-6x3>=-1", "4x3,5x0>=2", "3.1x1,1.1x2>=0"]
    objective_str = "2x1,-3x2,0x3"
    non_basic_variables = [0, 1, 2, 3]
    matrix = np.array([
        [0, 4, 5, -6],
        [5, 0, 0, 4],
        [0, 3.1, 1.1, 0],
    ])

    lp = LpProgram(matrix_str, objective_str)
    assert np.array_equal(lp.An, matrix)
    assert np.array_equal(lp.b, [-1, 2, 0])
    assert sorted(lp.Xn) == sorted(non_basic_variables)

    basic_variables = [4, 5, 6]
    basic_matrix = np.eye(len(basic_variables))
    assert sorted(lp.Xb) == sorted(basic_variables)
    assert np.array_equal(lp.B, basic_matrix)

    objective_non_basic = np.array([0, 2, -3, 0])
    objective_basic = np.zeros(shape=len(basic_variables))
    assert np.array_equal(lp.Cb, objective_basic)
    assert np.array_equal(lp.Cn, objective_non_basic)
    lp.dump()

if __name__ == "__main__":
    test_initialize()
