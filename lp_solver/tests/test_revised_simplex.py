import pytest
import numpy as np

from lp_solver.eta_matrix import EtaMatrix
from lp_solver.revised_simplex import backward_transformation, forward_transformation

test_cases_btran = [[EtaMatrix([-4, 3, 2], 1), [1, 2, 3], [1, 0, 3]]]
test_cases_ftran = [[EtaMatrix([-4, 3, 2], 1), [1, 2, 3], [1 + 8 / 3, 2 / 3, 3 - 4 / 3]]]


@pytest.mark.parametrize(['B', 'c', 'expected_result'], test_cases_btran)
def test_backward_transformation(B, c, expected_result):
    c = np.array(c)
    actual_result_with_eta_matrix = backward_transformation(B, c)
    actual_result_with_matrix = backward_transformation(B.get_matrix(), c)

    assert np.array_equal(actual_result_with_eta_matrix, actual_result_with_matrix)
    assert np.array_equal(actual_result_with_eta_matrix, np.array(expected_result))


@pytest.mark.parametrize(['B', 'a', 'expected_result'], test_cases_ftran)
def test_backward_transformation(B, a, expected_result):
    a = np.array(a)
    actual_result_with_eta_matrix = forward_transformation(B, a)
    actual_result_with_matrix = forward_transformation(B.get_matrix(), a)

    assert np.array_equal(actual_result_with_eta_matrix, actual_result_with_matrix)
    assert np.array_equal(actual_result_with_eta_matrix, np.array(expected_result))
