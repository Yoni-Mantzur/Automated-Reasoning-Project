import pytest
import numpy as np

from lp_solver.eta_matrix import EtaMatrix
from lp_solver.revised_simplex import backward_transformation, forward_transformation

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

    assert np.array_equal(actual_result_with_eta_matrix, actual_result_with_matrix)
    assert np.array_equal(actual_result_with_eta_matrix, expected_result)
