import pytest
import numpy as np

from lp_solver.eta_matrix import EtaMatrix

test_cases = [[[1, 2, 3], 2],
              [[1, 2, 3], 1],
              [[1, 2, 3], 0],
              [[1, 2, 3, 4], 3],
              [[1, -2, 3, 4], 1]]


@pytest.mark.parametrize(['column', 'idx'], test_cases)
def test_invert(column, idx):
    eta_matrix = EtaMatrix(np.array(column), column_idx=idx)
    expected_inv_matrix = np.linalg.inv(eta_matrix.get_matrix())

    actual_inv_matrix = eta_matrix.invert().get_matrix()

    assert np.array_equal(expected_inv_matrix, actual_inv_matrix)
