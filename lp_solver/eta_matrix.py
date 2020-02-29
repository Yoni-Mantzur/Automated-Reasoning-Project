from typing import List

import numpy as np


class EtaMatrix(object):
    def __init__(self, column: np.ndarray, column_idx: int):
        self.column = column
        self.column_idx = column_idx

    def invert(self) -> 'EtaMatrix':
        diagonal_element_inverted = 1 / self.column[self.column_idx]
        inverted_column = diagonal_element_inverted * (-self.column)
        inverted_column[self.column_idx] = diagonal_element_inverted

        return EtaMatrix(inverted_column, self.column_idx)

    def get_matrix(self) -> np.ndarray:
        eta_matrix = np.identity(len(self.column))
        eta_matrix[:,self.column_idx] = self.column

        return eta_matrix