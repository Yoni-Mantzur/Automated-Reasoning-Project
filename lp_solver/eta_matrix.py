from typing import List, Union

import numpy as np


class EtaMatrix(object):
    def __init__(self, column: Union[List, np.ndarray], column_idx: int):
        self.column = np.array(column) if type(column) == list else np.copy(column)
        self.column_idx = column_idx
        self.cache_invert = None

    def invert(self) -> 'EtaMatrix':
        if not self.cache_invert:
            diagonal_element_inverted = 1 / self.column[self.column_idx]
            inverted_column = diagonal_element_inverted * (-self.column)
            inverted_column[self.column_idx] = diagonal_element_inverted
            self.cache_invert = EtaMatrix(inverted_column, self.column_idx)

        return self.cache_invert

    def get_matrix(self) -> np.ndarray:
        eta_matrix = np.identity(len(self.column))
        eta_matrix[:, self.column_idx] = self.column

        return eta_matrix

    @property
    def T(self):
        return self.get_matrix().T

    def __mul__(self, other):
        return np.dot(self.get_matrix(), other.get_matrix())

    def __eq__(self, other):
        return np.array_equal(self.get_matrix(), other.get_matrix())

    def __hash__(self):
        return hash(self.get_matrix())
