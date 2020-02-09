from typing import List, Dict

from sat_solver.formula import Formula, Literal


def tseitins_transformation(formula: Formula):
    pass


class CnfFormula(object):

    def __init__(self, formula: Formula):
        self._formula = tseitins_transformation(formula)  # type: List[List[Literal]]
        self._variable_to_clause = {}  # type: Dict[int, int]
