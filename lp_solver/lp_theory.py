import re
from itertools import count
from typing import List, Union, Optional, Dict, Set
import numpy as np
from common.operator import Operator
from lp_solver.unbounded_exception import InfeasibleException, UnboundedException
from lp_solver.lp_program import Equation, LpProgram
from sat_solver.DPLL import DPLL

from sat_solver.patterns import binary_operator_pattern
from sat_solver.preprocessor import preprocess_from_sat
from sat_solver.sat_formula import Variable, Literal, SatFormula


class EquationsFormula():
    _ids = count(-1)

    def __init__(self,
                 left: 'EquationsFormula' = None,
                 right: 'EquationsFormula' = None,
                 operator: Operator = None,
                 is_leaf: bool = False):
        self.operator = operator
        self.left = left
        self.right = right
        self.is_leaf = is_leaf

        self.idx = next(self._ids)

    @staticmethod
    def create_leaf(raw_eq: str, negated=False) -> 'EquationsFormula':
        formula = EquationsFormula(is_leaf=True)

        eq = Equation.get_equation(raw_eq, negated)
        formula.__setattr__('value', eq)
        return formula

    @staticmethod
    def from_str(s):
        def get_op(s):
            op = re.search(binary_operator_pattern, s).group(0)
            return op, len(op)

        def get_next_special_idx(s, idx):
            for i in range(idx, len(s)):
                special_chars = {'(', ')'}
                special_chars.update({op.value for op in Operator})
                if s[i] in special_chars or s[i:i + 2] in special_chars or s[i:i + 3] in special_chars:
                    return i
            return len(s)

        def _from_str_helper(idx):
            if s[idx] == Operator.NEGATION.value:
                first, new_idx = _from_str_helper(idx + 1)
                return EquationsFormula(first, operator=Operator.NEGATION), new_idx

            if s[idx] == '(':
                first, new_idx = _from_str_helper(idx + 1)
                root, gap_idx = get_op(s[new_idx:])

                second, new_idx = _from_str_helper(new_idx + gap_idx)
                return EquationsFormula(first, second, Operator(root)), new_idx + 1  # The new_idx + 1 is for the ')'

            var_idx = get_next_special_idx(s, idx)
            v = s[idx:var_idx]
            if v not in equations:
                equations[v] = EquationsFormula.create_leaf(v)

            return equations[v], var_idx

        equations = {}
        return _from_str_helper(0)[0]


def extract_equations_from_formula(raw_equations: str) -> Set[str]:
    special_chars = {'(', ')'}
    special_chars.update({op.value for op in Operator})

    def get_next_special_idx(s, idx):
        for i in range(idx, len(s)):
            if s[i] in special_chars:
                return i, 1
            elif s[i:i + 2] in special_chars:
                return i, 2
            elif s[i:i + 3] in special_chars:
                return i, 3
        return len(s), 0

    equations = set()
    i = 0
    while i < len(raw_equations):
        new_idx, gap = get_next_special_idx(raw_equations, i)
        eq = raw_equations[i:new_idx]
        if eq and eq not in special_chars:
            equations.add(eq)
        i = new_idx + gap
    return equations


class LpTheory():
    def __init__(self, formula: str = None, objective: Union[Equation, str] = None, rule: str = 'Dantzig'):
        equations = extract_equations_from_formula(formula)
        self.equations = [equation if isinstance(equation, str) else Equation.get_equation(equation) for equation in
                          equations]  # type: List[Equation]
        self.rule = rule
        self.objective = objective

        self.var_eq_map = {}
        for i, eq in enumerate(equations):
            var_name = 'x{}'.format(i)
            formula = formula.replace(eq, var_name)
            self.var_eq_map.update({var_name: eq})

        self.sat_formula = preprocess_from_sat(SatFormula.from_str(formula))

    def conflict(self, partial_assignment: Dict[Variable, bool]) -> List[Literal]:
        eqs = []
        for k, v in partial_assignment.items():
            if k.name not in self.var_eq_map:
                continue
            eqs.append(self.var_eq_map[k.name])
            if not v:
                eqs[-1].negate(copy=True)

        try:
            lp = LpProgram(equations=eqs, objective=self.objective, rule=self.rule)
            res = lp.solve()
        except InfeasibleException:
            res = None
        except UnboundedException:
            res = np.inf

        if res is None:
            return self._negate_assignment(partial_assignment)
        else:
            return []

    @staticmethod
    def _negate_assignment(assignment: Dict[Variable, bool]) -> List[Literal]:
        res = []
        for k, v in assignment.items():
            res.append(Literal(k, negated=not v))
        return res

    def propagate(self, partial_assignment: Dict[Variable, bool]) -> Optional[Dict[Variable, bool]]:
        return {} if self.conflict(partial_assignment) == [] else partial_assignment

    def solve(self) -> bool:
        dpll_algorithm = DPLL(self.sat_formula, propagate_helper=self.propagate, conflict_helper=self.conflict)

        # Solve sat formula
        is_sat = dpll_algorithm.search()

        # The search is infeasible
        if not is_sat:
            return False

        return True
