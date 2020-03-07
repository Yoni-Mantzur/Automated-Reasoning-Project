from typing import List, Union, Optional, Dict

from lp_solver.UnboundedException import InfeasibleException
from lp_solver.lp_program import Equation, LpProgram
from sat_solver.DPLL import DPLL
from sat_solver.cnf_formula import CnfFormula
from sat_solver.preprocessor import preprocess
from sat_solver.sat_formula import Variable, Literal


class LpTheory():
    def __init__(self, equations: List[Union[Equation, str]] = None, objective: Union[Equation, str] = None,
                 rule: str = 'Dantzig'):
        self.equations = [equation if isinstance(equation, str) else Equation.get_equation(equation) for equation in
                          equations]  # type: List[Equation]
        self.rule = rule
        self.objective = objective

        self.var_eq_map = {Variable(str(i)): eq for i, eq in enumerate(self.equations)}
        # The sat formula is just a conjunction of all the equations
        clauses = [[Literal(v, negated=False)] for v in self.var_eq_map.keys()]
        self.sat_formula = preprocess(CnfFormula(clauses))
        self.last_obj = None

    def conflict(self, partial_assignment: Dict[Variable, bool]) -> List[Literal]:
        eqs = []
        for k, v in partial_assignment.items():
            eqs.append(self.var_eq_map[k])
            if not v:
                eqs[-1].negate(copy=True)

        try:
            lp = LpProgram(equations=eqs, objective=self.objective, rule=self.rule)
            res = lp.solve()
        except InfeasibleException:
            res = None

        if res is None:
            return self._negate_assignment(partial_assignment)
        else:
            # Since we have a big conjunction, the last call to conflict will be with the full assignment
            # so we can save it and use it later instead of re calculating
            self.last_obj = res
            return []

    @staticmethod
    def _negate_assignment(assignment: Dict[Variable, bool]) -> List[Literal]:
        res = []
        for k, v in assignment.items():
            res.append(Literal(k, negated=not v))
        return res

    def propagate(self, partial_assignment: Dict[Variable, bool]) -> Optional[Dict[Variable, bool]]:
        return {} if self.conflict(partial_assignment) == [] else partial_assignment

    def solve(self) -> [bool, Optional[float]]:
        dpll_algorithm = DPLL(self.sat_formula, propagate_helper=self.propagate, conflict_helper=self.conflict)

        # Solve sat formula
        is_sat = dpll_algorithm.search()

        # The search is infeasible
        if not is_sat:
            return False, None

        return True, self.last_obj
