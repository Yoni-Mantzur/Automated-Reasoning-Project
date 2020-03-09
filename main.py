from pprint import pprint

import numpy as np
import sys
from sat_solver.DPLL import DPLL
from sat_solver.preprocessor import preprocess_from_sat
from smt_solver.formula import Formula

from smt_solver.formula import SatFormula

from lp_solver.lp_program import LpProgram
from lp_solver.unbounded_exception import UnboundedException, InfeasibleException
from lp_solver.lp_theory import LpTheory

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("execution instructions [SAT|SMT|LP] [query string]")
        exit(1)

    q = sys.argv[2]
    if sys.argv[1].lower() == 'sat':
        sat = SatFormula.from_str(q)
        formula = preprocess_from_sat(sat)
        dpll = DPLL(formula)
        res = dpll.search()
        print(f"Got {'SAT' if res else 'UNSAT'} for query\n{q}")
        # optional to retrieve the assignment
        if res:
            assignment = dpll.get_full_assignment()
            all_variables = set(l.variable for l in sat.get_literals())
            actual_assignments = map(lambda k: (k, assignment[k]), all_variables)
            pprint(dict(actual_assignments))
    if sys.argv[1].lower() == 'smt':
        smt_query = Formula.from_str(q)
        res, model = smt_query.solve()
        print(f"Got {'SAT' if res else 'UNSAT'} for query\n{q}")

    if sys.argv[1].lower() == 'lp':
        if len(sys.argv) < 4:
            print("lp requires at least one equation and one objective")
            exit(1)
        equations = sys.argv[2:-1]
        objective = sys.argv[-1]
        try:
            z = LpProgram(equations, objective).solve()
            # To use the LP that uses the sat solver uncomment:
            # res, z = LpTheory(equations, objective).solve()
            printable_z = z
        except UnboundedException:
            z = np.inf
        except InfeasibleException:
            z = None

        if z is None:
            printable_z = 'Infeasible'
        elif z == np.inf:
            printable_z = 'Unbounded'

        print(f"Got {printable_z} for objective\n\t{objective.replace(',', '+')}")