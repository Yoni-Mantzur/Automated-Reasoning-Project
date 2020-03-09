from functools import partial
from random import uniform, seed
from typing import Dict

import numpy as np
import pytest
from gurobipy import GRB, Model, LinExpr

from lp_solver.unbounded_exception import InfeasibleException
from lp_solver.lp_program import LpProgram

seed(0)


class lp_variable():
    def __init__(self, index, gurobi_model):
        self.index = index
        self.gurobi_var = gurobi_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x{}".format(index))

    def __str__(self):
        return 'x{}'.format(self.index)

    def __repr__(self):
        return str(self)

    def get_gurobi_var(self):
        return self.gurobi_var


def solve_our(equations_str, objective_str) -> [float, Dict[int, float]]:
    rule = 'dantzig'
    try:
        lp = LpProgram(equations_str, objective_str, rule=rule)
    except InfeasibleException:
        return None, None
    return lp.solve(), lp.get_assignment()


constraint_coefficent_sample = partial(uniform, a=-1, b=1)
obj_coefficent_sample = partial(uniform, a=-0.5, b=1)
constraint_scalar_sample = partial(uniform, a=-1, b=1)

lp_random_tests_aux = [
    [2, 2], [1, 3], [4, 3],
    [5, 2], [5, 6], [2, 10], [3, 4],
    [50, 30], [30, 12]
]


@pytest.mark.gurobi
@pytest.mark.timeout(55)
@pytest.mark.parametrize(['num_variables', 'num_equations'], lp_random_tests_aux)
def test_random_lp_auxiliary(num_variables, num_equations):
    return run_one_compare(num_variables, num_equations)


lp_random_tests = [[3, 2], [20, 2], [5, 4], [7, 4], [4, 2], [4, 4], [5, 3], [50, 30]]

@pytest.mark.gurobi
@pytest.mark.timeout(55)
@pytest.mark.parametrize(['num_variables', 'num_equations'], lp_random_tests)
def test_random_lp(num_variables, num_equations):
    return run_one_compare(num_variables, num_equations, constraint_scalar_sample=partial(uniform, a=0, b=1))


def run_one_compare(num_variables, num_equations, constraint_coefficent_sample=constraint_coefficent_sample,
                    obj_coefficent_sample=obj_coefficent_sample, constraint_scalar_sample=constraint_scalar_sample):
    seed(0)
    gmodel = Model("test")
    variables = [lp_variable(i, gmodel) for i in range(1, num_variables + 1)]
    equations_str = []

    for i in range(num_equations):
        cur_equation = []
        gurobi_eq = LinExpr()
        for v in variables:
            c = round(constraint_coefficent_sample(), 3)
            cur_equation.append("{}{}".format(c, str(v)))
            gurobi_eq += c * v.get_gurobi_var()
        b = round(constraint_scalar_sample(), 3)
        equations_str.append("{}<={}".format(','.join(cur_equation), b))
        gmodel.addConstr(gurobi_eq <= b, '{}'.format(i))

    objective_str = ''
    gurobi_obj = LinExpr()
    for v in variables:
        c = round(obj_coefficent_sample(), 3)
        objective_str += "{}{},".format(c, v)
        gurobi_obj += c * v.get_gurobi_var()

    # Remove the last ,
    objective_str = objective_str[:-1]
    gmodel.setObjective(gurobi_obj, GRB.MAXIMIZE)
    gmodel.optimize()

    our_objective, our_assignment = solve_our(equations_str, objective_str)

    if gmodel.status == GRB.INF_OR_UNBD:
        gmodel.setParam('DualReductions', 0)
        gmodel.optimize()
        print('UNBOUNDED' if gmodel.status == GRB.UNBOUNDED else 'INFEASIBLE')
    if our_objective == np.inf:
        assert gmodel.status == GRB.UNBOUNDED
    elif our_objective is None:
        assert gmodel.status == GRB.INFEASIBLE
    else:
        # IF the solution is not feasible the status would be GRB.INFEASIBLE
        assert gmodel.status == GRB.OPTIMAL
        gurobi_objective = gmodel.objVal
        print(gurobi_objective)
        for v in variables:
            np.testing.assert_almost_equal(our_assignment.get(v.index, 0), v.gurobi_var.x)
        return np.abs(gurobi_objective - our_objective) <= 10 ** -3
