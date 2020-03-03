from random import uniform, seed
import numpy as np
import math
import pytest
from gurobipy import GRB, Model, LinExpr

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


def solve_gurobi() -> float:
    return 2
    gmodel = Model("test")
    obj = LinExpr()


def solve_our(equations_str, objective_str) -> float:
    rule = 'dantzig'
    lp = LpProgram(equations_str, objective_str, rule=rule)
    return lp.solve()


def test_random_lp_dual():
    pytest.skip()
#     TODO: Make test_random_lp more general,and then just change the b that we sample to also contain negative numbers

lp_random_tests = [[2, 2],
                   [4, 2],
                   [4, 4],
                   [5, 10],
                   ]


@pytest.mark.parametrize(['num_variables', 'num_equations'], lp_random_tests)
def test_random_lp(num_variables, num_equations):
    pytest.skip("Need to detect loop assignments, why does it happen?")
    gmodel = Model("test")
    variables = [lp_variable(i, gmodel) for i in range(num_variables)]
    equations_str = []

    for i in range(num_equations):
        cur_equation = []
        gurobi_eq = LinExpr()
        for j in range(num_variables):
            c = uniform(-1, 1)
            # cur_equation.append(lp_variable(j,c))
            cur_equation.append("{}{}".format(c, str(variables[j])))
            gurobi_eq += c * variables[j].get_gurobi_var()
        b = uniform(0, 1)
        equations_str.append("{}<={}".format(','.join(cur_equation), b))
        gmodel.addConstr(gurobi_eq <= b, '{}'.format(i))

    objective_str = ''
    gurobi_obj = LinExpr()
    for j in range(num_variables):
        c = uniform(-0.5, 1)
        objective_str += "{}{},".format(c, variables[j])
        gurobi_obj += c * variables[j].get_gurobi_var()

    # Remove the last ,
    objective_str = objective_str[:-1]
    gmodel.setObjective(gurobi_obj, GRB.MAXIMIZE)
    gmodel.optimize()

    our_objective = solve_our(equations_str, objective_str)
    print(our_objective)
    if our_objective == np.inf:
        assert gmodel.status == GRB.UNBOUNDED or gmodel.status == GRB.INF_OR_UNBD
    else:
        # IF the solution is not feasible the status would be GRB.INFEASIBLE
        assert gmodel.status == GRB.OPTIMAL
        gurobi_objective = gmodel.objVal
        print(gurobi_objective)
        return math.abs(gurobi_objective - our_objective) <= 10**-3
