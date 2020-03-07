from functools import partial
from random import uniform, seed
from typing import Dict

import numpy as np
import pytest
from gurobipy import GRB, Model, LinExpr

from lp_solver.UnboundedException import InfeasibleException
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
    rule = 'bland'
    try:
        lp = LpProgram(equations_str, objective_str, rule=rule)
    except InfeasibleException:
        return None, None
    return lp.solve(), lp.get_assignment()


constraint_coefficent_sample = partial(uniform, a=-1, b=1)
obj_coefficent_sample = partial(uniform, a=-0.5, b=1)
constraint_scalar_sample = partial(uniform, a=-1, b=1)

lp_random_tests_aux = [
    [2,2],[1,3],[4,3],
    [5, 2], [5, 6], [2, 10], [3, 4]
    # [50, 30],
]


@pytest.mark.timeout(15)
@pytest.mark.parametrize(['num_variables', 'num_equations'], lp_random_tests_aux)
def test_random_lp_auxiliary(num_variables, num_equations):
    return run_one_compare(num_variables, num_equations)


lp_random_tests = [[3, 2], [20, 2], [5, 4], [7, 4], [4, 2], [4, 4], [5, 3]]
                   # [50, 30],
                 #  ]


@pytest.mark.timeout(15)
@pytest.mark.parametrize(['num_variables', 'num_equations'], lp_random_tests)
def test_random_lp(num_variables, num_equations):
    return run_one_compare(num_variables, num_equations, constraint_scalar_sample=partial(uniform, a=0, b=1))


def run_one_compare(num_variables, num_equations, constraint_coefficent_sample=constraint_coefficent_sample,
                    obj_coefficent_sample=obj_coefficent_sample, constraint_scalar_sample=constraint_scalar_sample):
    # pytest.skip("Need to detect loop assignments, why does it happen?")
    gmodel = Model("test")
    variables = [lp_variable(i, gmodel) for i in range(1, num_variables + 1)]
    equations_str = []

    for i in range(num_equations):
        cur_equation = []
        gurobi_eq = LinExpr()
        for v in variables:
            c = constraint_coefficent_sample()
            # cur_equation.append(lp_variable(j,c))
            cur_equation.append("{}{}".format(c, str(v)))
            gurobi_eq += c * v.get_gurobi_var()
        b = constraint_scalar_sample()
        equations_str.append("{}<={}".format(','.join(cur_equation), b))
        gmodel.addConstr(gurobi_eq <= b, '{}'.format(i))

    objective_str = ''
    gurobi_obj = LinExpr()
    for v in variables:
        c = obj_coefficent_sample()
        objective_str += "{}{},".format(c, v)
        gurobi_obj += c * v.get_gurobi_var()

    # Remove the last ,
    objective_str = objective_str[:-1]
    gmodel.setObjective(gurobi_obj, GRB.MAXIMIZE)
    gmodel.optimize()

    print(equations_str, "\n", objective_str)
    our_objective, our_assignment = solve_our(equations_str, objective_str)

    if our_objective == np.inf:
        assert gmodel.status == GRB.UNBOUNDED or gmodel.status == GRB.INF_OR_UNBD
    elif our_objective is None:
        if gmodel.status == GRB.INF_OR_UNBD:
            gmodel.setParam('DualReductions', 0)
            gmodel.optimize()
        assert gmodel.status == GRB.INFEASIBLE
    else:
        # IF the solution is not feasible the status would be GRB.INFEASIBLE
        assert gmodel.status == GRB.OPTIMAL
        gurobi_objective = gmodel.objVal
        print(gurobi_objective)
        for v in variables:
            np.testing.assert_almost_equal(our_assignment.get(v.index, 0), v.gurobi_var.x)
        return np.abs(gurobi_objective - our_objective) <= 10 ** -3

# @pytest.mark.timeout(25)
# def test_lp_from_str():
#     eq  = ['-0.4989873172751189x1,0.8194925119364802x2,0.9655709520753062x3,0.6204344719931791x4<=0.8043319008791654', '-0.37970486136133474x1,0.4596634965202573x2,0.797676575935987x3,0.36796786383088254x4<=-0.055714569094573285', '-0.7985975838632684x1,-0.13165632909243263x2,0.2217739468876032x3,0.8260221064757964x4<=0.9332127355415176']
#     obj = '0.21551466482907555x1,0.7979648916574602x2,-0.10926153441206088x3,0.7075417405195334x4'
#     our_objective, our_assignment = solve_our(eq, obj)
#     assert our_objective == np.inf


# if __name__ == "__main__":
#     test_lp_from_str()