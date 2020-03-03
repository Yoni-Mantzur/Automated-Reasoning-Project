import pytest
from random import uniform, seed
from lp_solver.lp_program import LpProgram


seed(0)
class lp_variable():
    def __init__(self, index, coefficient):
        self.index = index
        self.coefficient = coefficient

    def __str__(self):
        return '{}x{}'.format(self.coefficient,self.index)
    def __repr__(self):
        return str(self)

def solve_gurobi() -> float:
    return 2
    gmodel = Model("test")

def solve_our(equations_str, objective_str) -> float:
    rule = 'dantzig'
    lp = LpProgram(equations_str, objective_str, rule=rule)
    return lp.solve()


def test_random_lp_dual():
    pytest.skip()

lp_random_tests = [[4 ,2],
                   [5, 10]
                 ]

@pytest.mark.parametrize(['num_variables', 'num_equations'], lp_random_tests)
def test_random_lp(num_variables, num_equations):

    equations_str = []
    for i in range(num_equations):
        cur_equation = []
        for j in range(num_variables):
            c = uniform(-1,1)
            cur_equation.append(lp_variable(j,c))
        b = uniform(0,1)
        equations_str.append("{}<={}".format(str(cur_equation).replace('[', '').replace(']', ''), b))

    for j in range(num_variables):
        c = uniform(-0.5, 1)
        cur_equation.append(lp_variable(j,c))

    objective_str = str(cur_equation).replace('[', '').replace(']', '')

    our_objective = solve_our(equations_str, objective_str)
    print(our_objective)
    return 1