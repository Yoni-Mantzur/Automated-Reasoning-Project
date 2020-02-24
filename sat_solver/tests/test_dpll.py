from itertools import count

import pytest

from sat_solver.DPLL import DPLL
from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import Literal, Variable, SatFormula
from sat_solver.preprocessor import preprocess


def test_up_unsat():
    # (x1|~x2|x3)&x2&(~x1|x3)&(x2|~x3) --> UNSAT
    x1_var = Variable('x1')
    x2_var = Variable('x2')
    x3_var = Variable('x3')

    x1 = Literal(x1_var, negated=False)
    not_x1 = Literal(x1_var, negated=True)
    x2 = Literal(x2_var, negated=False)
    not_x2 = Literal(x2_var, negated=True)
    x3 = Literal(x3_var, negated=False)
    not_x3 = Literal(x3_var, negated=True)

    clauses = [[x1, not_x2, x3], [x2], [not_x1, x3], [not_x2, not_x3]]
    # literal_to_clauses = {x1: {0}, not_x1: {2}, not_x2: {0, 3}, x2: {1}, x3: {0, 2}, not_x3: {3}}

    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)

    dpll = DPLL(cnf)
    actual_cnf = dpll.unit_propagation()

    assert actual_cnf is None
    assert not dpll.get_full_assignment()[x3_var]
    assert dpll.get_full_assignment()[x2_var]


def test_up_simple():
    # (x1|~x2|x3)&x2&(~x1|x3) --> (x1|x3) & (~x1|x3)
    x1_var = Variable('x1')
    x2_var = Variable('x2')
    x3_var = Variable('x3')

    x1 = Literal(x1_var, negated=False)
    not_x1 = Literal(x1_var, negated=True)
    x2 = Literal(x2_var, negated=False)
    not_x2 = Literal(x2_var, negated=True)
    x3 = Literal(x3_var, negated=False)

    clauses = [[x1, not_x2, x3], [x2], [not_x1, x3]]
    # literal_to_clauses = {x1: {0}, not_x1: {2}, not_x2: {0}, x2: {1}, x3: {0, 2}}

    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    dpll = DPLL(cnf)
    actual_cnf = dpll.unit_propagation()
    expected_cnf = [[x1,not_x2, x3], [not_x1, x3]]

    assert dpll.get_full_assignment()[x2_var]
    actual_cnf_real = [cl for cl in actual_cnf.clauses if cl != []]
    assert actual_cnf_real == expected_cnf, dpll.get_full_assignment()


def test_search_simple():
    # (x1|x2|~x3) | (x3 | ~x2)
    x1_var = Variable('x1')
    x2_var = Variable('x2')
    x3_var = Variable('x3')

    x1 = Literal(x1_var, negated=False)
    not_x1 = Literal(x1_var, negated=True)
    x2 = Literal(x2_var, negated=False)
    not_x2 = Literal(x2_var, negated=True)
    x3 = Literal(x3_var, negated=False)
    not_x3 = Literal(x3_var, negated=True)

    clauses = [[x1, x2, not_x3], [x3, not_x2]]
    # literal_to_clauses = {x1: {0}, x2: {0}, x3: {0}} #not_x1: {2}, not_x2: {0}
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    dpll = DPLL(cnf)
    search_result = dpll.search()
    # print("(x1|x2|~x3) | (x3 | ~x2)", dpll.get_assignment())
    assert search_result


def test_search_simple_unsat():
    # (x1|~x2) | (~x1 | x2)
    x1_var = Variable('x1')
    x2_var = Variable('x2')
    # x3_var = Variable('x3')

    x1 = Literal(x1_var, negated=False)
    not_x1 = Literal(x1_var, negated=True)
    x2 = Literal(x2_var, negated=False)
    not_x2 = Literal(x2_var, negated=True)

    clauses = [[x1], [x2], [not_x1, not_x2]]
    # literal_to_clauses = {x1: {0}, x2: {0}, x3: {0}} #not_x1: {2}, not_x2: {0}
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    dpll = DPLL(cnf)
    search_result = dpll.search()

    assert not search_result


def test_search_complex():
    # x1 = x, x2 = z, x3=y, x4=w
    # (~x|z) & (~x|~z|~y) & (~z|w) & (~w|~y) (lec3, slide 18)
    # (~x1 | x2) & (~x1 | ~x2 | ~x3) & (~x2 | x4) & (~x4| ~x3)
    x1_var = Variable('x1')
    x2_var = Variable('x2')
    x3_var = Variable('x3')
    x4_var = Variable('x4')
    x5_var = Variable('x5')
    x6_var = Variable('x6')

    x1 = Literal(x1_var, negated=False)
    not_x1 = Literal(x1_var, negated=True)
    x2 = Literal(x2_var, negated=False)
    not_x2 = Literal(x2_var, negated=True)
    x3 = Literal(x3_var, negated=False)
    not_x3 = Literal(x3_var, negated=True)
    x4 = Literal(x4_var, negated=False)
    not_x4 = Literal(x4_var, negated=True)

    x5 = Literal(x5_var, negated=False)
    not_x5 = Literal(x5_var, negated=True)
    x6 = Literal(x6_var, negated=False)
    not_x6 = Literal(x6_var, negated=True)
    # (~x1 | x2) & (~x1 | ~x2 | ~x3) & (~x2 | x4) & (~x4| ~x3)
    clauses = [[not_x1, x2], [not_x1,not_x2, not_x3], [not_x2, x4], [not_x4, not_x3], [x5, x6], [not_x5, not_x6]]
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    dpll = DPLL(cnf)
    search_result = dpll.search()
    # print('(~x1 | x2) & (~x1 | ~x2 | ~x3) & (~x2 | x4) & (~x4| ~x3)\n', dpll.get_assignment())
    assert search_result


def test_search_complex_unsat():
    x1_var = Variable('x1')
    x2_var = Variable('x2')
    x3_var = Variable('x3')
    x4_var = Variable('x4')

    x1 = Literal(x1_var, negated=False)
    not_x1 = Literal(x1_var, negated=True)
    x2 = Literal(x2_var, negated=False)
    not_x2 = Literal(x2_var, negated=True)
    x3 = Literal(x3_var, negated=False)
    not_x3 = Literal(x3_var, negated=True)
    x4 = Literal(x4_var, negated=False)
    not_x4 = Literal(x4_var, negated=True)

    # x1 = T, x2 = F, x3 =
    clauses = [[not_x1, x2, not_x3], [x3, not_x2, x1], [x1, x2], [not_x1, not_x2], [x3, x2], [not_x3, not_x2]]
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    dpll = DPLL(cnf)
    search_result = dpll.search()
    # print(dpll.get_assignment())
    assert not search_result


def test_multi_level_deduction_sat():
    x1_var = Variable('x1')
    x2_var = Variable('x2')
    x3_var = Variable('x3')
    # x4_var = Variable('x4')
    # x5_var = Variable('x5')
    # x6_var = Variable('x6')

    x1 = Literal(x1_var, negated=False)
    not_x1 = Literal(x1_var, negated=True)
    x2 = Literal(x2_var, negated=False)
    not_x2 = Literal(x2_var, negated=True)
    x3 = Literal(x3_var, negated=False)
    not_x3 = Literal(x3_var, negated=True)
    # x4 = Literal(x4_var, negated=False)
    # not_x4 = Literal(x4_var, negated=True)
    # x5 = Literal(x5_var, negated=False)
    # not_x5 = Literal(x5_var, negated=True)
    # x6 = Literal(x6_var, negated=False)
    # not_x6 = Literal(x6_var, negated=True)

    # X1 = True, detect x2=False, then need to decide x2 = True, then use x1=True and x2=True to detect x3=False
    clauses = [[x1], [x3, x2], [not_x2, not_x3, not_x1]]
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    dpll = DPLL(cnf)
    search_result = dpll.search()

    assert search_result


def test_multi_level_conflict_sat():
    vars =  [Variable('x{}'.format(i + 1)) for i in range(8)]
    pos_l = [None] + [Literal(v, negated=False) for v in vars]
    neg_l = [None] + [Literal(v, negated=True) for v in vars]

    c1 = [pos_l[2], pos_l[3]]
    # c2 = [pos_l[1], pos_l[4], neg_l[8]]
    c3 = [neg_l[3], neg_l[4]]
    c4 = [neg_l[4], neg_l[2], neg_l[1]]
    c5 = [neg_l[6], neg_l[5], pos_l[4]]
    c6 = [pos_l[7], pos_l[5]]
    c7 = [neg_l[8], pos_l[7], pos_l[6]]
    conflict = [c3, c4, c5, c6, c7, c1] #c2,
    # If we implement pure_literal will need to change this
    # this is just to make sure the order decisions will be: x1=True, x2=True, x3=True, the conflict is because x1
    n_temps = 4
    temp_literals = [Literal(Variable('x1_temp{}'.format(idx)), negated=False) for idx in range(n_temps)]
    x1_clauses = [[pos_l[1], l] for l in temp_literals]
    temp_literals = [Literal(Variable('x8_temp{}'.format(idx)), negated=False) for idx in range(n_temps)]
    x8_clauses = [[pos_l[8], l] for l in temp_literals[:-1]]
    temp_literals = [Literal(Variable('x7_temp{}'.format(idx)), negated=False) for idx in range(n_temps)]
    x7_clauses = [[neg_l[7], l] for l in temp_literals[:-2]]

    clauses = x1_clauses + x8_clauses + x7_clauses + conflict
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    dpll = DPLL(cnf)
    search_result = dpll.search()

    assert search_result



@pytest.fixture(autouse=True)
def clean_counters():
    Variable._ids = count(-1)
    SatFormula._ids = count(-1)


if __name__ == '__main__':
    # test_up_unsat()
    # print("*" * 100)
    # print("pass test_up_unsat")
    # print("*" * 100)
    # test_up_simple()
    # print("*" * 100)
    # print("pass test_up_simple")
    # print("*" * 100)
    test_search_complex_unsat()
    print("*" * 100)
    print("pass test_search_complex_unsat")
    print("*" * 100)
    # test_search_complex()
    # print("*" * 100)
    # print("pass test_search_complex")
    # print("*" * 100)
    # test_search_simple_unsat()
    # print("*" * 100)
    # print("pass test_search_simple_unsat")
    # print("*" * 100)
    # test_multi_level_deduction_sat()
    # print("*" * 100)
    # print("pass test_multi_level_deduction_sat")
    # print("*" * 100)
    # test_search_complex()
    # print("*" * 100)
    # print("pass test_search_complex")
    # print("*" * 100)
    # test_search_complex()
    # print("*" * 100)
    # print("pass test_search_complex")
    # print("*" * 100)
    test_multi_level_conflict_sat()
    print("*" * 100)
    print("pass test_multi_level_conflict_sat")
    print("*" * 100)
    #
    # test_search_complex_unsat()
    # print("*" * 100)
    # print("pass test_search_complex_unsat")
    # print("*" * 100)
