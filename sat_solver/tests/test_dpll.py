from collections import defaultdict
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
    dpll = DPLL(cnf, partial_assignment={}, watch_literals=defaultdict(set))
    actual_cnf = dpll.unit_propagation(dpll.formula)

    assert actual_cnf is None
    assert not dpll.get_assignment()['x3']
    assert dpll.get_assignment()['x2']


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
    dpll = DPLL(cnf, partial_assignment={}, watch_literals=defaultdict(set))
    actual_cnf = dpll.unit_propagation(cnf)
    expected_cnf = [[x1,not_x2, x3], [not_x1, x3]]

    assert dpll.get_assignment()['x2']
    actual_cnf_real = [cl for cl in actual_cnf.clauses if cl != []]
    assert actual_cnf_real == expected_cnf, dpll.get_assignment()


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
    dpll = DPLL(cnf, partial_assignment={}, watch_literals=defaultdict(set))
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
    dpll = DPLL(cnf, partial_assignment={}, watch_literals=defaultdict(set))
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

    x1 = Literal(x1_var, negated=False)
    not_x1 = Literal(x1_var, negated=True)
    x2 = Literal(x2_var, negated=False)
    not_x2 = Literal(x2_var, negated=True)
    x3 = Literal(x3_var, negated=False)
    not_x3 = Literal(x3_var, negated=True)
    x4 = Literal(x4_var, negated=False)
    not_x4 = Literal(x4_var, negated=True)

    # (~x1 | x2) & (~x1 | ~x2 | ~x3) & (~x2 | x4) & (~x4| ~x3)
    clauses = [[not_x1, x2], [not_x1,not_x2, not_x3], [not_x2, x4], [not_x4, not_x3]]
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    dpll = DPLL(cnf, partial_assignment={}, watch_literals=defaultdict(set))
    search_result = dpll.search()
    # print('(~x1 | x2) & (~x1 | ~x2 | ~x3) & (~x2 | x4) & (~x4| ~x3)\n', dpll.get_assignment())
    assert search_result


def test_search_complex_unsat():
    x1_var = Variable('x1')
    x2_var = Variable('x2')
    x3_var = Variable('x3')

    x1 = Literal(x1_var, negated=False)
    not_x1 = Literal(x1_var, negated=True)
    x2 = Literal(x2_var, negated=False)
    not_x2 = Literal(x2_var, negated=True)
    x3 = Literal(x3_var, negated=False)
    not_x3 = Literal(x3_var, negated=True)

    # x1 = T, x2 = F, x3 =
    clauses = [[not_x1, x2, not_x3], [x3, not_x2, x1], [x1, x2], [not_x1, not_x2], [x3, x2], [not_x3, not_x2]]
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    dpll = DPLL(cnf, partial_assignment={}, watch_literals=defaultdict(set))
    search_result = dpll.search()
    # print(dpll.get_assignment())
    assert not search_result

@pytest.fixture(autouse=True)
def clean_counters():
    Variable._ids = count(-1)
    SatFormula._ids = count(-1)

if __name__ == '__main__':
    test_up_simple()
#     test_search_complex()
#     exit(0)
    test_search_complex_unsat()
    test_search_complex()
    # test_search_simple_unsat()