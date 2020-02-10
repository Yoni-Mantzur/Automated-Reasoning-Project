from sat_solver.DPLL import DPLL
from sat_solver.cnf_formula import CnfFormula
from sat_solver.formula import Literal, Variable
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
    literal_to_clauses = {x1: {0}, not_x1: {2}, not_x2: {0, 3}, x2: {1}, x3: {0, 2}, not_x3: {3}}

    cnf = CnfFormula(clauses, literal_to_clauses)
    cnf = preprocess(cnf)
    dpll = DPLL(cnf)
    actual_cnf = dpll.unit_propagation(cnf)

    assert actual_cnf is None


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
    literal_to_clauses = {x1: {0}, not_x1: {2}, not_x2: {0}, x2: {1}, x3: {0, 2}}

    cnf = CnfFormula(clauses, literal_to_clauses)
    cnf = preprocess(cnf)
    actual_cnf = DPLL(cnf).unit_propagation(cnf)
    expected_cnf = [[x1, x3], [not_x1, x3]]
    actual_cnf_filter = [ls for ls in actual_cnf.clauses if ls != []]
    assert actual_cnf_filter == expected_cnf


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
    print("(x1|x2|~x3) | (x3 | ~x2)", dpll.get_assignment())
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
    dpll = DPLL(cnf)
    search_result = dpll.search()
    print('(~x1 | x2) & (~x1 | ~x2 | ~x3) & (~x2 | x4) & (~x4| ~x3)\n', dpll.get_assignment())
    assert search_result


def test_search_complx_unsat():
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
    dpll = DPLL(cnf)
    search_result = dpll.search()
    print(dpll.get_assignment())
    assert not search_result


if __name__ == '__main__':
    # test_search_simple()
    test_search_complex()
