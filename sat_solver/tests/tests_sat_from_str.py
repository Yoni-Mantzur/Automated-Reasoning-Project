from sat_solver.DPLL import DPLL
from sat_solver.preprocessor import preprocess_from_sat
from sat_solver.sat_formula import SatFormula


def get_result(query: str):
    sat_formula = SatFormula.from_str(query)
    formula = preprocess_from_sat(sat_formula)
    dpll = DPLL(formula)
    res = dpll.search()
    return res


def test_single_variable():
    s = 'x1'
    assert get_result(s)


def test_simple_sat():
    s = '((x1&x2)|(~x1&~x2))'
    assert get_result(s)


def test_simple_unsat():
    s = '(x1&~x1)'
    assert not get_result(s)

    s = '((x1&x2)&(~x1&~x2))'
    assert not get_result(s)
