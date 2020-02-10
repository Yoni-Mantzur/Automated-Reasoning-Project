import pytest

from sat_solver.cnf_formula import CnfFormula
from sat_solver.formula import Literal, Variable
from sat_solver.preprocessor import remove_redundant_literals, delete_trivial_clauses

v1 = Variable(name='v1')
v2 = Variable(name='v2')

l1 = Literal(v1, negated=False)
l2 = Literal(v1, negated=True)
l3 = Literal(v2, negated=False)
l4 = Literal(v2, negated=True)


def test_remove_redundant_literals():
    formula = CnfFormula([[l1, l2], [l1, l1, l3]])
    expected_formula = CnfFormula([[l1, l2], [l1, l3]])

    actual_formula = remove_redundant_literals(formula)

    assert str(expected_formula) == str(actual_formula)


def test_delete_trivial_clauses():
    formula = remove_redundant_literals(CnfFormula([[l1, l2], [l1, l1, l3]]))
    expected_formula = CnfFormula([[l1, l3]])

    actual_formula = delete_trivial_clauses(formula)

    assert str(expected_formula) == str(actual_formula)

def test_exception_raised_when_remove_redundant_literals_was_not_called_before():
    formula = CnfFormula([[l1, l2], [l1, l1, l3]])

    with pytest.raises(Exception):
        delete_trivial_clauses(formula)
