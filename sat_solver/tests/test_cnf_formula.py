from itertools import count

import pytest

from common.operator import Operator
from sat_solver.cnf_formula import tseitins_transformation
from sat_solver.sat_formula import SatFormula, Variable


def test_simple_tseitins_transformation():
    # Formula: (x1 & x2) || x3
    # Actual Result: tse4 & (tse4 \iff p_g1 || x3) & (tse3 \iff x1 & x2)
    # tse4 & (~tse4 || tse3 || x3) & (tse4 || ~tse3) & (tse4 || ~x3) &
    # & (tse3 || ~x1 || ~x2) & (~tse3 || x1) & (~tse3 || x2)
    x1 = SatFormula.create_leaf("x1")
    x2 = SatFormula.create_leaf("x2")
    x3 = SatFormula.create_leaf("x3")
    x1andx2 = SatFormula(x1, x2, Operator.AND)
    f = SatFormula(x1andx2, x3, Operator.OR)

    actual_cnf = tseitins_transformation(f)
    actual_cnf_set = [set(map(str, ls)) for ls in actual_cnf]
    expected_result = [['tse3'], ['~tse3', 'tse2', 'x3'], ['tse3', '~tse2'], ['tse3', '~x3'], ['tse2', '~x1', '~x2'],
                       ['~tse2', 'x1'], ['~tse2', 'x2']]

    assert len(expected_result) == len(actual_cnf)
    assert all([set(expected) in actual_cnf_set for expected in expected_result])


def test_simple_negate_tseitins_transformation():
    # Formula: ~r1
    # Actual Result: ~r1 & ~tse0 || tse0 & r1 || tse0

    f = SatFormula.from_str('~r1')
    actual_cnf = tseitins_transformation(f)
    expected_result = [['tse0'], ['~tse0', '~r1'], ['r1', 'tse0']]

    actual_cnf_set = [set(map(str, ls)) for ls in actual_cnf]

    assert len(expected_result) == len(actual_cnf)
    assert all([set(expected) in actual_cnf_set for expected in expected_result])


def test_complex_negate_tseitins_transformation():
    # Formula: ~(r1|r2)
    # Actual Result: tse1

    f = SatFormula.from_str('~(r1|r2)')
    actual_cnf = tseitins_transformation(f)
    expected_result = [['tse2'], ['~tse1', 'r1', 'r2'], ['~r1', 'tse1'], ['~r2', 'tse1'], ['~tse2', '~tse1'],
                       ['tse2', 'tse1']]

    actual_cnf_set = [set(map(str, ls)) for ls in actual_cnf]

    assert len(expected_result) == len(actual_cnf)
    assert all([set(expected) in actual_cnf_set for expected in expected_result])


def test_complex_tseitins_transformation():
    # Formula: ~(~(p & q) -> ~r)
    # Actual Result: tse6 & (~tse6 || ~tse5) & (tse6 || tse5) & (~tse5 || ~tse4 || ~r1)
    # & (tse4 || tse5) & (tse5 || r1) & (~tse4 || tse3) & (tse4 || tse3) & (~tse3 || p1)
    # & (~tse3 || q1) & (~p || ~q || tse3)

    f = SatFormula.from_str('~(~(p1&q1)->~r1)')

    actual_cnf = tseitins_transformation(f)

    expected_result = [['tse6'], ['~tse6', '~tse5'], ['tse6', 'tse5'], ['~tse5', '~tse4', 'tse0'],
                       ['~tse0', '~r1'], ['r1', 'tse0'], ['tse4', 'tse5'], ['tse5', '~tse0'],
                       ['~tse4', '~tse3'], ['tse4', 'tse3'], ['~tse3', 'p1'], ['~tse3', 'q1'],
                       ['~p1', '~q1', 'tse3']]

    actual_cnf_set = [set(map(str, ls)) for ls in actual_cnf]

    assert len(expected_result) == len(actual_cnf)
    assert all([set(expected) in actual_cnf_set for expected in expected_result])


@pytest.fixture(autouse=True)
def clean_counters():
    Variable._ids = count(-1)
    SatFormula._ids = count(-1)
