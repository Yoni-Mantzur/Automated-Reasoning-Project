from itertools import count

import pytest

from sat_solver.ImplicationGraph import ImplicationGraph, Node
from sat_solver.cnf_formula import CnfFormula
from sat_solver.preprocessor import preprocess
from sat_solver.sat_formula import Literal, Variable


def test_find_all_paths():
    vars = [Variable('x{}'.format(i + 1)) for i in range(8)]
    pos_l = [None] + [Literal(v, negated=False) for v in vars]
    neg_l = [None] + [Literal(v, negated=True) for v in vars]

    clauses = [[pos_l[1]], [neg_l[1], pos_l[5]], [neg_l[1], pos_l[3]], [neg_l[2], neg_l[3], pos_l[4]],
               [neg_l[5], pos_l[2]]]
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    g = ImplicationGraph(cnf)

    g.add_decide_node(1, pos_l[1])
    g.add_node(1, pos_l[3], None, 2)
    g.add_node(1, pos_l[5], None, 1)
    g.add_node(1, pos_l[2], None, 4)
    # g.add_node(1, pos_l[2], None, 3)
    g.add_node(1, pos_l[4], None, 3)
    source_node = g._nodes[pos_l[1]]
    target_node = g._nodes[pos_l[4]]
    actual_paths = g._find_all_paths(source_node, target_node)
    expected_paths = [[pos_l[1].variable, pos_l[5].variable, pos_l[2].variable, pos_l[4].variable],
                      [pos_l[1].variable, pos_l[3].variable, pos_l[4].variable]]

    assert {(frozenset(item)) for item in actual_paths} == {(frozenset(item)) for item in expected_paths}


def test_first_uip_simple():
    vars = [Variable('x{}'.format(i + 1)) for i in range(8)]
    pos_l = [None] + [Literal(v, negated=False) for v in vars]
    neg_l = [None] + [Literal(v, negated=True) for v in vars]

    # x1 =True --> x3=True, x5=True  --> x2 = True,
    clauses = [[pos_l[1]], [neg_l[1], pos_l[5]], [neg_l[1], pos_l[3]], [neg_l[2], neg_l[3]],
               [neg_l[5], pos_l[2]]]
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    g = ImplicationGraph(cnf)

    g.add_decide_node(1, pos_l[1])
    g.add_node(1, pos_l[3], None, 2)
    g.add_node(1, pos_l[5], None, 1)
    g.add_node(1, neg_l[2], None, 4)
    # g.add_node(1, pos_l[2], None, 3)
    # g.add_node(1, pos_l[4], None, 3)
    # source_node = g._nodes[pos_l[1]]
    # target_node = g._nodes[pos_l[4]]
    actual_uip = g.find_first_uip(3)
    expected_uip = g._nodes[pos_l[1].variable]
    assert actual_uip == expected_uip


def test_first_uip_complicated():
    vars = [Variable('x{}'.format(i + 1)) for i in range(10)]
    pos_l = [None] + [Literal(v, negated=False) for v in vars]
    neg_l = [None] + [Literal(v, negated=True) for v in vars]

    c0 = [pos_l[10]] # to stay aligned to the slides
    c1 = [pos_l[2], pos_l[3]]
    c2 = [pos_l[9]]  # to stay aligned to the slides
    c3 = [neg_l[3], neg_l[4]]
    c4 = [neg_l[4], neg_l[2], neg_l[1]]
    c5 = [neg_l[6], neg_l[5], pos_l[4]]
    c6 = [pos_l[7], pos_l[5]]
    c7 = [neg_l[8], pos_l[7], pos_l[6]]
    conflict = [c0, c1, c2, c3, c4, c5, c6, c7]
    # If we implement pure_literal will need to change this
    # this is just to make sure the order decisions will be: x1=True, x2=True, x3=True, the conflict is because x1
    n_temps = 4
    temp_literals = [Literal(Variable('x1_temp{}'.format(idx)), negated=False) for idx in range(n_temps)]
    x1_clauses = [[pos_l[1], l] for l in temp_literals]
    temp_literals = [Literal(Variable('x8_temp{}'.format(idx)), negated=False) for idx in range(n_temps)]
    x8_clauses = [[pos_l[8], l] for l in temp_literals[:-1]]
    temp_literals = [Literal(Variable('x7_temp{}'.format(idx)), negated=False) for idx in range(n_temps)]
    x7_clauses = [[neg_l[7], l] for l in temp_literals[:-2]]

    clauses = conflict # + x1_clauses + x8_clauses + x7_clauses
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)

    g = ImplicationGraph(cnf)

    g.add_decide_node(1, pos_l[1])
    g.add_decide_node(2, pos_l[8])
    g.add_decide_node(3, pos_l[7])

    g.add_node(3, pos_l[6], None, 7)
    g.add_node(3, pos_l[5], None, 6)
    g.add_node(3, pos_l[4], None, 5)
    g.add_node(3, neg_l[3], None, 3)
    g.add_node(3, neg_l[2], None, 1)

    actual_uip = g.find_first_uip(4)
    expected_uip = g._nodes[pos_l[4].variable]
    assert actual_uip == expected_uip


def test_boolean_resolution():
    vars = [Variable('x{}'.format(i + 1)) for i in range(10)]
    pos_l = [None] + [Literal(v, negated=False) for v in vars]
    neg_l = [None] + [Literal(v, negated=True) for v in vars]

    c0 = [pos_l[2], pos_l[3]]
    c1 = [neg_l[4], neg_l[3]]
    c_tag = ImplicationGraph.boolean_resolution(c0, c1, pos_l[3].variable)
    expected_c_tag = [pos_l[2], neg_l[4]]
    assert c_tag == expected_c_tag

    c0 = [pos_l[2], neg_l[3], neg_l[5]]
    c1 = [neg_l[4], pos_l[3], pos_l[1]]
    c_tag = ImplicationGraph.boolean_resolution(c0, c1, pos_l[3].variable)
    expected_c_tag = [pos_l[2], neg_l[4], neg_l[5], pos_l[1]]
    assert sorted(c_tag) == sorted(expected_c_tag)


def test_learn_conflict_simple():

    vars = [Variable('x{}'.format(i + 1)) for i in range(8)]
    pos_l = [None] + [Literal(v, negated=False) for v in vars]
    neg_l = [None] + [Literal(v, negated=True) for v in vars]

    # x1 =True --> x3=True, x5=True  --> x2 = True,
    clauses = [[pos_l[1]], [neg_l[1], pos_l[5]], [neg_l[1], pos_l[3]], [neg_l[2], neg_l[3]],
               [neg_l[5], pos_l[2]]]
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    g = ImplicationGraph(cnf)

    g.add_decide_node(1, pos_l[1])
    g.add_node(1, pos_l[3], None, 2)
    g.add_node(1, pos_l[5], None, 1)
    g.add_node(1, neg_l[2], None, 4)
    # g.add_node(1, pos_l[2], None, 3)
    # g.add_node(1, pos_l[4], None, 3)

    conflict_clause = g.learn_conflict(pos_l[3], 3)
    expected_conflict_clause = [neg_l[1], neg_l[2]]
    assert sorted(conflict_clause) == sorted(expected_conflict_clause)

    conflict_clause = g.learn_conflict(neg_l[2], 3)
    expected_conflict_clause = [neg_l[1], neg_l[3]]
    assert sorted(conflict_clause) == sorted(expected_conflict_clause)

@pytest.fixture(autouse=True)
def clean_counters():
    Node._ids = count(-1)


if __name__ == "__main__":
    test_learn_conflict_simple()
    # Node._ids = count(-1)
    # test_find_all_paths()
    # test_first_uip_simple()
    # test_first_uip_complicated()
    # test_boolean_resolution()
