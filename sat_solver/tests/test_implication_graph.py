from itertools import count
from typing import List

import pytest

from sat_solver.ImplicationGraph import ImplicationGraph, Node
from sat_solver.cnf_formula import CnfFormula
from sat_solver.preprocessor import preprocess
from sat_solver.sat_formula import Literal, Variable


def get_complicated_graph() -> (ImplicationGraph, int, List[Variable]):
    variables = [Variable('TEMP')] + [Variable('x{}'.format(i + 1)) for i in range(10)]
    pos_l = [Literal(v, negated=False) for v in variables]
    neg_l = [Literal(v, negated=True) for v in variables]

    c0 = [pos_l[10]]  # unused clause to keep the numbering aligned to the slides
    c1 = [pos_l[2], pos_l[3]]
    c2 = [pos_l[9]]  # unused clause to keep the numbering aligned to the slides
    c3 = [neg_l[3], neg_l[4]]
    c4 = [neg_l[4], neg_l[2], neg_l[1]]
    c5 = [neg_l[6], neg_l[5], pos_l[4]]
    c6 = [pos_l[7], pos_l[5]]
    c7 = [neg_l[8], pos_l[7], pos_l[6]]
    conflict = [c0, c1, c2, c3, c4, c5, c6, c7]
    # If we implement pure_literal will need to change this

    # n_temps = 4
    # temp_literals = [Literal(Variable('x1_temp{}'.format(idx)), negated=False) for idx in range(n_temps)]
    # x1_clauses = [[pos_l[1], l] for l in temp_literals]
    # temp_literals = [Literal(Variable('x8_temp{}'.format(idx)), negated=False) for idx in range(n_temps)]
    # x8_clauses = [[pos_l[8], l] for l in temp_literals[:-1]]
    # temp_literals = [Literal(Variable('x7_temp{}'.format(idx)), negated=False) for idx in range(n_temps)]
    # x7_clauses = [[neg_l[7], l] for l in temp_literals[:-2]]

    clauses = conflict  # + x1_clauses + x8_clauses + x7_clauses
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
    g.add_node(3, neg_l[2], None, 4)

    return g, 1, variables


def get_simple_graph() -> (ImplicationGraph, int, List[Variable]):
    variables = [Variable('TEMP')] + [Variable('x{}'.format(i + 1)) for i in range(8)]
    pos_l = [Literal(v, negated=False) for v in variables]
    neg_l = [Literal(v, negated=True) for v in variables]

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

    return g, 3, variables


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
    g, conflict_idx, variables = get_simple_graph()

    actual_uip = g.find_first_uip(conflict_idx)
    expected_uip = g._nodes[variables[1]]
    assert actual_uip == expected_uip


def test_first_uip_complicated():
    g, conflict_idx, variables = get_complicated_graph()

    actual_uip = g.find_first_uip(conflict_idx)
    expected_uip = g._nodes[variables[4]]
    assert actual_uip == expected_uip


def test_boolean_resolution():
    vars = [Variable('x{}'.format(i + 1)) for i in range(10)]
    pos_l = [None] + [Literal(v, negated=False) for v in vars]
    neg_l = [None] + [Literal(v, negated=True) for v in vars]

    c0 = [pos_l[2], pos_l[3]]
    c1 = [neg_l[4], neg_l[3]]
    c_tag = ImplicationGraph.boolean_resolution(c0, c1, pos_l[3].variable)
    expected_c_tag = [pos_l[2], neg_l[4]]
    assert sorted(c_tag) == sorted(expected_c_tag)

    c0 = [pos_l[2], neg_l[3], neg_l[5]]
    c1 = [neg_l[4], pos_l[3], pos_l[1]]
    c_tag = ImplicationGraph.boolean_resolution(c0, c1, pos_l[3].variable)
    expected_c_tag = [pos_l[2], neg_l[4], neg_l[5], pos_l[1]]
    assert sorted(c_tag) == sorted(expected_c_tag)


def test_learn_conflict_simple():
    g, conflict_idx, variables = get_simple_graph()

    conflict_clause = g.learn_conflict(Literal(variables[3], False), conflict_idx)
    expected_conflict_clause = [Literal(variables[1], True)]
    assert sorted(conflict_clause) == sorted(expected_conflict_clause)

    conflict_clause = g.learn_conflict(Literal(variables[2], True), conflict_idx)
    expected_conflict_clause = [Literal(variables[1], True)]
    assert sorted(conflict_clause) == sorted(expected_conflict_clause)


def test_learn_conflict_complicated():
    g, conflict_idx, variables = get_complicated_graph()

    conflict_clause = g.learn_conflict(Literal(variables[3], True), conflict_idx)
    expected_conflict_clause = [Literal(variables[1], True), Literal(variables[4], True)]
    assert sorted(conflict_clause) == sorted(expected_conflict_clause)

    # actual_uip = g.find_first_uip(conflict_idx)
    # expected_uip = g._nodes[variables[4]]
    # assert actual_uip == expected_uip


def test_backjump_simple():
    g, conflict_idx, variables = get_simple_graph()

    conflict_clause = g.learn_conflict(Literal(variables[3], False), conflict_idx)
    level = g.get_backjump_level(conflict_clause)
    expected_level = 0
    assert level == expected_level


def test_backjump_complicated():
    g, conflict_idx, variables = get_complicated_graph()

    conflict_clause = g.learn_conflict(Literal(variables[3], True), conflict_idx)
    level = g.get_backjump_level(conflict_clause)
    expected_level = 1
    assert level == expected_level


@pytest.fixture(autouse=True)
def clean_counters():
    Node._ids = count(-1)


if __name__ == "__main__":
    test_backjump_complicated()
    test_learn_conflict_complicated()
    test_learn_conflict_simple()
    test_first_uip_complicated()
    # Node._ids = count(-1)
    # test_find_all_paths()
    # test_first_uip_simple()
    # test_first_uip_complicated()
    # test_boolean_resolution()
