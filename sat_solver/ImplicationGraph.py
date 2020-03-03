from collections import defaultdict, OrderedDict
from copy import copy
from itertools import count
from typing import List, Dict, Set

from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import Literal, Variable


class Node(object):
    _ids = count(-1)
    next(_ids)

    def __init__(self, literal: Literal, level: int):
        '''
        Create a node, the literal is used as a unique id, the level is just a property we might use
        '''
        self.literal = literal
        self.level = level
        self.id = next(self._ids)

    def __eq__(self, other):
        return self.literal.variable == other.literal.variable

    def __hash__(self):
        return hash(self.literal.variable)

    def __str__(self):
        return "{}:{}".format(str(self.literal), self.level)

    def __repr__(self):
        return str(self)

class Edge(object):
    def __init__(self, source: Node, target: Node, reason: int):
        '''

        :param source: source node
        :param target: target node
        :param reason: clause number that caused this edge
        '''
        assert source != target
        self.source = source
        self.target = target
        self.reason = reason

    def __str__(self):
        return "{} -> {} ({})".format(self.source, self.target, self.reason)

    def __repr__(self):
        return str(self)

class ImplicationGraph(object):
    def __init__(self, cnf_formula: CnfFormula):
        self._nodes = {}  # type: Dict[Variable, Node]
        self._edges = {}  # type: Dict[Node, List[Edge]]
        self._nodes_order = {}  # type: Dict[Variable, int]
        self._level_to_nodes = defaultdict(set)  # type: Dict[int, Set[Variable]]
        self._incoming_edges = defaultdict(list)  # type: Dict[Node, List[Edge]]
        self._conflict_var = Literal(Variable('Conflict'), False)
        self._conflict_node = Node(self._conflict_var, -1)
        self._last_decide_node = None
        self._formula = cnf_formula

    def remove_level(self, level):
        for v in self._level_to_nodes[level]:
            n = self._nodes[v]
            for e in self._edges[n]:
                self._incoming_edges[e.target].remove(e)
            del self._edges[n]
            del self._nodes[v]
        # Remove also all edges to conflict, might be that we backjump to level 11 but the conflict involved something
        # from level 10
        for e in self._incoming_edges[self._conflict_node]:
            if e.source in self._edges and e in self._edges[e.source]:
                self._edges[e.source].remove(e)
        del self._level_to_nodes[level]

    def add_decide_node(self, level, literal) -> None:
        '''
        Add node that origined from a decide operation
        :param level: when the decided was made (Zero based count on number of desecions)
        :param literal: The literal we are doing the decided on
        '''
        # if isinstance(variable, Literal):
        #     variable = variable.variable
        assert isinstance(literal, Literal)
        root_node = Node(literal, level)
        self._nodes[literal.variable] = root_node

        self._nodes_order[root_node.literal.variable] = len(self._nodes.keys())
        self._level_to_nodes[level].add(root_node.literal.variable)
        self._edges[root_node] = []
        self._last_decide_node = root_node

    def add_node(self, level: int, literal: Literal, reason_formula: List[Variable], reason_idx: int) -> None:
        # if isinstance(variable, Literal):
        #     variable = variable.variable

        assert isinstance(literal, Literal)
        if reason_formula is None:
            reason_formula = self._formula.clauses[reason_idx]
        new_node = Node(literal, level)
        self._nodes[literal.variable] = new_node
        self._nodes_order[new_node.literal.variable] = len(self._nodes.keys())
        self._edges[new_node] = []
        self._level_to_nodes[level].add(new_node.literal.variable)
        for reason_literal in reason_formula:
            reason_variable = reason_literal.variable
            assert isinstance(reason_variable, Variable)
            if literal.variable == reason_variable:
                continue
            reason_node = self._nodes[reason_variable]
            e = Edge(reason_node, new_node, reason=reason_idx)
            self._edges[reason_node].append(e)
            self._incoming_edges[new_node].append(e)

    def find_last_assigned_literal(self, clause: List[Literal]) -> Literal:
        max_level = -1
        last_literal = None
        for lit in clause:
            if self._nodes_order[lit.variable] > max_level and lit.variable in self._nodes:
                last_literal = lit
                max_level = self._nodes_order[lit.variable]
        return last_literal

    def get_backjump_level(self, conflict_clause: List[Literal]) -> int:
        '''
        Get the second highest decision level in the conflict clause
        if only one decision was made, zero
        '''
        max_level = 0
        second_max = 0

        levels = set(self.get_decision_levels(conflict_clause))
        for level in levels:
            if level > max_level:
                second_max = max_level
                max_level = level
            elif level > second_max:
                second_max = level
        return second_max

    @staticmethod
    def boolean_resolution(c1: List[Literal], c2: List[Literal], shared_var: Variable) -> List[Literal]:
        c = c1 + c2
        return list(set(l for l in c if l.variable != shared_var))

    def get_decision_levels(self, clause: List[Literal]):
        return [self._nodes[lit.variable].level for lit in clause if lit.variable in self._nodes]

    def is_two_literals_in_level(self, level: int, clause: List[Literal]) -> bool:
        '''
        Check whether there are two literals in the clause that come from the given decision_level
        '''
        found_first = False
        for lit in clause:
            if self._nodes[lit.variable].level == level:
                if found_first:
                    return True
                else:
                    found_first = True
        return False

    def learn_conflict(self, last_assigned: Literal, formula_idx: int) -> List[Literal]:
        # add the conflict node
        first_uip = self.find_first_uip(formula_idx)
        if first_uip is None:
            return []
        uip_negate = copy(first_uip.literal).negate()
        uip_level = first_uip.level
        node = self._nodes[last_assigned.variable]
        clause = self._formula.clauses[formula_idx]
        if not self._incoming_edges[node]:
            # No incoming edges to the node, can't resolve the conflict
            return clause

        # clause = c
        while uip_negate not in clause or self.is_two_literals_in_level(uip_level, clause):
            shared_var = node.literal.variable
            clause_on_edge = self._formula.clauses[self._incoming_edges[node][-1].reason]  # c' in the slides
            clause = self.boolean_resolution(clause, clause_on_edge, shared_var)
            # TODO: The presentation (lec 3 slide 29) says to pick the next node as one of the incoming to the new_clause
            # but that might create a loop (as in test_learn_conflict_simple case 2)
            # node = self._nodes[self.find_last_assigned_literal(clause_on_edge).variable]

            node = self._nodes[self.find_last_assigned_literal(clause).variable]
        return clause


    def find_first_uip(self, formula_idx: int) -> Node:
        if self._last_decide_node is None or formula_idx >= len(self._formula.clauses):
            return None
        for lit in self._formula.clauses[formula_idx]:
            if lit.variable not in self._nodes:
                continue
            n = self._nodes[lit.variable]
            e = Edge(n, self._conflict_node, reason=formula_idx)
            self._edges[n].append(e)
            self._incoming_edges[self._conflict_node].append(e)

        paths = self._find_all_paths(self._last_decide_node, self._conflict_node)
        other_paths = paths[1:]
        # After finding all paths we need to find first unique implication point
        # Do that looking for a node that apperas in all paths, we do so be walking on one of the paths in reverse
        for path in paths:
            path = list(path.keys())
            assert path[-1].name == 'Conflict'
            # reverse path and ignore the last node which should be conflict
            for node in path[:-1][::-1]:
                in_other = [node in other for other in other_paths]
                if all(in_other):
                    return self._nodes[node]

    def _find_all_paths(self, source: Node, target: Node):
        '''
        Find all paths between source and target in linear time using dynamic programing
        '''
        partial_paths = defaultdict(list)
        partial_paths[source.id].append({source.literal.variable: None})

        def _find_all_path_rec(source: Node, target: Node, partial_paths: List[List[OrderedDict]]):
            for e in self._incoming_edges[target]:
                if not partial_paths[e.source.id]:
                    _find_all_path_rec(source, e.source, partial_paths)
                for path in partial_paths[e.source.id]:
                    new_path = copy(path)
                    new_path.update({target.literal.variable: None})
                    partial_paths[target.id].append(new_path)

        _find_all_path_rec(source, target, partial_paths)
        return partial_paths[target.id]

    def __str__(self):
        s = ""
        for n in self._nodes_order:
            # if n in self._nodes:
            node = self._nodes[n] if n in self._nodes else None
            if node in self._edges:
                s += "{}: {}\n".format(n, self._edges[node])


        return s