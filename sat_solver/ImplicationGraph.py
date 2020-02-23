from collections import defaultdict, OrderedDict
from copy import copy
from typing import List, Dict

from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import Literal, Variable
from itertools import count

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

class ImplicationGraph(object):
    def __init__(self, cnf_formula: CnfFormula):
        self._nodes = {}  # type: Dict[Variable, Node]
        self._edges = {}  # type: Dict[Node, List[Edge]]
        self._incoming_edges = defaultdict(list) # type: Dict[Node, List[Edge]]
        self._conflict_var = Literal(Variable('Conflict'), False)
        self._conflict_node = Node(self._conflict_var, -1)
        # self._nodes[self._conflict_var] = self._conflict_node
        self._last_decide_node = None
        self._formula = cnf_formula
        # self.edges[0] = []

    def remove_level(self, level):
        # TODO: Implement if needed
        pass

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
        self._edges[new_node] = []

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
        for l in clause:
            if self._nodes[l.variable].level > max_level:
                last_literal = l
                max_level = self._nodes[l.variable].level
        return last_literal

    @staticmethod
    def boolean_resolution(c1: List[Literal], c2: List[Literal], shared_var: Variable) -> List[Literal]:
        c = c1 + c2
        return [l for l in c if l.variable != shared_var]

    def learn_conflict(self, last_assigned: Literal, formula_idx: int) -> List[Literal]:
        # add the conflict node
        first_uip = self.find_first_uip(formula_idx)
        if first_uip is None:
            return []
        uip_negate = copy(first_uip.literal).negate()
        node = self._nodes[last_assigned.variable]
        c = self._formula.clauses[formula_idx]  # conflict clause
        while True:
            shared_var = node.literal.variable
            clause_on_edge = self._formula.clauses[self._incoming_edges[node][-1].reason]  # c' in the slides
            new_clause = self.boolean_resolution(c, clause_on_edge, shared_var)
            if uip_negate in new_clause:
                return new_clause

            c = new_clause
            # TODO: The presentation (lec 3 slide 29) says to pick the next node as one of the incoming to the new_clause
            # but that might create a loop (as in test_learn_conflict_simple case 2)
            node = self._nodes[self.find_last_assigned_literal(clause_on_edge).variable]

    def find_first_uip(self, formula_idx: int) -> Node:
        if self._last_decide_node is None:
            return None
        for lit in self._formula.clauses[formula_idx]:
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
