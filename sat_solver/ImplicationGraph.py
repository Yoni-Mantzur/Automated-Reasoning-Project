from collections import defaultdict
from typing import List, Dict

from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import Literal, Variable

class Node(object):
    def __init__(self, literal: Literal, level: int):
        '''
        Create a node, the literal is used as a unique id, the level is just a property we might use
        '''
        self.literal = literal
        self.level = level

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
        uip_negate = first_uip.literal.negate()
        node = self._nodes[last_assigned.variable]
        while True:
            shared_var = node.literal.variable
            clause_on_edge = self._formula.clauses[self._incoming_edges[node][-1].reason]
            new_clause = self.boolean_resolution(self._formula.clauses[formula_idx], clause_on_edge, shared_var)
            if uip_negate in new_clause:
                return new_clause

            node = self._nodes[self.find_last_assigned_literal(new_clause).variable]

    # def find_first_uip(self, formula_idx: int) -> Node:
    #     if self._last_decide_node is None:
    #         return None
    #
    #     # Add the conflict node and the edges
    #     for lit in self._formula.clauses[formula_idx]:
    #         n = self._nodes[lit.variable]
    #         e = Edge(n, self._conflict_node, reason=formula_idx)
    #         self._edges[n].append(e)
    #         self._incoming_edges[self._conflict_node] = e
    #
    #

    def find_first_uip(self, formula_idx: int) -> Node:
        if self._last_decide_node is None:
            return None
        for lit in self._formula.clauses[formula_idx]:
            n = self._nodes[lit.variable]
            e = Edge(n, self._conflict_node, reason=formula_idx)
            self._edges[n].append(e)
            self._incoming_edges[self._conflict_node] = e

        paths = self._find_all_paths(self._last_decide_node, self._conflict_node)
        other_paths = [set(p) for p in paths[1:]]
        for path in paths:
            assert path[-1].name == 'Conflict'
            # reverse path and ignore the last node which should be conflict
            for node in path[:-1][::-1]:
                in_other = [node in other for other in other_paths]
                if all(in_other):
                    return self._nodes[node]

    def _find_all_paths(self, source: Node, target: Node):
        if source == target:
            return [[source.literal.variable]]
        paths = []
        for e in self._edges[source]:
            t = self._find_all_paths(e.target, target)
            t = [v for ls in t for v in ls]
            paths.append([source.literal.variable] + t)

        # if isinstance(paths[0][0], list):
        #     return [v for ls in paths for v in ls]
        # else:
        return paths

