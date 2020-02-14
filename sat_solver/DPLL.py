from collections import defaultdict
from copy import copy, deepcopy
from random import randint
from typing import Optional

from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import Literal, Variable


# TODO: Do we need to implement also pure literal?
class DPLL(object):
    def __init__(self, cnf_forumla: CnfFormula, partial_assignment=None, watch_literals=None, implication_graph=None,
                 implication_graph_root=None, is_first_run=True):
        self._assignment = partial_assignment or [{}]
        self.watch_literals = watch_literals or defaultdict(set)
        self.implication_graph = implication_graph or {}
        self.implication_graph_root = implication_graph_root or [0]
        self.formula = cnf_forumla

        if self.watch_literals == {} and self.formula:
            self.initialize_watch_literals()


        # if self.formula is not None:
        #     self.formula = self.unit_propagation(self.formula)

        self.unsat = None
        if is_first_run:
            self.unit_propagation()

    def initialize_watch_literals(self):
        '''
        for each clause pick two random literals that will be the watchers (update self.watch_literals)
        '''

        for i, clause in enumerate(self.formula.clauses):
            if len(clause) <= 1:
                self.watch_literals[clause[0]].add(i)
                continue
            first_lit = randint(0, len(clause) - 1)
            second_lit = first_lit
            while second_lit == first_lit:
                second_lit = randint(0, len(clause) - 1)
            self.watch_literals[clause[first_lit]].add(i)
            self.watch_literals[clause[second_lit]].add(i)

    def get_full_assignment(self):
        # if self.unsat is None or self.unsat is True:
        #     # You should first run search
        #     return None
        full_assignment = copy(self._assignment[-1])
        if self.formula:
            for v in self.formula.get_variables():
                if v.name not in full_assignment:
                    full_assignment[v.name] = True
        return full_assignment

    def assign_true_to_literal(self, literal: Literal, reason: Optional[int]):
        '''
        Assigns true to the given literal, i.e. remove all clauses that this literal is in
        And remove the negation of this literal from all clauses, if it is the only literal return None
        :param literal: literal to assign to
        :param reason: the clause that caused this assignment, if None the reason is decided
        :return: None if can't do this assignment, otherwise a new formula (also editing the self._formulas[-1])
        '''
        assert literal.name not in self._assignment[-1]
        self._assignment[-1][literal.name] = not literal.negated

        # Add a node to the implication graph

        level = len(self._assignment) - 1
        if level == 0:
            self.implication_graph[(0,0)] = []
        if reason is None:

            self.implication_graph[(literal.name, level)] = []
            self.implication_graph_root.append(literal.name)
            assert len(self.implication_graph_root) - 1 == level
        else:
            root_node = (self.implication_graph_root[level], level)
            self.implication_graph[root_node].append({(literal.name, level): reason})

        # We decided that lit is True (lit might be ~x_1, then x_1 is False)
        # Remove all clauses that has lit (they are satisfied), and delete ~lit from all other clauses
        removed_clauses = set(self.formula.literal_to_clauses[literal])
        if not self.formula.remove_clause(removed_clauses):
            # Indicate the query is SAT
            self.formula.clauses = [[]]
            return self.formula

        clause_unsatisfied = self._check_unsat(literal)
        if clause_unsatisfied is not None:
            # One of the clauses can not be satisfied with the current assignment
            return None

        self.formula = self._unit_propagation_watch_literals(literal)
        return self.formula

    def _check_unsat(self, literal: Literal):
        '''
        Check if one of the clauses is UNSAT after assingng a value to the given literal
        :param literal: The literal we just assigned
        :return: clause number if it is know UNSAT, otherwise None (otherwise == all clauses can still be satisfied)
        '''
        not_lit = copy(literal).negate()
        for i, clause in enumerate(self.watch_literals[not_lit]):
            clause_unsatisfied = True if self.formula.clauses[clause] != [] else False
            for lit in self.formula.clauses[clause]:
                if lit.name not in self._assignment[-1]:
                    # One of the literals in the clause is assigned, skip to next clause
                    clause_unsatisfied = False
                    break
                else:
                    # Make sure there is no True assignment that we forgot to delete
                    assert self._assignment[-1][lit.name] == lit.negated

            if clause_unsatisfied:
                # all literals are already assigned, this clause will not be satisfied
                return self.watch_literals[i]

        return None

    def _unit_propagation_watch_literals(self, literal: Literal):
        '''
        Perform unit propoagation on the given cnf formula (editing the formula)
        :return the formula after unit propogation
        '''

        if self.formula is None:
            return None
        n_literal = copy(literal).negate()

        for sub_formula_idx in self.watch_literals[n_literal]:
            hope = True
            other_watch = None
            for lit in self.formula.clauses[sub_formula_idx]:
                if lit.name not in self._assignment[-1]:
                    if sub_formula_idx in self.watch_literals[lit]:
                        other_watch = lit
                    else:
                        # No hope, there is at least one variable that is not assigned and is not the watch,
                        # set it as the new watch_literal
                        hope = False
                        self.watch_literals[lit].add(sub_formula_idx)
                        break
            # only other_watch and the current assigned literal are not assigned, assign the other_watch
            if other_watch is not None and hope:
                self.formula = self.assign_true_to_literal(other_watch, sub_formula_idx)
            if self.formula is None:
                return self.formula
        return self.formula

    def unit_propagation(self):
        '''
        Perform unit propoagation on the given cnf formula (editing the formula)
        :return the formula after unit propogation
        '''
        if self.formula is None:
            return None

        found = True
        while found == True:
            found = False
            for i, sub_forumla in enumerate(self.formula.clauses):
                if len(sub_forumla) == 1:
                    found = True
                    lit = sub_forumla[0]
                    # assert lit.name not in self._assignment
                    self.formula = self.assign_true_to_literal(lit, reason=i)
                    if self.formula is None:
                        return self.formula

        # for i, clause in enumerate(self.formula.clauses):
        #     clause_unsatisfied = True
        #     for lit in clause:
        #         if lit.name not in self._assignment[-1]:
        #             # One of the literals in the clause is assigned, skip to next clause
        #             clause_unsatisfied = False
        #             break
        #         else:
        #             # Make sure there is no True assignment that we forgot to delete
        #             assert self._assignment[-1][lit.name] == lit.negated
        #
        #     if clause_unsatisfied:
        #         # all literals are already assigned, this clause will not be satisfied
        #         return None

        return self.formula

    def decision_heuristic(self):
        '''
        Implement the Dynamic Largest Individual Sum (DLIS) heuristic
        :return: the literal which has maximum appears in clauses
        '''
        return self.formula.get_literal_appears_max(self._assignment[-1])

    def search(self):
        # if self.formula == []:
        #     # Forumla is UNSAT
        #     return True
        if self.formula is None:
            # Formula is unsat
            return False
        real_formula = [f for f in self.formula.clauses if f != []]
        if real_formula == []:
            # Found valid assinment
            self.unsat = False
            return True

        # current_variables = self.formula.get_variables()
        # TOOD: does this heuristic make sense? we try the literal which appears the most, but if it is UNSAT we negate
        # it and try again, the second literal might not be very helpful
        next_decision_lit = self.decision_heuristic()
        assert isinstance(next_decision_lit, Literal)
        next_decision = next_decision_lit.variable
        assert isinstance(next_decision, Variable)

        assignment_order = [True, False] #[False, True] if next_decision_lit.negated else [True, False]
        for d in assignment_order:
            # current_assignment = copy(self._assignment)
            cur_assignment = self._assignment
            cur_assignment.append(copy(self._assignment[-1]))
            dpll = DPLL(deepcopy(self.formula), watch_literals=self.watch_literals, partial_assignment=cur_assignment,
                        implication_graph=self.implication_graph, implication_graph_root=self.implication_graph_root,
                        is_first_run=False)
            dpll.assign_true_to_literal(Literal(next_decision, not d), reason=None)
            # formula = self.unit_propagation(formula)

            sat = dpll.search()
            if sat:
                self.unsat = False
                return sat

            # clear the last decision
            decision_level = len(self._assignment) - 1
            # decision_literal = self.implication_graph_root[decision_level] if decision_level > 0 else 0
            # del self.implication_graph[(decision_literal, decision_level)]
            self.implication_graph_root.pop(decision_level)
            self._assignment.pop()

        return False
