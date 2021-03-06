from collections import defaultdict
from copy import copy, deepcopy
from random import randint
from typing import Optional, List, Dict, Callable

from sat_solver.ImplicationGraph import ImplicationGraph
from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import Literal, Variable


class DPLL(object):
    def __init__(self, cnf_formula: CnfFormula, partial_assignment=None, watch_literals=None, implication_graph=None,
                 is_first_run=True,
                 propagate_helper: Callable[[Dict[Variable, bool]], Optional[Dict[Variable, bool]]] = None,
                 conflict_helper:  Callable[[Dict[Variable, bool]], List[Literal]] = None, rec_num=None):
        self._assignment = partial_assignment or [{}]
        self.watch_literals = watch_literals or defaultdict(set)
        self.implication_graph = implication_graph or ImplicationGraph(cnf_formula)  # implication_graph or {}
        self.rec_num = rec_num or None

        self.formula = cnf_formula
        self.conflict_helper = conflict_helper
        self.propagate_helper = propagate_helper
        self.learned_sat_conflicts = None
        self.backjump = None

        if self.watch_literals == {} and self.formula:
            self.initialize_watch_literals()

        # if self.formula is not None:
        #     self.formula = self.unit_propagation(self.formula)

        self.unsat = None
        if is_first_run:
            self.rec_num = 0
            self.unit_propagation()

    def initialize_watch_literals(self):
        '''
        for each clause pick two random literals that will be the watchers (update self.watch_literals)
        '''

        for i, clause in enumerate(self.formula.clauses):
            self.create_watch_literals(i, clause)

    def create_watch_literals(self, cluse_idx: int, clause: List[int] = []):
        if clause == []:
            clause = self.formula.clauses[cluse_idx]
        if len(clause) == 0:
            return
        if len(clause) == 1:
            self.watch_literals[clause[0]].add(cluse_idx)
            return
        first_lit = randint(0, len(clause) - 1)
        second_lit = first_lit
        while second_lit == first_lit:
            second_lit = randint(0, len(clause) - 1)
        self.watch_literals[clause[first_lit]].add(cluse_idx)
        self.watch_literals[clause[second_lit]].add(cluse_idx)

    def get_full_assignment(self) -> Dict[Variable, bool]:
        # if self.unsat is None or self.unsat is True:
        #     # You should first run search
        #     return None
        full_assignment = copy(self._assignment[-1])
        if self.formula:
            for v in self.formula.get_variables():
                if v not in full_assignment:
                    full_assignment[v] = True
        return full_assignment

    def get_partial_assignment(self) -> Dict[Variable, bool]:
        return self._assignment[-1]

    def remove_clauses_by_assignment(self, literal: Literal):
        # We decided that lit is True (lit might be ~x_1, then x_1 is False)
        # Remove all clauses that has lit (they are satisfied), and delete ~lit from all other clauses
        removed_clauses = set(self.formula.literal_to_clauses[literal])
        return self.formula.remove_clause(removed_clauses)

    def assign_true_to_literal(self, literal: Literal, reason: Optional[int]):
        '''
        Assigns true to the given literal, i.e. remove all clauses that this literal is in
        And remove the negation of this literal from all clauses, if it is the only literal return None
        :param literal: literal to assign to
        :param reason: the clause that caused this assignment, if None the reason is decided
        :return: None if can't do this assignment, otherwise a new formula (also editing the self._formulas[-1])
        '''
        if self.formula is None:
            return None

        assert literal.variable not in self._assignment[-1], literal.variable
        self._assignment[-1][literal.variable] = not literal.negated

        # Add a node to the implication graph

        level = len(self._assignment) - 1

        if reason is None:
            self.implication_graph.add_decide_node(level, literal)
        else:
            self.implication_graph.add_node(level, literal, self.formula.clauses[reason], reason)

        if not self.remove_clauses_by_assignment(literal):
            self.formula.clauses = [[]]
            return self.formula

        clause_unsatisfied = self._check_unsat(literal)
        if clause_unsatisfied is not None:
            # One of the clauses can not be satisfied with the current assignment, learn the conflict and go up the tree
            conflict_clause = self.implication_graph.learn_conflict(literal, clause_unsatisfied)
            self.learned_sat_conflicts = conflict_clause
            self.backjump = self.implication_graph.get_backjump_level(conflict_clause)
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
            if clause >= len(self.formula.clauses):
                continue
            clause_unsatisfied = True if self.formula.clauses[clause] != [] else False
            for lit in self.formula.clauses[clause]:
                if lit.variable not in self._assignment[-1]:
                    # One of the literals in the clause is assigned, skip to next clause
                    clause_unsatisfied = False
                    break
                else:
                    # Make sure there is no True assignment that we forgot to delete
                    # assert self._assignment[-1][lit.variable] == lit.negated
                    pass

            if clause_unsatisfied:
                # all literals are already assigned, this clause will not be satisfied
                return clause

        return None

    def _unit_propagation_watch_literals(self, literal: Literal) -> CnfFormula:
        '''
        Perform unit propoagation on the given cnf formula (editing the formula)
        :return the formula after unit propogation
        '''

        if self.formula is None:
            return None
        n_literal = copy(literal).negate()

        learn_assingments = []
        for sub_formula_idx in self.watch_literals[n_literal]:
            if sub_formula_idx >= len(self.formula.clauses):
                continue
            hope = True
            other_watch = None
            for lit in self.formula.clauses[sub_formula_idx]:
                if lit.variable not in self._assignment[-1]:
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
                learn_assingments.append((other_watch, sub_formula_idx))
        del self.watch_literals[n_literal]
        if self.formula is None or self.formula.clauses == [[]]:
            return self.formula
        for (other_watch, sub_formula_idx) in learn_assingments:
            if other_watch.variable not in self._assignment[-1]:
                self.formula = self.assign_true_to_literal(other_watch, sub_formula_idx)
        return self.formula

    def unit_propagation(self):
        '''
        Perform unit propoagation on the given cnf formula (editing the formula)
        :return the formula after unit propogation
        '''
        if self.formula is None:
            return None

        found = True
        while found:
            found = False
            for i, sub_forumla in enumerate(self.formula.clauses):
                 if len(sub_forumla) == 1:
                    found = True
                    lit = sub_forumla[0]
                    # assert lit.name not in self._assignment
                    self.formula = self.assign_true_to_literal(lit, reason=i)
                    self._unit_propagation_watch_literals(lit)
                    if self.formula is None:
                        return self.formula

        return self.formula

    def decision_heuristic(self):
        '''
        Implement the Dynamic Largest Individual Sum (DLIS) heuristic
        :return: the literal which has maximum appears in clauses
        '''
        return self.formula.get_literal_appears_max(self._assignment[-1])

    def check_theory_conflict(self) -> bool:
        if self.conflict_helper:
            smt_conflict = self.conflict_helper(self._assignment[-1])
            if smt_conflict:
                self.formula.add_clause(smt_conflict)
                self.create_watch_literals(len(self.formula.clauses) - 1)
                return False
        return True

    def search(self) -> bool:
        '''
        :return: True if Sat else False
        '''

        self.rec_num += 1
        if self.formula is None:
            # Formula is unsat
            return False
        real_formula = [f for f in self.formula.clauses if f != []]
        if not real_formula:
            if not self.check_theory_conflict():
                # The current assignment satisfies all clauses but theory finds a conflict
                return False
            # Found valid assignment
            self.unsat = False
            return True

        # it and try again, the second literal might not be very helpful
        self.check_theory_conflict()
        if self.propagate_helper:
            smt_res = self.propagate_helper(self._assignment[-1])
            # smt_res will be None if partial_assignment is empty
            if smt_res:
                self._assignment[-1].update(smt_res)
                for variable, val in smt_res.items():
                    if val:
                        lit = Literal(variable, False)
                    else:
                        lit = Literal(variable, True)

                    if not self.remove_clauses_by_assignment(lit):
                        self.formula.clauses = [[]]
                        return self.formula

                if not smt_res:
                    # Conflict
                    return False

        next_decision_lit = self.decision_heuristic()
        if next_decision_lit is None:
            return False
        assert isinstance(next_decision_lit, Literal)
        next_decision = next_decision_lit.variable
        assert isinstance(next_decision, Variable)

        assignment_order = [False, True] if next_decision_lit.negated else [True, False]
        for d in assignment_order:
            if self.backjump and self.backjump > len(self._assignment) - 1:
                # Skip this level
                self.implication_graph.remove_level(len(self._assignment) - 1)
                return False
            elif self.backjump == len(self._assignment) - 1:
                # Finished to do backjump
                self.backjump = None
            # current_assignment = copy(self._assignment)
            cur_assignment = self._assignment
            cur_assignment.append(copy(self._assignment[-1]))


            dpll = DPLL(deepcopy(self.formula), watch_literals=self.watch_literals, partial_assignment=cur_assignment,
                        implication_graph=self.implication_graph, is_first_run=False,
                        conflict_helper=self.conflict_helper, propagate_helper=self.propagate_helper, rec_num=self.rec_num)
            dpll.assign_true_to_literal(Literal(next_decision, not d), reason=None)

            sat = dpll.search()
            if dpll.learned_sat_conflicts:
                self.formula.add_clause(dpll.learned_sat_conflicts)
                self.create_watch_literals(len(self.formula.clauses) - 1)
                self.backjump = dpll.backjump

            if sat:
                if not self.check_theory_conflict():
                    # The current assignment satisfies all clauses but theory finds a conflict
                    return False
                self.unsat = False
                return sat

            # clear the last decision
            decision_level = len(self._assignment) - 1
            self.implication_graph.remove_level(decision_level)
            self._assignment.pop()

        return False
