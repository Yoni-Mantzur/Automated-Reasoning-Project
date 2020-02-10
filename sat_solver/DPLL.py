from collections import defaultdict
from copy import copy, deepcopy
from random import randint
from typing import Dict, Set

from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import Literal, Variable


# TODO: Do we need to implement also pure literal?
class DPLL(object):
    def __init__(self, cnf_forumla: CnfFormula, partial_assignment={},
                 watch_literals: Dict[Literal, Set[int]] = defaultdict(set)):
        self.formula = cnf_forumla
        self.watch_literals = watch_literals
        if self.watch_literals == {}:
            self.initialize_watch_literals()

        self._assignment = partial_assignment
        if self.formula is not None:
            self.formula = self.unit_propagation(self.formula)

        self.unsat = None

    def initialize_watch_literals(self):
        '''
        for each clause pick two random literals that will be the watchers (update self.watch_literals)
        '''

        for i, clause in enumerate(self.formula.clauses):
            if len(clause) <= 1:
                continue
            first_lit = randint(0, len(clause)-1)
            second_lit = first_lit
            while second_lit == first_lit:
                second_lit = randint(0, len(clause)-1)
            self.watch_literals[clause[first_lit]].add(i)
            self.watch_literals[clause[second_lit]].add(i)

    def get_assignment(self):
        # if self.unsat is None or self.unsat is True:
        #     # You should first run search
        #     return None
        if self.formula:
            for v in self.formula.get_variables():
                if v.name not in self._assignment:
                    self._assignment[v.name] = True
        return self._assignment

    def _assign_true_to_literal(self, formula: CnfFormula, literal: Literal):
        '''
        Assigns true to the given literal, i.e. remove all clauses that this literal is in
        And remove the negation of this literal from all clauses, if it is the only literal return None
        :param literal:
        :return: None if can't do this assignment, otherwise a new formula (also editing the self._formulas[-1])
        '''
        not_lit = copy(literal).negate()
        # assert literal.name not in self._assignment
        self._assignment[literal.name] = not literal.negated
        # We decided that lit is True (lit might be ~x_1, then x_1 is False)
        # Remove all clauses that has lit (they are satisfied), and delete ~lit from all other clauses
        removed_clauses = set(formula.literal_to_clauses[literal])
        formula.remove_clause(removed_clauses)

        # TODO: This does not make sense but how otherwise can I understand that a clause not satisfied
        for clause in formula.literal_to_clauses[not_lit]:
            clause_unsatisfied = True
            for lit in self.formula.clauses[clause]:
                if lit.name not in self._assignment:
                    clause_unsatisfied = False
                    break
            # all literals are assinged, and the clause is still present
            if clause_unsatisfied:
                return None

        formula = self._unit_propagation_watch_literals(formula, literal)
        return formula

    def _unit_propagation_watch_literals(self, formula: CnfFormula, literal: Literal):
        '''
        Perform unit propoagation on the given cnf formula (editing the formula)
        :return the formula after unit propogation
        '''
        other_watch = None
        if formula is None:
            return None
        n_literal = copy(literal).negate()

        for sub_formula_idx in self.watch_literals[n_literal]:
            hope = True
            for lit in self.formula.clauses[sub_formula_idx]:
                if lit.variable.name not in self._assignment:
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
                formula = self._assign_true_to_literal(formula, other_watch)
            if formula is None:
                return formula
        return formula

    def unit_propagation(self, formula: CnfFormula):
        '''
        Perform unit propoagation on the given cnf formula (editing the formula)
        :return the formula after unit propogation
        '''
        if formula is None:
            return None

        found = True
        while found == True:
            found = False
            for sub_forumla in formula.clauses:
                if len(sub_forumla) == 1:
                    found = True
                    lit = sub_forumla[0]
                    # assert lit.name not in self._assignment
                    formula = self._assign_true_to_literal(formula, lit)
                    if formula is None:
                        return formula

        return formula

    def decision_heuristic(self):
        '''
        Implement the Dynamic Largest Individual Sum (DLIS) heuristic
        :return: the literal which has maximum appears in clauses
        '''
        return self.formula.get_literal_appears_max(self._assignment)

    def search(self):
        if self.formula is None:
            # Formula is unsat
            return False
        real_formula = [f for f in self.formula.clauses if f != []]
        if real_formula == []:
            # Found valid assinment
            self.unsat = False
            return True

        # current_variables = self.formula.get_variables()
        next_decision = self.decision_heuristic()

        for d in [True, False]:
            formula = deepcopy(self.formula)
            assert isinstance(next_decision, Variable)
            formula = self._assign_true_to_literal(formula, Literal(next_decision, d))
            # formula = self.unit_propagation(formula)

            sat = DPLL(formula, partial_assignment=self._assignment, watch_literals=self.watch_literals).search()
            if sat:
                self.unsat = False
                return sat

        return False
