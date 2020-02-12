from collections import defaultdict
from copy import copy, deepcopy
from random import randint

from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import Literal, Variable


# TODO: Do we need to implement also pure literal?
class DPLL(object):
    def __init__(self, cnf_forumla: CnfFormula, partial_assignment=None, watch_literals=defaultdict(set),
                 is_first_run=True):
        partial_assignment = partial_assignment or [{}]
        watch_literals = watch_literals or defaultdict(set)
        self.formula = cnf_forumla
        self.watch_literals = watch_literals
        if self.watch_literals == {} and self.formula:
            self.initialize_watch_literals()

        self._assignment = partial_assignment
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

    def assign_true_to_literal(self, literal: Literal):
        '''
        Assigns true to the given literal, i.e. remove all clauses that this literal is in
        And remove the negation of this literal from all clauses, if it is the only literal return None
        :param literal:
        :return: None if can't do this assignment, otherwise a new formula (also editing the self._formulas[-1])
        '''
        if literal.name not in self._assignment[-1]:
            self._assignment[-1][literal.name] = not literal.negated
        else:
            raise Exception

        # We decided that lit is True (lit might be ~x_1, then x_1 is False)
        # Remove all clauses that has lit (they are satisfied), and delete ~lit from all other clauses
        removed_clauses = set(self.formula.literal_to_clauses[literal])
        if not self.formula.remove_clause(removed_clauses):
            return None

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
        other_watch = None
        if self.formula is None:
            return None
        n_literal = copy(literal).negate()

        for sub_formula_idx in self.watch_literals[n_literal]:
            hope = True
            for lit in self.formula.clauses[sub_formula_idx]:
                if lit.variable.name not in self._assignment[-1]:
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
                self.formula = self.assign_true_to_literal(other_watch)
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
            for sub_forumla in self.formula.clauses:
                if len(sub_forumla) == 1:
                    found = True
                    lit = sub_forumla[0]
                    # assert lit.name not in self._assignment
                    self.formula = self.assign_true_to_literal(lit)
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
        if self.formula is None or self.formula == []:
            # Formula is unsat
            return False
        real_formula = [f for f in self.formula.clauses if f != []]
        if real_formula == []:
            # Found valid assinment
            self.unsat = False
            return True

        # current_variables = self.formula.get_variables()
        next_decision = self.decision_heuristic()
        assert isinstance(next_decision, Variable)

        for d in [True, False]:
            # current_assignment = copy(self._assignment)
            current_assignment = self._assignment
            current_assignment.append(copy(self._assignment[-1]))
            dpll = DPLL(deepcopy(self.formula), watch_literals=self.watch_literals,
                        partial_assignment=current_assignment, is_first_run=False)
            dpll.assign_true_to_literal(Literal(next_decision, not d))
            # formula = self.unit_propagation(formula)

            sat = dpll.search()
            if sat:
                self.unsat = False
                return sat
            self._assignment.pop()

        return False
