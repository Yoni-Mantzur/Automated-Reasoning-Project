from copy import copy, deepcopy

from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import Literal


# TODO: Do we need to implement also pure literal?
class DPLL(object):
    def __init__(self, cnf_forumla: CnfFormula, partial_assignment={}):
        self.formula = cnf_forumla

        self._assignment = partial_assignment
        if self.formula is not None:
            self.formula = self.unit_propagation(self.formula)

        self.unsat = None

    def get_assignment(self):
        if self.unsat is None or self.unsat is True:
            # You should first run search
            return None

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
        negate_lit = copy(literal).negate()
        self._assignment[literal.name] = not literal.negated
        # We decided that lit is True (lit might be ~x_1, then x_1 is False)
        # Remove all clauses that has lit (they are satisfied), and delete ~lit from all other clauses
        removed_clauses = set(formula.literal_to_clauses[literal])
        formula.remove_clause(removed_clauses)

        for idx in formula.literal_to_clauses[negate_lit]:
            # if len(formula.clauses[idx]) == 0:
            #     continue
            if not formula.remove_literal(idx, negate_lit):
                return None

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
                    formula = self._assign_true_to_literal(formula, lit)
                    if formula is None:
                        return formula

        return formula

    def decision_heuristic(self):
        '''
        Implement the Dynamic Largest Individual Sum (DLIS) heuristic
        :return: the literal which has maximum appears in clauses
        '''
        return self.formula.get_literal_appears_max()

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
            formula = self._assign_true_to_literal(formula, Literal(next_decision, d))
            formula = self.unit_propagation(formula)

            sat = DPLL(formula, self._assignment).search()
            if sat:
                self.unsat = False
                return sat

        return False
