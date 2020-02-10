from copy import copy, deepcopy
from collections import defaultdict, OrderedDict

from sat_solver.cnf_formula import CnfFormula
from sat_solver.formula import Literal
from sat_solver.preprocessor import preprocess


class DPLL(object):
    def __init__(self, cnf_forumla: CnfFormula, partial_assignment = {}):

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

        for idx in sorted(removed_clauses, reverse=True):
            formula.clauses[idx] = []

        for idx in formula.literal_to_clauses[negate_lit]:
            if len(formula.clauses[idx]) == 0:
                continue
            if len(formula.clauses[idx]) == 1:
                # In this situation we want to delete a literal from the clause but it is the last one, UNSAT
                return None
            formula.clauses[idx].remove(negate_lit)

        return formula

    def unit_propagation(self, formula: CnfFormula):
        '''
        Perform unit propoagation on the given cnf formula (editing the formula)
        :return the formula after unit propogation
        '''
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

    def search(self):
        if self.formula is None:
            # Formula is unsat
            return False
        real_formula = [f for f in self.formula.clauses if f != []]
        if real_formula == []:
            # Found valid assinment
            self.unsat = False
            return True

        current_variables = self.formula.get_variables()
        next_decision = current_variables[0]

        for d in [True, False]:
            formula = deepcopy(self.formula)
            formula = self._assign_true_to_literal(formula, Literal(next_decision, d))
            formula = self.unit_propagation(formula)

            sat = DPLL(formula, self._assignment).search()
            if sat:
                self.unsat = False
                return sat

        return False




# if __name__ == "__main__":
#     f = CnfFormula.from_str("(x1|~x2)&x3")
#     print(f)
#     d = DPLL(f)
#     d.unit_propagation()
#     print(d._formulas[-1])