from copy import copy
from collections import defaultdict, OrderedDict

from sat_solver.cnf_formula import CnfFormula
from sat_solver.formula import Literal
from sat_solver.preprocessor import preprocess


class DPLL(object):
    def __init__(self, cnf_forumla: CnfFormula):
        self._formulas = []
        self._formulas.append(preprocess(cnf_forumla))
        self.unit_propagation(self._formulas[-1])
        self.unsat = None

    def update_variables(self):
        self.unassigned_variables = set([v.idx for subformula in self._formulas[-1].clauses for v in subformula])

    def _assign_true_to_literal(self, formula: CnfFormula, literal: Literal):
        '''
        Assigns true to the given literal, i.e. remove all clauses that this literal is in
        And remove the negation of this literal from all clauses, if it is the only literal return None
        :param literal:
        :return: None if can't do this assignment, otherwise a new formula (also editing the self._formulas[-1])
        '''
        negate_lit = copy(literal).negate()
        # We decided that lit is True (lit might be ~x_1, then x_1 is False)
        # Remove all clauses that has lit (they are satisfied), and delete ~lit from all other clauses
        removed_clauses = set(formula.literal_to_clauses[literal])

        for idx in sorted(removed_clauses, reverse=True):
            formula.clauses[idx] = []
            # real_formula = [cl for cl in formula.clauses if cl != []]
            # if len(real_formula) == 0:
            #     # Found SAT
            #     return []

        for idx in formula.literal_to_clauses[negate_lit]:
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

        self.update_variables()
        return formula

    #
    # def search(self, decisions=OrderedDict()):
    #     all_variables = self.unassigned_variables
    #     next_desicion = all_variables[0]
    #     if next_desicion in decisions:
    #         decisions[next_desicion] = False
    #     else:
    #         decisions[next_desicion] = True
    #
    #     self._assign_true_to_literal(Literal(next_desicion, decisions[next_desicion]))
    #     new_formula = self.unit_propagation(self._formulas[-1])
    #     if new_formula is None:
    #         # This branch is unsat, we can go back
    #         pass
    #     if self._formulas[-1] == []:
    #         self.unsat = False
    #         return 'SAT'
    #
    #
    # def search(self):
    #     all_variables = self.unassigned_variables
    #     not_found = True
    #     decisions = defaultdict(True) # type: Dict[Variable, bool]
    #     while not_found:
    #         # First try all variables as True
    #         for variable in all_variables:
    #             if variable in self.unassigned_variables:
    #                 self._formulas.append(self._formulas[-1])
    #                 # No need for else and true because we use defaultdict
    #                 if variable in decisions:
    #                     decisions[variable] = False
    #                 self._assign_true_to_literal(Literal(variable, decisions[variable]))
    #                 # self.unit_propagation()
    #                 if self._formulas[-1] == []:
    #                     self.unsat = False
    #                     return 'SAT'
    #
    #
    #     # Next try to change to False one by one
    #     for variable in all_variables:
    #         if variable in self.unassigned_variables:
    #             self._formulas.pop
    #             self._assign_true_to_literal(Literal(variable, True))
    #             self.unit_propagation()
    #             if self._formulas[-1] == []:
    #                 self.unsat = False
    #                 return 'SAT'
    #             self.update_variables()

# if __name__ == "__main__":
#     f = CnfFormula.from_str("(x1|~x2)&x3")
#     print(f)
#     d = DPLL(f)
#     d.unit_propagation()
#     print(d._formulas[-1])