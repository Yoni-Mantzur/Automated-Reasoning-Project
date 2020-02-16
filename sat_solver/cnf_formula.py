from copy import copy
from typing import List, Dict, Optional, Set

from common.operator import Operator
from sat_solver.sat_formula import SatFormula, Literal


def convert_iff_cnf_basic(lhs: Literal, rhs1: Literal, rhs2: Optional[Literal], operation: Operator):
    '''
    Convert x iff (y operation z) into a cnf format
    :param lhs: x in the example
    :param rhs1: first literal on the right hand side (if operation is negate, it's the only literal)
    :param rhs2: second literal (if operation is negate None)
    :param operation: right hand side operation
    :return: list of or clauses, each clause is a list of literals with OR between them
    '''
    assert operation in [Operator.AND, Operator.OR, Operator.NEGATION]
    assert (operation == Operator.NEGATION and rhs2 is None) or (
            operation != Operator.NEGATION and rhs2 is not None)

    if operation == Operator.AND:
        # x iff y & z \rightarrow (~x || y) & (~x || z) & (~y || ~z || x)
        return [[copy(lhs).negate(), copy(rhs1)], [copy(lhs).negate(), copy(rhs2)],
                [copy(rhs1).negate(), copy(rhs2).negate(), copy(lhs)]]
    if operation == Operator.OR:
        # x iff y || z \rightarrow (~x || y || z) & (~y || x) & (~z || x)
        return [[copy(lhs).negate(), copy(rhs1), copy(rhs2)],
                [copy(rhs1).negate(), copy(lhs)],
                [copy(rhs2).negate(), copy(lhs)]]
    if operation == Operator.NEGATION:
        # x iff ~y  \rightarrow (~x || ~y) & (x || y)
        return [[copy(lhs).negate(), copy(rhs1).negate()],
                [copy(rhs1), copy(lhs)]]


def convert_iff_cnf(lhs: Literal, rhs1: Literal, rhs2: Optional[Literal], operation: Operator):
    if operation == Operator.IMPLIES:
        # x <--> (y -> z) iff x <--> (~y || z)
        return convert_iff_cnf_basic(lhs, copy(rhs1).negate(), rhs2, Operator.OR)
    if operation == Operator.IFF:
        # x <--> (y <--> z) iff x <--> (y -> z) & x <--> (z -> y)
        return convert_iff_cnf(lhs, rhs1, rhs2, Operator.IMPLIES) + convert_iff_cnf(lhs, rhs2, rhs1, Operator.IMPLIES)
    else:
        return convert_iff_cnf_basic(lhs, rhs1, rhs2, operation)


def tseitins_transformation(formula: SatFormula):
    # for testing propose
    if not formula:
        return None

    new_variables = {}

    def recursion_tseitins_transformation(subformula: SatFormula):
        if subformula is None or subformula.is_leaf:
            return []

        if subformula.left.is_leaf:
            l_var = subformula.left.value
        else:
            if subformula.left.idx not in new_variables:
                new_variables[subformula.left.idx] = Literal.from_name("tse{}".format(subformula.left.idx),
                                                                       negated=False)
            l_var = new_variables[subformula.left.idx]

        if subformula.operator is Operator.NEGATION or subformula.right is None:
            r_var = None
        else:
            if subformula.right.is_leaf:
                r_var = subformula.right.value
            else:
                if subformula.right.idx not in new_variables:
                    new_variables[subformula.right.idx] = Literal.from_name("tse{}".format(subformula.right.idx),
                                                                            negated=False)
                r_var = new_variables[subformula.right.idx]

        if subformula.idx not in new_variables:
            new_variables[subformula.idx] = Literal.from_name("tse{}".format(subformula.idx), negated=False)

        return convert_iff_cnf(new_variables[subformula.idx], rhs1=l_var, rhs2=r_var, operation=subformula.operator) + \
               recursion_tseitins_transformation(subformula.right) + recursion_tseitins_transformation(subformula.left)

    return recursion_tseitins_transformation(formula) + [[new_variables[formula.idx]]]


class CnfFormula(object):

    def __init__(self, clauses: List[List[Literal]] = None, literal_to_clauses: Dict[Literal, Set[int]] = None):
        self.clauses = clauses
        self.empty_clauses = 0
        self.literal_to_clauses = literal_to_clauses

    @staticmethod
    def from_str(formula: str):
        clauses = tseitins_transformation(SatFormula.from_str(formula))
        return CnfFormula(clauses)

    def get_variables(self):
        # The formula holds literls, convert to variable and remove duplicates
        return list(set([v.variable for subformula in self.clauses for v in subformula]))

    def get_literals(self):
        return set([v.name for subformula in self.clauses for v in subformula])

    def __str__(self):
        return str(self.clauses)

    def __repr__(self):
        return str(self)

    def add_clause(self, literals):
        # TODO: when using this dont forget to update the watch_literals
        self.clauses += literals
        for lit in literals:
            self.literal_to_clauses[lit].update(len(self.clauses))

    def remove_clause(self, indices):
        if isinstance(indices, int):
            indices = [indices]

        for idx in sorted(indices, reverse=True):
            for lit in self.clauses[idx]:
                # Update the literal_tp_clauses we use it in the decision heuristic
                self.literal_to_clauses[lit].remove(idx)
            self.clauses[idx] = []
            self.empty_clauses += 1
        if self.empty_clauses == len(self.clauses):
            return False

        return True

    def remove_literal(self, clause_idx, literal):
        # if len(self.clauses[clause_idx]) == 1:
        #     # In this situation we want to delete a literal from the clause but it is the last one, UNSAT
        #     return False
        # self.clauses[clause_idx].remove(literal)
        self.literal_to_clauses[literal] = {}
        return True

    def get_literal_appears_max(self, assignments):
        literals_length = {k: len(v) for k,v in self.literal_to_clauses.items()}
        literals_length_sorted = sorted(literals_length, key=literals_length.get, reverse=1)
        for l in literals_length_sorted:
            if l.name not in assignments:
                return l
        # # TODO: Might be helpful to keep the literal that apperas the most (for the decision heuristic)
        # return keywithmaxval(self.literal_to_clauses)
