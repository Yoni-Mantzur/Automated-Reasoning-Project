from copy import copy
from typing import List, Dict, Optional, Set

from sat_solver.formula import Formula, Literal


def convert_iff_cnf_basic(lhs: Literal, rhs1: Literal, rhs2: Optional[Literal], operation: Formula.Operator):
    '''
    Convert x iff (y operation z) into a cnf format
    :param lhs: x in the example
    :param rhs1: first literal on the right hand side (if operation is negate, it's the only literal)
    :param rhs2: second literal (if operation is negate None)
    :param operation: right hand side operation
    :return: list of or clauses, each clause is a list of literals with OR between them
    '''
    assert operation in [Formula.Operator.AND, Formula.Operator.OR, Formula.Operator.NEGATION]
    assert (operation == Formula.Operator.NEGATION and rhs2 is None) or (
            operation != Formula.Operator.NEGATION and rhs2 is not None)

    if operation == Formula.Operator.AND:
        # x iff y & z \rightarrow (~x || y) & (~x || z) & (~y || ~z || x)
        return [[copy(lhs).negate(), copy(rhs1)], [copy(lhs).negate(), copy(rhs2)],
                [copy(rhs1).negate(), copy(rhs2).negate(), copy(lhs)]]
    if operation == Formula.Operator.OR:
        # x iff y || z \rightarrow (~x || y || z) & (~y || x) & (~z || x)
        return [[copy(lhs).negate(), copy(rhs1), copy(rhs2)],
                [copy(rhs1).negate(), copy(lhs)],
                [copy(rhs2).negate(), copy(lhs)]]
    if operation == Formula.Operator.NEGATION:
        # x iff ~y  \rightarrow (~x || ~y) & (x || y)
        return [[copy(lhs).negate(), copy(rhs1).negate()],
                [copy(rhs1), copy(lhs)]]


def convert_iff_cnf(lhs: Literal, rhs1: Literal, rhs2: Optional[Literal], operation: Formula.Operator):
    if operation == Formula.Operator.IMPLIES:
        # x <--> (y -> z) iff x <--> (~y || z)
        return convert_iff_cnf_basic(lhs, copy(rhs1).negate(), rhs2, Formula.Operator.OR)
    if operation == Formula.Operator.IFF:
        # x <--> (y <--> z) iff x <--> (y -> z) & x <--> (z -> y)
        return convert_iff_cnf(lhs, rhs1, rhs2, Formula.Operator.IMPLIES) + convert_iff_cnf(lhs, rhs2, rhs1,
                                                                                            Formula.Operator.IMPLIES)
    else:
        return convert_iff_cnf_basic(lhs, rhs1, rhs2, operation)


def tseitins_transformation(formula: Formula):
    # for testing propose
    if not formula:
        return None

    new_variables = {}

    def recursion_tseitins_transformation(subformula: Formula):
        if subformula is None or subformula.is_leaf:
            return []

        if subformula.left.is_leaf:
            l_var = subformula.left.value
        else:
            if subformula.left.idx not in new_variables:
                new_variables[subformula.left.idx] = Literal.from_name("tse{}".format(subformula.left.idx),
                                                                       negated=False)
            l_var = new_variables[subformula.left.idx]

        if subformula.operator is Formula.Operator.NEGATION or subformula.right is None:
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
        self.literal_to_clauses = literal_to_clauses

    @staticmethod
    def from_str(formula: str):
        clauses = tseitins_transformation(Formula.from_str(formula))
        return CnfFormula(clauses)

    def get_variables(self):
        # The formula holds literls, convert to variable and remove duplicates
        return list(set([v.variable for subformula in self.clauses for v in subformula]))

    def __str__(self):
        return str(self.clauses)

    def __repr__(self):
        return str(self)
