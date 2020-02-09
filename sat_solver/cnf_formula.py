from copy import copy
from typing import List, Dict, Optional

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
    # TODO: Support Implies and Bidirectional using recursion
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
        return convert_iff_cnf_basic(lhs, rhs1.negate(), rhs2, Formula.Operator.OR)
    if operation == Formula.Operator.BICONDITIONAL:
        # x <--> (y <--> z) iff x <--> (y -> z) & x <--> (z -> y)
        return convert_iff_cnf(lhs, rhs1, rhs2, Formula.Operator.IMPLIES) + convert_iff_cnf(lhs, rhs2, rhs1,
                                                                                            Formula.Operator.IMPLIES)
    else:
        return convert_iff_cnf_basic(lhs, rhs1, rhs2, operation)


def tseitins_transformation(formula: Formula):
    new_variables = {}
    all_clauses = []

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

        # FOR DEBUG:
        literals = convert_iff_cnf(new_variables[subformula.idx], rhs1=l_var, rhs2=r_var, operation=subformula.operator)

        return convert_iff_cnf(new_variables[subformula.idx], rhs1=l_var, rhs2=r_var, operation=subformula.operator) + \
               recursion_tseitins_transformation(subformula.right) + recursion_tseitins_transformation(subformula.left)

    return recursion_tseitins_transformation(formula) + [[new_variables[formula.idx]]]


class CnfFormula(object):

    def __init__(self, formula: Formula):
        self.formula = tseitins_transformation(formula)  # type: List[List[Literal]]
        self.variable_to_clause = {}  # type: Dict[int, int]
