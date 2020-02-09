import re
from enum import Enum


class Literal(object):
    def __init__(self, variable, negated):
        self.name = variable.name
        self.idx = variable.idx
        self.negated = negated


class Variable(object):
    variable_counter = 0

    def __init__(self, name: str):
        self.name = name
        self.idx = Variable.variable_counter
        Variable.variable_counter += 1


class Formula(object):
    class Operator(Enum):
        IMPLIES = '->'
        BICONDITIONAL = '<->'
        OR = '|'
        AND = '&'
        NEGATION = '~'

    def __init__(self, left: 'Formula', right: 'Formula', operator: Operator):
        self.operator = operator
        self.left = left
        self.right = right

    @staticmethod
    def from_str(formula: str):
        unary_operator_pattern = Formula.Operator.NEGATION.value
        binary_operator_pattern = '|'.join([op.value for op in Formula.Operator if op !=
                                            Formula.Operator.NEGATION])
        variable_pattern = '[a-z]*\\d+'
        unary_formula_pattern = '(?P<{unary}>)(\((?P<left>.*)\)|({variable})*)'.format(
            unary=unary_operator_pattern, variable=variable_pattern)
        sub_formula_pattern = '(\((?P<left>.*)\)(?P<{binary}>.*)'

        m = re.match('%s' % sub_formula_pattern, formula)

        left = m.group(sub_formula_group_name)
        right = m.group(sub_formula_group_name)

        formula_left = Formula.from_str(left)
        formula_right = Formula.from_str(right)

        if not left and not right:
            return None

        # Unary case
        if not right:
            operator = Formula.Operator(formula[0])

        else:
            operator = Formula.Operator(formula[len(left)])

        return Formula(formula_left, formula_right, operator)

    def satisfied(self, assignment: PartialAssignment):


if __name__ == '__main__':
    # print(f'{Formula.Operator.NEGATION}/((.*)/)')
    Formula.from_str("(g&(~asdsa)")
