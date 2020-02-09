import re
from enum import Enum
from itertools import count
from typing import Optional


class Literal(object):
    def __init__(self, variable, negated):
        self.name = variable.name
        self.idx = variable.idx
        self.negated = negated

    @staticmethod
    def from_name(name, negated):
        v = Variable(name)
        return Literal(v, negated)

    def __str__(self):
        s = "~" if self.negated else ""
        s += self.name
        return s

    def __repr__(self):
        return str(self)

    def negate(self):
        self.negated = not self.negated
        return self


class Variable(object):

    _ids = count(-1)

    def __init__(self, name: str):
        self.name = name
        self.idx = next(self._ids)


class Formula(object):
    class Operator(Enum):
        IMPLIES = '->'
        BICONDITIONAL = '<->'
        OR = '|'
        AND = '&'
        NEGATION = '~'

    @classmethod
    def init_counter(cls):
        if not hasattr(cls, '_ids'):
            cls._ids = count(-1)

    def __init__(self,
                 left: Optional['Formula'] = None,
                 right: Optional['Formula'] = None,
                 operator: Optional[Operator] = None,
                 is_leaf: bool = False):
        self.init_counter()
        self.operator = operator
        self.left = left
        self.right = right
        self.is_leaf = is_leaf

        self.idx = next(self._ids)

    def __str__(self):
        # Variable
        if self.is_leaf:
            return self.value.name

        # Binary case
        if self.right:
            return '({}{}{})'.format(self.left, self.operator.value, self.right)

        # Unary case
        return '{}{}'.format(self.operator.value, self.left)

    @staticmethod
    def create_leaf(variable_name: str) -> 'Formula':
        formula = Formula(is_leaf=True)
        variable = Literal.from_name(variable_name, negated=False)
        formula.__setattr__('value', variable)
        return formula

    @classmethod
    def from_str(cls, formula: str) -> 'Formula':
        cls.init_counter()
        unary_operator_pattern = Formula.Operator.NEGATION.value

        operators = map(lambda op: '\|' if op == Formula.Operator.OR.value else op, [op.value for op in Formula.Operator])
        binary_operator_pattern = '|'.join([op for op in operators if op != unary_operator_pattern])
        variable_pattern = '(T|F|[a-z]*\\d+)'
        sub_formula_or_variable_pattern = '(?P<{side}>\(.*\)|{variable}|{unary}.*)'
        sub_formula_for_left_side = sub_formula_or_variable_pattern.format(
            side='left', variable=variable_pattern, unary=unary_operator_pattern)
        sub_formula_for_right_side = sub_formula_or_variable_pattern.format(
            side='right', variable=variable_pattern, unary=unary_operator_pattern)
        unary_formula_pattern = '(?P<op>{unary}){sub_formula}'.format(
            unary=unary_operator_pattern, sub_formula=sub_formula_for_left_side)
        binary_formula_pattern = '\({left}(?P<op>{binary}){right}\)'.format(
            left=sub_formula_for_left_side, binary=binary_operator_pattern, right=sub_formula_for_right_side)

        m_variable = re.match(variable_pattern, formula)
        m_unary = re.match(unary_formula_pattern, formula)
        m_binary = re.match(binary_formula_pattern, formula)

        if m_variable:
            variable_name = m_variable.group()
            return Formula.create_leaf(variable_name)

        # Unary case
        if m_unary:
            m = m_unary
            right = None

        # Binary case
        elif m_binary:
            m = m_binary
            right = Formula.from_str(m_binary.group('right'))

        else:
            raise Exception

        left = Formula.from_str(m.group('left'))
        op = Formula.Operator(m.group('op'))

        return Formula(left, right, op)
