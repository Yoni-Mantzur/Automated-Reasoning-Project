import re

from itertools import count
from typing import Optional, List

from sat_solver.patterns import variable_pattern, unary_formula_pattern, binary_formula_pattern

from common.operator import Operator

class Literal(object):
    def __init__(self, variable, negated):
        self.name = variable.name
        self.idx = variable.idx
        self.negated = negated
        self.variable = variable

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

    def __eq__(self, other):
        return self.name == other.name and self.idx == other.idx and self.negated == other.negated

    def __hash__(self):
        return hash(self.name) + hash(self.idx) + hash(self.negated)

    def negate(self):
        self.negated = not self.negated
        return self


class Variable(object):

    _ids = count(-1)

    def __init__(self, name: str):
        self.name = name
        self.idx = next(self._ids)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.name == other.name and self.idx == other.idx

    def __hash__(self):
        return hash(self.name) + hash(self.idx)

class SatFormula(object):
    _ids = count(-1)

    def __init__(self,
                 left: Optional['SatFormula'] = None,
                 right: Optional['SatFormula'] = None,
                 operator: Optional[Operator] = None,
                 is_leaf: bool = False):
        self.operator = operator
        self.left = left
        self.right = right
        self.is_leaf = is_leaf

        self.idx = next(self._ids)

    def get_literals(self) -> List[Literal]:
        if self.is_leaf:
            return [self.value]

        literals = []
        if self.left:
            literals += self.left.get_literals()

        if self.right:
            literals += self.right.get_literals()

        return literals


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
    def create_leaf(variable_name: str) -> 'SatFormula':
        formula = SatFormula(is_leaf=True)
        variable = Literal.from_name(variable_name, negated=False)
        formula.__setattr__('value', variable)
        return formula

    @classmethod
    def from_str(cls, formula: str) -> 'SatFormula':
        m_variable = re.match(variable_pattern, formula)
        m_unary = re.match(unary_formula_pattern, formula)
        m_binary = re.match(binary_formula_pattern, formula)

        if m_variable:
            variable_name = m_variable.group()
            return SatFormula.create_leaf(variable_name)

        # Unary case
        if m_unary:
            m = m_unary
            right = None

        # Binary case
        elif m_binary:
            m = m_binary
            right = SatFormula.from_str(m_binary.group('right'))

        else:
            raise Exception

        left = SatFormula.from_str(m.group('left'))
        op = Operator(m.group('op'))

        return SatFormula(left, right, op)

