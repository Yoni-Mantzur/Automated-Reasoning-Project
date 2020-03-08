import re
from itertools import count
from typing import List

from common.operator import Operator
from sat_solver.patterns import variable_pattern, unary_formula_pattern, binary_formula_pattern


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

    def __lt__(self, other):
        return self.idx < other.idx if self.idx != other.idx else self.negated < other.negated

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
                 left: 'SatFormula' = None,
                 right: 'SatFormula' = None,
                 operator: Operator = None,
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
            return str(self.value)

        # Binary case
        if self.right:
            return '({}{}{})'.format(self.left, self.operator.value, self.right)

        # Unary case
        return '{}{}'.format(self.operator.value, self.left)

    @staticmethod
    def create_leaf(variable_name: str, variable=None) -> 'SatFormula':
        formula = SatFormula(is_leaf=True)
        if variable:
            variable = Literal(variable, negated=False)
        else:
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

    def from_infix(self, s):
        def get_op(s):
            pass


        def _from_infix_helper(idx):
            if s[idx] == Operator.NEGATION.value:
                first, new_idx = _from_infix_helper(idx + 1)
                return SatFormula(first, operator=Operator.NEGATION), new_idx

            if s[idx] == '(':
                first, new_idx = _from_infix_helper(idx + 1)
                root, gap_idx = get_op(s[new_idx:])

                second, new_idx = _from_infix_helper(new_idx + gap_idx)
                return SatFormula(first, second, Operator(root)), new_idx + 1  # The new_idx + 1 is for the ')'

            return SatFormula.create_leaf(s[idx]), idx + 1

            var_idx = Formula.get_variable_index(s, idx)
            if var_idx > 0:
                return Formula(s[idx:var_idx]), var_idx

            return None

        return _from_infix_helper(0)[0]