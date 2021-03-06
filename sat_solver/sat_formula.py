import re
from itertools import count
from typing import List

from common.operator import Operator
from sat_solver.patterns import variable_pattern, unary_formula_pattern, binary_formula_pattern, binary_operator_pattern


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
    def create_leaf(variable_name: str, variable=None, negated=False) -> 'SatFormula':
        formula = SatFormula(is_leaf=True)
        if variable:
            variable = Literal(variable, negated=negated)
        else:
            variable = Literal.from_name(variable_name, negated=negated)
        formula.__setattr__('value', variable)
        return formula

    @staticmethod
    def from_str(s):
        def get_op(s):
            op = re.search(binary_operator_pattern, s).group(0)
            return op, len(op)

        def get_variable(s, idx):
            if 'a' <= s[idx] <= 'z':
                idx += 1

            while idx < len(s) and s[idx].isdigit():
                idx += 1

            return idx

        def _from_str_helper(idx):
            if s[idx] == Operator.NEGATION.value:
                first, new_idx = _from_str_helper(idx + 1)
                return SatFormula(first, operator=Operator.NEGATION), new_idx

            if s[idx] == '(':
                first, new_idx = _from_str_helper(idx + 1)
                root, gap_idx = get_op(s[new_idx:])

                second, new_idx = _from_str_helper(new_idx + gap_idx)
                return SatFormula(first, second, Operator(root)), new_idx + 1  # The new_idx + 1 is for the ')'

            if s[idx] in constants:
                return constants[s[idx]], idx + 1

            var_idx = get_variable(s, idx)
            v = s[idx:var_idx]
            if v not in variables:
                variables[v] = SatFormula.create_leaf(v)

            return variables[v], var_idx

        constants = {'T': SatFormula.create_leaf('T'),
                     'F': SatFormula.create_leaf('F'),
                     '~F': 'T',
                     '~T': 'F'}

        variables = {}

        return _from_str_helper(0)[0]