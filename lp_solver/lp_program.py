import enum
import re
from typing import Dict, Tuple, Union, List

import numpy as np


class Equation(object):
    class Type(enum.Enum):
        LE = '<='
        GE = '>='
        EQ = '='

    TYPES = tuple(t.value for t in Type)

    def __init__(self, units: Dict[int, float] = None, type: Type = None, scalar = None):
        self.units = units or {}
        self.type = type
        self.max_variable_index = -1
        self.scalar = scalar

    @staticmethod
    def unit_from_str(unit: str) -> Tuple[float, int]:
        split_unit = unit.split('x')
        assert len(split_unit) == 2
        c, v = split_unit if split_unit[0] else [1, split_unit[1]]
        return int(v), float(c)

    @staticmethod
    def from_str(equation_str: str) -> 'Equation':
        equation = Equation()
        max_index = -1
        # Most of the equations will be 2x1,3x2>=2, but the objective will be just 2x1,3x2
        if any([t.value in equation_str for t in Equation.Type]):
            r = '(.*?)(>=|<=|=)(-?[0-9]*$)'.format('|'.join([t.value for t in Equation.Type]))
            groups = re.search(r,equation_str)
            lhs = groups[1]
            t = groups[2]
            rhs = groups[3]
            equation.type = Equation.Type(t)
            equation.scalar = int(rhs)
        else:
            lhs = equation_str
        units = lhs.split(',')
        for unit in units:
            unit = unit.strip()
            if True:
                v, c = Equation.unit_from_str(unit)
                equation.units[v] = c
                if v > max_index:
                    max_index = v

        equation.max_variable_index = max_index
        # TODO: Turn into normal form, i.e. transform the types into LE
        return equation

    def __str__(self):
        return '%s %s 0' % (
            ' '.join(['%fx%d' % (c, v) for v, c in self.units.items()]),
            self.type.value)

    def __repr__(self):
        return 'Equation(%s)'

    def __hash__(self):
        return hash(self.units) + hash(self.type)

    def __eq__(self, other):
        return str(self) == str(other)


class LpProgram(object):
    def __init__(self, equations: List[Union[Equation, str]], objective: Union[Equation, str]):
        # Basis, coefficients and variables
        self.B = np.array([[]])  # type: np.array
        self.Xb = []  # type: List[int]

        # Non-basic coefficients and variables
        self.An = np.array([[]])  # type: np.array
        self.Xn = []  # type: List[int]

        # constraints scalars
        self.b = []  # type: List[float]

        self._add_equations(equations)

        # Objective coefficients for basic and non basic variables
        self.Cb = None
        self.Cn = None
        self._create_objective(objective)

    def _create_objective(self, equation: Union[Equation, str]):
        if isinstance(equation, str):
            equation = Equation.from_str(equation)
        self.Cb = np.zeros(shape=len(self.Xb))
        self.Cn = np.zeros(shape=len(self.Xn))

        for v, c in equation.units.items():
            assert v <= len(self.Cn), "There is an unbounded variable"
            self.Cn[v] = c

    def _add_equations(self, equations: List[Union[Equation, str]]) -> None:
        n = -1  # Number of variables
        m = len(equations)  # Number of equations
        for i, equation in enumerate(equations):
            if isinstance(equation, str):
                equation = Equation.from_str(equation)
            assert isinstance(equation, Equation)
            equations[i] = equation

            if equation.max_variable_index > n:
                n = equation.max_variable_index + 1

        self.An = np.zeros(shape=(m, n))
        for i, equation in enumerate(equations):
            cur_equation = np.zeros(shape=(n))
            for variable, coefficient in equation.units.items():
                # TODO: Can Xn be a set?
                if variable not in self.Xn:
                    self.Xn.append(variable)
                assert variable < cur_equation.shape[0]
                cur_equation[variable] = coefficient

            # Add row
            self.An[i, :] = cur_equation
            self.b.append(equation.scalar)
        # Create the basic variables
        self.Xb = list(range(n, n + m))
        self.B = np.eye(len(self.Xb))
