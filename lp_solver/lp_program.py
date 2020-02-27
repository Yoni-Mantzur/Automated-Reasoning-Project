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

    def __init__(self, units: Dict[int, float] = None, type_equation: Type = Type.LE, scalar: float = None):
        self.units = units or {}
        self.type = type_equation
        self.max_variable_index = -1
        self.scalar = scalar

    @staticmethod
    def unit_from_str(unit: str) -> Tuple[int, float]:
        split_unit = unit.split('x')
        assert len(split_unit) == 2
        c, v = split_unit if split_unit[0] else [1, split_unit[1]]
        return int(v), float(c)

    def parse_lhs(self, lhs: str):
        # Parse left side
        units = lhs.split(',')
        for unit in units:
            v, c = Equation.unit_from_str(unit.strip())

            # TODO: should we leave 0 coef?
            if c != 0:
                self.units[v] = c

        self.max_variable_index = max(self.units.keys())

    def from_str(self, equation_str: str) -> None:
        r = '(.*?)({})(-?[0-9]+$)'.format('|'.join(Equation.TYPES))
        lhs, t, rhs = re.search(r, equation_str).groups()

        self.parse_lhs(lhs)
        self.type = Equation.Type(t)
        self.scalar = int(rhs)

        # TODO: Turn into normal form, i.e. transform the types into LE

    @staticmethod
    def get_equation(equation_str: str) -> 'Equation':
        eq = Equation()
        eq.from_str(equation_str)
        return eq

    @staticmethod
    def get_expression(expression: str) -> 'Equation':
        eq = Equation()
        eq.parse_lhs(expression)
        return eq

    def __str__(self):
        units_str = ''
        for v, c in self.units.items():

            # TODO: should we need c==0?
            if c != 0:
                sign = ('+' if c > 0 else '')
                units_str += '%s%.2fx%d' % (sign, c, v)

        if self.scalar:
            return '%s %s %.2f' % (units_str, self.type.value, self.scalar)

        return units_str

    def __repr__(self):
        return 'Equation(%s)' % str(self)

    def __hash__(self):
        return hash(self.units) + hash(self.type)

    def __eq__(self, other):
        return self.units == other.units and self.type == other.type and self.scalar == other.scalar


class LpProgram(object):
    def __init__(self, equations: List[Union[Equation, str]], objective: Union[Equation, str]):
        # Basis, coefficients and variables
        self.B = np.array([[]])  # type: np.ndarray
        self.Xb = np.array([])  # type: np.ndarray

        # Non-basic coefficients and variables
        self.An = np.array([[]])  # type: np.ndarray
        self.Xn = np.array([])  # type: np.ndarray

        # constraints scalars
        self.b = np.array([])  # type: np.ndarray

        self._add_equations(equations)

        # Objective coefficients for basic and non basic variables
        self.Cb = None
        self.Cn = None
        self._create_objective(objective)

    def _create_objective(self, objective: Union[Equation, str]):
        if isinstance(objective, str):
            objective = Equation.get_expression(objective)

        self.Cb = np.zeros(shape=len(self.Xb))
        self.Cn = np.zeros(shape=len(self.Xn))

        for v, c in objective.units.items():
            assert v <= len(self.Cn), "There is an unbounded variable"
            self.Cn[v] = c

    def _add_equations(self, equations: List[Union[Equation, str]]) -> None:
        n = -1  # Number of variables
        m = len(equations)  # Number of equations

        for i, equation in enumerate(equations):
            if isinstance(equation, str):
                equation = Equation.get_equation(equation)
            assert isinstance(equation, Equation)
            equations[i] = equation

            if equation.max_variable_index > n:
                n = equation.max_variable_index + 1

        self.An = np.zeros(shape=(m, n))
        for i, equation in enumerate(equations):
            cur_equation = np.zeros(shape=(n,))
            for variable, coefficient in equation.units.items():
                if variable not in self.Xn:
                    self.Xn = np.append(self.Xn, variable)
                assert variable < cur_equation.shape[0]
                cur_equation[variable] = coefficient

            # Add row
            self.An[i, :] = cur_equation
            self.b = np.append(self.b, equation.scalar)

        # Create the basic variables
        self.Xb = list(range(n, n + m))
        self.B = np.eye(len(self.Xb))

    def __str__(self):
        return 'B: \n %s\n Xb: \n %s\n' \
               'An: \n %s\n Xn: \n %s\n' \
               'Cb: \n %s\n Cn: \n %s\n' \
               'b: \n %s\n' \
               % (str(self.B), str(self.Xb), str(self.An), str(self.Xn), str(self.Cb), str(self.Cn), str(self.b))

    def dump(self):
        print("Lp program:")
        print('*'*100)

        print("Equations:")
        variables = np.append(self.Xn, self.Xn)
        for i, coefs in enumerate(np.append(self.An, self.B, axis=1)):
            units = {c: v for c, v in zip(variables, coefs)}
            scalar = self.b[i]
            print(str(Equation(units=units, scalar=scalar)))

        print("Objective:")
        variables = np.append(self.Xn, self.Xb)
        coefs = np.append(self.Cn, self.Cb)
        units = {c: v for c, v in zip(variables, coefs)}

        print(str(Equation(units)))
        print('*'*100)


