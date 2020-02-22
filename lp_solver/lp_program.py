import enum
from typing import Dict, Tuple, Union


class Equation(object):
    class Type(enum.Enum):
        LE = '<='
        GE = '>='
        EQ = '='

    TYPES = tuple(t.value for t in Type)

    def __init__(self, units: Dict[float, int] = None, type: Type = None):
        self.units = units or {}
        self.type = type

    @staticmethod
    def unit_from_str(unit: str) -> Tuple[float, int]:
        split_unit = unit.split('x')
        assert len(split_unit) == 2
        c, v = split_unit if split_unit[0] else [1, split_unit[1]]
        return float(c), int(v)

    @staticmethod
    def from_str(equation_str: str) -> 'Equation':
        equation = Equation()

        units = equation_str.split(',')
        for unit in units:
            unit = unit.strip()
            if unit in Equation.TYPES:
                equation.type = Equation.Type(unit)

            else:
                c, v = Equation.unit_from_str(unit)
                equation.units[c] = v

        return equation

    def __str__(self):
        return '%s %s 0' % (
            ' '.join(['%fx%d' % (c, v) for c, v in self.units.items()]),
            self.type.value)

    def __repr__(self):
        return 'Equation(%s)'

    def __hash__(self):
        return hash(self.units) + hash(self.type)

    def __eq__(self, other):
        return str(self) == str(other)


class LpProgram(object):
    def __init__(self):
        self.An = []
        self.B = []
        self.Xb = []
        self.Xn = []
        self.b = []

    def add_equation(self, equation: Union[Equation, str]) -> None:
        pass
