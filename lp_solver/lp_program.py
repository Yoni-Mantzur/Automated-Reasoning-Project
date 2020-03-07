import enum
import re
from typing import Dict

import numpy as np

from lp_solver.UnboundedException import InfeasibleException
from lp_solver.revised_simplex import *


# from lp_solver import UnboundedException

class Equation(object):
    class Type(enum.Enum):
        LE = '<='
        GE = '>='
        EQ = '='

    TYPES = tuple(t.value for t in Type)

    def __init__(self, units: Dict[int, float] = None, type_equation: Type = Type.LE,
                 scalar: float = None):
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

            if c != 0:
                self.units[v] = c

        self.max_variable_index = max(self.units.keys())

    def from_str(self, equation_str: str) -> None:
        r = '(.*?)({})(-?[0-9.]+$)'.format('|'.join(Equation.TYPES))
        lhs, t, rhs = re.search(r, equation_str).groups()

        self.parse_lhs(lhs)
        self.type = Equation.Type(t)
        self.scalar = float(rhs)

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
            # if c != 0:
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
    def __init__(self, equations: List[Union[Equation, str]] = None, objective: Union[Equation, str] = None,
                 rule: str = 'Dantzig', is_aux=False):
        # Basis, coefficients and variables
        # self.B = [] # type: # List[EtaMatrix]
        self.is_aux = is_aux
        self.need_solve_auxiliry = False
        self.B = np.array([[]])  # type: np.ndarray
        self.etas = []  # type: List[EtaMatrix]
        self.Xb = np.array([])  # type: np.ndarray

        # Non-basic coefficients and variables
        self.An = np.array([[]])  # type: np.ndarray
        self.Xn = np.array([])  # type: np.ndarray

        # constraints scalars
        self.b = np.array([])  # type: np.ndarray

        if equations:
            self._add_equations(equations)


        # Objective coefficients for basic and non basic variables
        self.Cb = None
        self.Cn = None
        if objective:
            self._create_objective(objective)

        from functools import partial
        rules = {'dantzig': dantzig_rule, 'bland': blands_rule}

        self.rule = rules.get(rule.lower(), rules['dantzig'])
        self.initial_b = np.copy(self.b)

        if self.need_solve_auxiliry:
            lp_aux = LpProgram(equations, objective='-1x0', is_aux=True)
            aux_obj = lp_aux.solve()
            assert aux_obj != np.inf
            if aux_obj != 0:
                raise InfeasibleException

            self.initialize_from_auxiliry(lp_aux)
            self.need_solve_auxiliry = False

    def _create_objective(self, objective: Union[Equation, str]):
        if isinstance(objective, str):
            objective = Equation.get_expression(objective)

        self.Cb = np.zeros(shape=len(self.Xb))
        self.Cn = np.zeros(shape=len(self.Xn))

        for v, c in objective.units.items():
            assert v <= len(self.Cn), "There is an unbounded variable"
            self.Cn[v] = c

    def initialize_from_auxiliry(self, lp_aux: 'LpProgram'):
        assignment = lp_aux.get_assignment()
        # Pivot to the assignment basis
        entering_vars = set(assignment.keys()) - set(self.Xb)
        leaving_vars = set(self.Xb) - set(assignment.keys())
        # self.B = lp_aux.B
        # self.etas = lp_aux.etas
        self.b = lp_aux.b

        for e,l in zip(entering_vars, leaving_vars):
            e_idx, l_idx = int(np.where(self.Xn == e)[0]), int(np.where(self.Xb == l)[0])
            d_eta = EtaMatrix(self.An[:, e_idx], l_idx)

            self.swap_basis(e_idx, l_idx, d_eta)

            self.b[l_idx] = assignment[self.Xb[l_idx]]
        return

    def _add_equations(self, equations: List[Union[Equation, str]]) -> None:
        n = -1  # Number of variables
        m = len(equations)  # Number of equations

        for i, equation in enumerate(equations):
            if isinstance(equation, str):
                equation = Equation.get_equation(equation)
            assert isinstance(equation, Equation)
            equations[i] = equation
            if self.is_aux:
                equations[i].units.update({0: -1})
            if equation.max_variable_index > n:
                n = equation.max_variable_index + 1
                self.Xn = list(range(n))

        self.An = np.zeros(shape=(m, n))
        for i, equation in enumerate(equations):
            cur_equation = np.zeros(shape=(n,))
            for variable, coefficient in equation.units.items():
                assert variable < cur_equation.shape[0]
                cur_equation[variable] = coefficient

            # Add row
            self.An[i, :] = cur_equation
            if equation.scalar < 0 and not self.is_aux:
                self.need_solve_auxiliry = True

            self.b = np.append(self.b, equation.scalar)

        # Create the basic variables
        self.Xb = np.arange(n, n + m)
        self.B = np.eye(len(self.Xb))

    def __str__(self):
        return 'B (shape: %s): \n%s\nXb (shape=%s): \n %s\n' \
               'An (shape=%s): \n %s\nXn (shape=%s): \n %s\n' \
               'Cb (shape=%s): \n %s\nCn (shape=%s): \n %s\n' \
               'b (shape=%s): \n %s\n' \
               % (
                   self.B.shape, self.B,
                   self.Xb.shape, self.Xb, self.An.shape, self.An, self.Xn.shape,
                   self.Xn,
                   self.Cb.shape, self.Cb, self.Cn.shape, self.Cn, self.b.shape, self.b)

    def dump(self):
        print("Lp program:")
        print('*' * 100)

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
        print('*' * 100)

    def swap_basis(self, entering_idx, leaving_idx, d_eta):
        self.Xb[leaving_idx], self.Xn[entering_idx] = self.Xn[entering_idx], self.Xb[leaving_idx]
        self.Cb[leaving_idx], self.Cn[entering_idx] = self.Cn[entering_idx], self.Cb[leaving_idx]

        t1, t2 = np.copy(self.An[:, entering_idx]), np.copy(self.B[:, leaving_idx])

        self.An[:, entering_idx] = self.B[:, leaving_idx]
        self.etas.append(d_eta)
        self.B = np.dot(self.B, self.etas[-1].get_matrix())

        # np.testing.assert_almost_equal(t1, self.B[:, leaving_idx])

    def get_good_entering_leaving_idx(self) -> [int, int, float, EtaMatrix]:
        '''
        Makes sure the eta matrix will not contain very small numbers, to keep numerical stability
        '''
        bad_vars = set()
        if self.is_aux:
            entering_idx = 0 #int(np.where(self.Xn == 0)[0])
            assert self.Xn[entering_idx] == 0
        else:
            entering_idx = get_entering_variable_idx(self, bad_vars)
        leaving_idx, t, d_eta = get_leaving_variable_idx(self, entering_idx)

        self.is_aux = False
        d_eta_stable = np.bitwise_and(np.abs(d_eta.column) <= EPSILON, d_eta.column != 0)
        while entering_idx >= 0 and any(d_eta_stable):
            entering_idx = get_entering_variable_idx(self, bad_vars)
            leaving_idx, t, d_eta = get_leaving_variable_idx(self, entering_idx)
            bad_vars.add(entering_idx)

        return entering_idx, leaving_idx, t, d_eta

    def solve(self) -> float:
        # TODO: Implement refactorization
        # TODO: Connect LP to the SMT solver  ???
        try:
            entering_idx = 0  # get_entering_variable_idx(self)
            while entering_idx >= 0:
                # leaving_idx, t, d = get_leaving_variable_idx(self, entering_idx)
                entering_idx, leaving_idx, t, d_eta = self.get_good_entering_leaving_idx()

                if entering_idx < 0:
                    break
                if leaving_idx == -1:
                    return np.inf

                self.swap_basis(entering_idx, leaving_idx, d_eta)
                # Update the assignments
                self.b -= t * d_eta.column
                self.b[leaving_idx] = t

                # entering_idx = get_entering_variable_idx(self)

            # TODO: Do we need to extract the actual assignment (for the real variables)?
            # y = backward_transformation(self.B, self.Cb)
            # coefs = self.Cn - np.dot(y, self.An)
            # np.dot(self.b, coefs) +

            if self.safeguard():
                self.refactorize()
            return float(np.dot(self.Cb, self.b))
        except UnboundedException:
            return np.inf
        except InfeasibleException:
            return None

    def get_assignment(self):
        assignment = dict(zip(self.Xb, self.b))
        # assignment_zero = dict(zip(self.Xn, [0] * len(self.Xn)))
        #
        # assignment.update(assignment_zero)
        return assignment

    def safeguard(self):
        return not np.allclose(np.dot(self.B, self.b), self.initial_b, atol=EPSILON)

    def refactorize(self):
        from scipy.linalg import lu
        (p, l, u) = lu(self.B)
