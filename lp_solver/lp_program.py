import enum
import re
from copy import copy
from typing import Dict, Optional

from scipy import sparse
from scipy.linalg import lu

from lp_solver.revised_simplex import *

DANTZIG_TH = 100
FACTORIZATION_TH = 20
NUMERIC_STABILITY_TH = 30


class Equation(object):
    class Type(enum.Enum):
        LE = '<='
        GE = '>='
        EQ = '='

    TYPES = tuple(t.value for t in Type)

    def __init__(self, units: Dict[int, float] = None, type_equation: Type = Type.LE,
                 scalar: float = None, negated: bool = False):
        self.units = units or {}
        self.type = type_equation
        self.max_variable_index = -1
        self.scalar = scalar
        self.negated = negated

    @staticmethod
    def unit_from_str(unit: str) -> Tuple[int, float]:
        split_unit = unit.split('x')
        assert len(split_unit) == 2
        c, v = split_unit if split_unit[0] else [1, split_unit[1]]
        return int(v), float(c)

    def parse_lhs(self, lhs: str):
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

        # We are assuming getting the equations in LE format
        assert self.type == Equation.Type.LE

    def negate(self, do_copy=True):
        if do_copy:
            new_equation = copy(self)
        else:
            new_equation = self

        if new_equation.type == Equation.Type.LE:
            # not(x <= 4) <--> x > 4 <--> x >= 4.0001
            new_equation.type = Equation.Type.GE
            new_equation.scalar += EPSILON
        elif new_equation.type == Equation.Type.GE:
            # not(x >= 5) <--> x < 5 <--> x <= 4.999
            new_equation.type = Equation.Type.LE
            new_equation.scalar -= EPSILON
        else:
            raise NotImplementedError()

        return new_equation

    @staticmethod
    def get_equation(equation_str: str, negated: bool= False) -> 'Equation':
        eq = Equation(negated=negated)
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
    def __init__(self, equations: List[Union[Equation, str]] = None, objective: Union[Equation, str] = None,
                 rule: str = 'Dantzig', is_aux=False):
        self.is_aux = is_aux
        self.need_solve_auxiliry = False
        self.B = np.array([[]])  # type: np.ndarray

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

        # Lists to help fast calculation for B^-1, etas are the changes in B, from time to time the list gets to long
        # and then we do LU decomposition and store it in l,u,p lists (as eta matrices)
        self.etas = []  # type: List[EtaMatrix]
        self.l_etas = []  # type: List[EtaMatrix]
        self.u_etas = []  # type: List[EtaMatrix]
        self.p_inv = None
        self.p = None

        rules = {'dantzig': dantzig_rule, 'bland': blands_rule}

        self.rule = rules.get(rule.lower(), rules['dantzig'])
        self.initial_b = np.copy(self.b)

        if self.need_solve_auxiliry:
            lp_aux = LpProgram(equations, objective='-1x0', is_aux=True, rule=rule)

            aux_obj = lp_aux.solve()

            if aux_obj == np.inf:
                assert 0 not in lp_aux.Xb

            if aux_obj != 0 and aux_obj != np.inf:
                raise InfeasibleException

            self.initialize_from_auxiliry(lp_aux)
            self.need_solve_auxiliry = False

    def _create_objective(self, objective: Union[Equation, str]):
        if isinstance(objective, str):
            objective = Equation.get_expression(objective)

        self.Cb = np.zeros(shape=len(self.Xb))
        self.Cn = np.zeros(shape=len(self.Xn))

        for v, c in objective.units.items():
            if v >= len(self.Cn):
                raise UnboundedException
            self.Cn[v] = c

    def initialize_from_auxiliry(self, lp_aux: 'LpProgram'):
        assignment = lp_aux.get_assignment()
        # Pivot to the assignment basis
        entering_vars = set(assignment.keys()) - set(self.Xb)
        leaving_vars = set(self.Xb) - set(assignment.keys())
        self.b = lp_aux.b

        for e, l in zip(entering_vars, leaving_vars):
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
            if equation.max_variable_index >= n:
                n = equation.max_variable_index + 1
                self.Xn = list(range(n))

        self.An = np.zeros(shape=(m, n))
        for i, equation in enumerate(equations):
            cur_equation = np.zeros(shape=(n,))
            for variable, coefficient in equation.units.items():
                assert variable < cur_equation.shape[0], "var: {}, cur_equation.shape[0]: {}".format(variable,
                                                                                                     cur_equation.shape[
                                                                                                         0])
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

    @classmethod
    def permute_matrix(cls, vec: np.ndarray, permutation_order: np.ndarray) -> np.ndarray:
        if permutation_order is None:
            return vec

        permute_vec = np.copy(vec)
        for i, p in enumerate(permutation_order):
            permute_vec[p] = vec[i]

        return permute_vec

    def recalculate_b(self):
        b_tag = np.eye(self.B.shape[0])

        if self.p is not None:
            for i, p in enumerate(self.p):
                b_tag[i, :] = np.eye(self.B.shape[0])[p, :]
            for e in self.l_etas:
                b_tag = np.dot(b_tag, e.get_matrix())
            for e in self.u_etas[::-1]:
                b_tag = np.dot(b_tag, e.get_matrix().T)

        for e in self.etas:
            b_tag = np.dot(b_tag, e.get_matrix())

        return b_tag

    def swap_basis(self, entering_idx, leaving_idx, d_eta):
        self.Xb[leaving_idx], self.Xn[entering_idx] = self.Xn[entering_idx], self.Xb[leaving_idx]
        self.Cb[leaving_idx], self.Cn[entering_idx] = self.Cn[entering_idx], self.Cb[leaving_idx]

        t1, t2 = np.copy(self.An[:, entering_idx]), np.copy(self.B[:, leaving_idx])

        self.An[:, entering_idx] = self.B[:, leaving_idx]
        self.etas.append(d_eta)

        self.B = np.dot(self.B, self.etas[-1].get_matrix())

        np.testing.assert_array_almost_equal(self.B, self.recalculate_b())

    def get_good_entering_leaving_idx(self) -> [int, int, float, EtaMatrix]:
        '''
        Makes sure the eta matrix will not contain very small numbers, to keep numerical stability
        '''
        bad_vars = set()
        if self.is_aux:
            entering_idx = 0
            assert self.Xn[entering_idx] == 0
        else:
            entering_idx = get_entering_variable_idx(self, bad_vars)
        leaving_idx, t, d_eta = get_leaving_variable_idx(self, entering_idx)

        self.is_aux = False
        # Diagonal elements numerical stability
        d_eta_stable = np.bitwise_and(np.abs(d_eta.column) <= EPSILON, d_eta.column != 0)
        while entering_idx >= 0 and any(d_eta_stable):
            entering_idx = get_entering_variable_idx(self, bad_vars)
            leaving_idx, t, d_eta = get_leaving_variable_idx(self, entering_idx)
            bad_vars.add(entering_idx)

        return entering_idx, leaving_idx, t, d_eta

    def solve(self) -> Optional[float]:
        iteration = 0
        try:
            entering_idx = 0  # get_entering_variable_idx(self)
            while entering_idx >= 0:
                entering_idx, leaving_idx, t, d_eta = self.get_good_entering_leaving_idx()

                if entering_idx < 0:
                    break
                if leaving_idx == -1:
                    return np.inf

                self.swap_basis(entering_idx, leaving_idx, d_eta)
                # Update the assignments
                self.b -= t * d_eta.column
                self.b[leaving_idx] = t

                iteration += 1
                if iteration > DANTZIG_TH and self.rule == dantzig_rule:
                    self.rule = blands_rule

                if iteration % 100 == 0:
                    print(float(np.dot(self.Cb, self.b)))

                if self.safeguard(iteration) or len(self.etas) % FACTORIZATION_TH == 0:
                    self.refactorize()
            return float(np.dot(self.Cb, self.b))
        except UnboundedException:
            return np.inf
        except InfeasibleException:
            return None

    def get_assignment(self):
        assignment = dict(zip(self.Xb, self.b))

        return assignment

    def safeguard(self, iteration) -> bool:
        if iteration % NUMERIC_STABILITY_TH != 0:
            return False
        b_approx = FTRAN_using_eta(self, self.initial_b)
        return not np.allclose(self.b, b_approx, atol=EPSILON)

    def refactorize(self, B=None):
        B = B if B is not None else self.B

        p, l, u = lu(B)
        p_inv = np.linalg.inv(p)

        basic_size = len(B)

        Li, Ui = [], []
        eye = np.eye(basic_size)
        for i in range(basic_size):
            # avoid degenerate etas
            id_column = eye[i]

            # L is lower triangular
            if not np.array_equal(id_column, l[:, i]):
                Li += [EtaMatrix(l[:, i], i)]

            # U is upper triangular
            if not np.array_equal(id_column, u.T[:, i]):
                Ui += [EtaMatrix(u.T[:, i], i)]

        self.l_etas = Li
        self.u_etas = Ui
        self.etas = []

        # p maps each row to the pivoted raw
        _, self.p, v = sparse.find(p)
        assert np.array_equal(v, np.ones(len(v)))
        _, self.p_inv, v = sparse.find(p_inv)
        assert np.array_equal(v, np.ones(len(v)))

        return Li, Ui, self.p_inv
