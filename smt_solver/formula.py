import re
from abc import abstractmethod
from itertools import count
from typing import List, Dict, Optional

from common.operator import Operator
from sat_solver.sat_formula import SatFormula
from smt_solver.patterns import *


class Term(object):
    _ids = count(0)

    def __init__(self, name):
        self.idx = next(self._ids)
        self.name = name

    def get_terms(self):
        return []

    def get_functions(self):
        return []

    @staticmethod
    def from_str(term: str) -> 'Term':
        m = re.match(term_pattern, term)
        assert m

        if m.group('function'):
            return FunctionTerm.from_str(term)

        if m.group('variable'):
            return PureTerm.from_str(term)

        raise Exception('Not a term string')


class PureTerm(Term):

    def __init__(self, name):
        super(PureTerm, self).__init__(name)
        self.name = name

    def get_terms(self):
        return [self]

    @staticmethod
    def from_str(term: str) -> 'Term':
        return PureTerm(term)

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'PureTerm(%s)' % self.name


class FunctionTerm(Term):

    def __init__(self, name, input_terms: List[Term]):
        super(FunctionTerm, self).__init__(name)
        self.input_terms = input_terms

    def get_terms(self):
        terms = []
        for input_term in self.input_terms:
            terms.append(input_term.get_terms())
        return terms

    def get_functions(self):
        functions = [self]
        for input_term in self.input_terms:
            functions.append(input_term.get_functions())
        return functions

    @staticmethod
    def from_str(term: str) -> Term:
        m = re.match(function_pattern, term)
        assert m;
        function_name = m.group('function_name')
        inputs = m.group('inputs').strip().split(',')
        # TODO: handle nested functions
        inputs_terms = list(map(lambda term: Term.from_str(term), inputs))

        return FunctionTerm(function_name, inputs_terms)

    def __str__(self):
        return self.name + '({})'.format(','.join([str(term) for term in self.input_terms]))

    def __repr__(self):
        return 'FunctionTerm(%s)' % str(self)


class EquationTerm(Term):

    def __init__(self, lhs: Term, rhs: Term):
        super(EquationTerm, self).__init__('')
        self.name = 'v%d' % self.idx
        self.lhs = lhs
        self.rhs = rhs

    def get_terms(self):
        return self.lhs.get_terms() + self.rhs.get_terms()

    def get_functions(self):
        return self.lhs.get_functions() + self.rhs.get_functions()

    @staticmethod
    def from_str(term: str) -> Term:
        m = re.match(equation_pattern, term)
        assert m;
        lhs = m.group('lhs')
        rhs = m.group('rhs')

        return EquationTerm(Term.from_str(lhs), Term.from_str(rhs))

    def __str__(self):
        return '{}={}'.format(str(self.lhs), str(self.rhs))

    def __repr__(self):
        return 'EquationTerm(%s)' % str(self)


class Formula(object):
    def __init__(self,
                 sat_formula: Optional[SatFormula] = None,
                 var_to_equation_mapping: Optional[Dict[str, EquationTerm]] = None):

        self.var_equation_mapping = var_to_equation_mapping or dict()
        self.sat_formula = sat_formula
        self.terms = []  # type: List[PureTerm]
        self.functions = []  # type: List[EquationTerm]

    def __str__(self):
        formula = str(self.sat_formula)
        for var, equation in self.var_equation_mapping.items():
            formula = formula.replace(var, str(equation))
        return formula

    def __repr__(self):
        return 'Formula(functions: %s\n equations: %s\n terms: %s\n encoded formula: %s' % \
               (str(self.functions), str(self.var_equation_mapping.values()), str(self.terms), str(self.sat_formula))

    @staticmethod
    def from_str(raw_formula: str) -> 'Formula':
        def from_str_helper(raw_formula: str) -> Optional[SatFormula]:
            if not raw_formula:
                return None

            m_unary = re.match(unary_formula_pattern, raw_formula)
            m_binary = re.match(binary_formula_pattern, raw_formula)

            if not m_unary and not m_binary:
                equation = EquationTerm.from_str(raw_formula)
                formula.var_equation_mapping[equation.name] = equation
                return SatFormula.create_leaf(equation.name)

            if m_binary:
                op = Operator(m_binary.group('op'))
                left, right = m_binary.group('left'), m_binary.group('right')

            else:
                op = Operator(m_unary.group('op'))
                left, right = m_unary.group('left'), None

            left = from_str_helper(left)
            right = from_str_helper(right)
            return SatFormula(left, right, op)

        formula = Formula()
        formula.sat_formula = from_str_helper(raw_formula)
        formula.get_terms(update=True)
        formula.get_functions(update=True)
        return formula

    def get_terms(self, update=False) -> List[Term]:
        if update or not self.terms:
            self.terms = []
            for equation in self.var_equation_mapping.values():
                self.terms += equation.get_terms()

        return self.terms

    def get_functions(self, update=False) -> List[Term]:
        if update or not self.terms:
            self.functions = []
            for equation in self.var_equation_mapping.values():
                self.terms += equation.get_functions()

        return self.functions

    def get_equations(self) -> List[Term]:
        return list(self.var_equation_mapping.values())
