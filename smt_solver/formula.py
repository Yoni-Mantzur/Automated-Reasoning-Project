import re
from abc import abstractmethod
from itertools import count
from typing import List, Dict, Optional, Tuple, cast, Set, Union

from common.operator import Operator
from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import SatFormula
from smt_solver.patterns import *


class Term(object):
    _ids = count(0)

    def __init__(self, name):
        self.idx = next(self._ids)
        self.name = name
        self.parent = None

    def get_terms(self) -> Set['Term']:
        return set()

    def get_functions(self) -> Set['Term']:
        return set()

    @staticmethod
    def from_str(term: str, formula) -> 'Term':
        m = re.match(term_pattern, term)
        assert m

        if m.group('function'):
            return FunctionTerm.from_str(term, formula)

        if m.group('variable'):
            return PureTerm.from_str(term, formula)

        raise Exception('Not a term string')

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

class PureTerm(Term):

    def __init__(self, name):
        super(PureTerm, self).__init__(name)
        self.name = name

    def get_terms(self):
        return {self}

    @staticmethod
    def from_str(term: str, formula) -> 'Term':
        if term in formula.terms_to_idx:
            return formula.terms[formula.terms_to_idx[term]]

        pure_term = PureTerm(term)
        formula.terms_to_idx[term] = pure_term.idx
        formula.terms[pure_term.idx] = pure_term
        return pure_term

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'PureTerm(%s)' % self.name


class FunctionTerm(Term):

    def __init__(self, name, input_terms: List[Term]):
        super(FunctionTerm, self).__init__(name)
        self.input_terms = input_terms

        for input_term in self.input_terms:
            input_term.parent = self

    def get_terms(self):
        terms = set()
        for input_term in self.input_terms:
            terms.update(input_term.get_terms())
        return terms

    def get_functions(self):
        functions = {self}
        for input_term in self.input_terms:
            functions.update(input_term.get_functions())
        return functions

    @staticmethod
    def from_str(term: str, formula) -> 'FunctionTerm':
        def is_function(t: str) -> bool:
            match = re.search(function_pattern, t)
            return match and match.start() == 0

        def get_name(t: str) -> str:
            return re.search(variable_pattern, t).group(1)

        def extract_args(name: str, term_suffix: str) -> Tuple[List[Term], str]:
            args = []
            term_suffix = term_suffix[len(name) + 1:]
            while term_suffix[0] != ')':
                term, term_suffix = from_str_helper(term_suffix)
                args.append(term)
            return args, term_suffix

        def from_str_helper(prefix_term: str) -> Tuple[Term, str]:
            if prefix_term[0] == ',':
                return from_str_helper(prefix_term[1:])

            if is_function(prefix_term):
                func_name = get_name(prefix_term)
                args, prefix_term = extract_args(func_name, prefix_term)
                function_term = FunctionTerm(func_name, args)
                if function_term in formula.terms_to_idx:
                    return formula.terms[formula.terms_to_idx[function_term]], prefix_term[1:]

                formula.terms_to_idx[str(function_term)] = function_term.idx
                formula.terms[function_term.idx] = function_term
                return function_term, prefix_term[1:]

            var_name = get_name(prefix_term)
            if var_name in formula.terms_to_idx:
                return formula.terms[formula.terms_to_idx[var_name]], prefix_term[len(var_name):]

            variable = PureTerm(var_name)
            formula.terms_to_idx[var_name] = variable.idx
            formula.terms[variable.idx] = variable
            return variable, prefix_term[len(var_name):]

        return cast(FunctionTerm, from_str_helper(term)[0])

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
        self.negated = False

    def get_terms(self):
        terms = self.lhs.get_terms()
        terms.update(self.rhs.get_terms())
        return terms

    def get_functions(self):
        functions = self.lhs.get_functions()
        functions.update(self.rhs.get_functions())
        return functions

    @staticmethod
    def from_str(term: str, formula) -> Term:
        if term in formula.equations_to_idx:
            return formula.equations[formula.equations_to_idx[term]]

        m = re.match(equation_pattern, term)
        assert m;
        lhs = m.group('lhs')
        rhs = m.group('rhs')

        equation_term = EquationTerm(Term.from_str(lhs, formula), Term.from_str(rhs, formula))
        formula.equations_to_idx[str(equation_term)] = equation_term.idx
        formula.equations[equation_term.idx] = equation_term
        return equation_term

    def __str__(self):
        return '{}={}'.format(str(self.lhs), str(self.rhs))

    def __repr__(self):
        return 'EquationTerm(%s)' % str(self)


class Formula(object):
    def __init__(self):
        self.sat_formula = None  # type: Optional[Union[SatFormula, CnfFormula]]

        self.equations = {}  # type: Dict[int, EquationTerm]
        self.equations_to_idx = {}  # type: Dict[str, int]

        self.terms = {}  # type: Dict[int, Term]
        self.terms_to_idx = {}  # type: Dict[str, int]

        self.var_equalities_mapping = {}  # type: Dict[str, int]
        self.var_inequalities_mapping = {}  # type: Dict[str, int]


    def __str__(self):
        def replace_vars(formula, mapping, negate):
            for var, equation_idx in mapping.items():
                formula = formula.replace('%s%s' %(negate, var), str(self.equations[equation_idx]))
            return formula

        formula = str(self.sat_formula)
        formula = replace_vars(formula, self.var_equalities_mapping, negate='')
        formula = replace_vars(formula, self.var_inequalities_mapping, negate='~')

        return formula

    def __repr__(self):
        return 'Formula(%s)' % str(self)

    @staticmethod
    def from_str(raw_formula: str) -> 'Formula':
        def from_str_helper(raw_formula: str) -> Optional[SatFormula]:
            if not raw_formula:
                return None

            m_unary = re.match(unary_formula_pattern, raw_formula)
            m_binary = re.match(binary_formula_pattern, raw_formula)

            if not m_unary and not m_binary:
                equation = EquationTerm.from_str(raw_formula, formula)
                return SatFormula.create_leaf(equation.name)

            left, right = None, None
            if m_binary:
                op = Operator(m_binary.group('op'))
                left = from_str_helper(m_binary.group('left'))
                right = from_str_helper(m_binary.group('right'))

            else:
                op = Operator(m_unary.group('op'))
                left = from_str_helper(m_unary.group('left'))
                if left.is_leaf:
                    left.value.negate()

            return SatFormula(left, right, op)

        formula = Formula()
        formula.sat_formula = from_str_helper(raw_formula)
        formula.update_mappings()
        return formula


    def update_mappings(self):
        for idx, equation in self.equations.items():
            if equation.negated:
                self.var_inequalities_mapping[equation.name] = idx
            else:
                self.var_equalities_mapping[equation.name] = idx

    def get_equalities(self) -> Set[Term]:
        return set(map(lambda idx: self.equations[idx], self.var_equalities_mapping.values()))

    def get_inequalities(self) -> Set[Term]:
        return set(map(lambda idx: self.equations[idx], self.var_inequalities_mapping.values()))
