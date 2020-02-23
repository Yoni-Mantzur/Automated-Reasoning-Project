import re
from itertools import count
from typing import List, Dict, Optional, Tuple, cast, Set

from common.operator import Operator
from sat_solver.DPLL import DPLL
from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import SatFormula
from smt_solver.patterns import *


class Term(object):
    _ids = count(0)

    def __init__(self, name):
        self.idx = next(self._ids)
        self.next_ptr = self.idx
        self.name = name
        self.parents = set()

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

    def __init__(self, name, input_terms: List[Term], add_parent=True):
        super(FunctionTerm, self).__init__(name)
        self.input_terms = input_terms

        if add_parent:
            for input_term in self.input_terms:
                input_term.parents.add(self.idx)

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
                function_term = FunctionTerm(func_name, args, add_parent=False)

                # Function already exits
                if function_term in formula.terms_to_idx:
                    f_term_ptr = formula.terms[formula.terms_to_idx[function_term]]
                    # Update parents
                    for input_term in f_term_ptr.input_terms:
                        input_term.parents.add(f_term_ptr.idx)
                    term = f_term_ptr
                else:
                    term = FunctionTerm(func_name, args)
                prefix_term = prefix_term[1:]

            else:
                var_name = get_name(prefix_term)
                # variable already exits
                if var_name in formula.terms_to_idx:
                    return formula.terms[formula.terms_to_idx[var_name]], prefix_term[len(var_name):]

                term, prefix_term = PureTerm(var_name), prefix_term[len(var_name):]

            # Update term mapping
            formula.terms_to_idx[str(term)] = term.idx
            formula.terms[term.idx] = term

            return term, prefix_term

        return cast(FunctionTerm, from_str_helper(term)[0])

    def __str__(self):
        return self.extract_function_as_str()

    def extract_function_as_str(self):
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

        # If equation exists will override it, and that is ok
        # Update equations mappings
        formula.equations_to_idx[str(equation_term)] = equation_term.idx
        formula.equations[equation_term.idx] = equation_term
        formula.var_equation_mapping[equation_term.name] = equation_term.idx

        return equation_term

    def __str__(self):
        return '{}={}'.format(str(self.lhs), str(self.rhs))

    def __repr__(self):
        return 'EquationTerm(%s)' % str(self)


class Formula(object):
    def __init__(self):
        self.sat_formula = None  # type: Optional[SatFormula]
        self.conflicts = set()

        self.equations = {}  # type: Dict[int, EquationTerm]
        self.equations_to_idx = {}  # type: Dict[str, int]

        self.terms = {}  # type: Dict[int, Term]
        self.terms_to_idx = {}  # type: Dict[str, int]

        self.var_equation_mapping = {}  # type: Dict[str, int]
        self.equalities = set()  # type: Set[int]
        self.inequalities = set()  # type: Set[int]

    def __str__(self):
        formula = str(self.sat_formula)
        for var, equation_idx in self.var_equation_mapping.items():
            negate = '~' if self.equations[equation_idx].negated else ''
            formula = formula.replace('%s%s' % (negate, var), str(self.equations[equation_idx]))
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
        formula.update_mappings({literal.name: literal.negated for literal in
                                 formula.sat_formula.get_literals()})
        return formula

    def update_mappings(self, partial_assignment: Dict[str, bool]):
        self.equalities = set()
        self.inequalities = set()
        for var_name, is_negated in partial_assignment.items():
            equation_idx = self.var_equation_mapping[var_name]
            is_negated |= self.equations[equation_idx].negated
            if is_negated:
                self.inequalities.add(equation_idx)
            else:
                self.equalities.add(equation_idx)

    def satisfied(self) -> bool:
        from smt_solver.algorithms import CongruenceClosureAlgorithm
        return CongruenceClosureAlgorithm(self).is_legal_sets()

    def conflict(self, partial_assignment: Dict[str, bool]) -> List[str]:
        '''
        Assuming there is a conflict in the current assignment and returns the clause which casues it
        :param partial_assignment:
        :return:
        '''
        assert not self.satisfied()
        conflict = []
        for var, value in partial_assignment.items():
            negated = '' if value else Operator.NEGATION.value
            conflict.append(negated + var)

        return conflict

    def propagate(self, partial_assignment: Dict[str, bool]) -> bool:
        '''
        Checks if the partial assignment is valid, if so, extend the assignment (propagation) and return True
        otherwise, return False
        '''
        from smt_solver.algorithms import CongruenceClosureAlgorithm

        self.update_mappings(partial_assignment)
        classes_algorithm = CongruenceClosureAlgorithm(self)

        if not classes_algorithm.is_legal_sets():
            return False

        unassigned_var_to_equations = filter(lambda v, _: v in partial_assignment, self.var_equation_mapping.items())
        for var, equation in unassigned_var_to_equations:
            if classes_algorithm.is_equation_is_true(equation):
                # Learn equality to be true and inequality to be false
                # TODO: verify with yuval that assignment can be by var name
                partial_assignment[var] = self.equations[equation].negated

    def solve(self) -> bool:
        cnf_formula = CnfFormula.from_str(str(self.sat_formula))
        dpll_algorithm = DPLL(cnf_formula)

        # Solve sat formula
        is_sat = dpll_algorithm.search(self.propagate, self.conflict)

        # Sat formula is unsat hence, smt formula is ussat
        if not is_sat:
            return False

        partial_assignment = dpll_algorithm.get_partial_assignment()
        #DEBUG - Sat formula is sat hence need to check smt formula
        self.update_mappings(partial_assignment)
        if self.satisfied():
            print(partial_assignment)
            return True
