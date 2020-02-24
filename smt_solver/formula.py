import re
from itertools import count
from typing import List, Dict, Optional, Tuple, Set

from common.operator import Operator
from sat_solver.DPLL import DPLL
from sat_solver.cnf_formula import CnfFormula
from sat_solver.sat_formula import SatFormula, Variable, Literal
from smt_solver.patterns import *

NEGATED = True


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
            return FunctionTerm.from_str(term, formula)[0]

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
    def from_str(term: str, formula) -> Tuple[Term, str]:
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

        return from_str_helper(term)

    def __str__(self) -> str:
        return self.extract_function_as_str()

    def extract_function_as_str(self) -> str:
        return self.name + '({})'.format(','.join([str(term) for term in self.input_terms]))

    def __repr__(self):
        return 'EquationTerm(%s)' % str(self)


class EquationTerm(Term):
    def __init__(self, lhs: Term, rhs: Term):
        super(EquationTerm, self).__init__(name=None)
        self.fake_variable = Variable(name='v%d' % self.idx)
        self.fake_literals = [None, None]  # type: List[Optional[Literal]]
        self.name = self.fake_variable.name
        self.lhs = lhs
        self.rhs = rhs

    def add_literal(self, negated: bool):
        self.fake_literals[negated] = Literal(self.fake_variable, negated)

    def get_terms(self):
        terms = self.lhs.get_terms()
        terms.update(self.rhs.get_terms())
        return terms

    def get_functions(self):
        functions = self.lhs.get_functions()
        functions.update(self.rhs.get_functions())
        return functions

    def __str__(self):
        return str({self.lhs, self.rhs})

    def __repr__(self):
        return 'EquationTerm(%s)' % str(self)


class Formula(object):
    def __init__(self):
        self.sat_formula = None  # type: SatFormula
        self.conflicts = set()

        self.equations = {}  # type: Dict[int, EquationTerm]
        self.equations_to_idx = {}  # type: Dict[str, int]

        self.terms = {}  # type: Dict[int, Term]
        self.terms_to_idx = {}  # type: Dict[str, int]

        self.var_equation_mapping = {}  # type: Dict[Variable, int]
        self.equalities = set()  # type: Set[int]
        self.inequalities = set()  # type: Set[int]

    def __str__(self):
        formula = str(self.sat_formula)
        for literal, equation_idx in self.var_equation_mapping.items():

            equation = self.equations[equation_idx]
            formula = formula.replace(str(literal), '{}={}'.format(str(equation.lhs), str(equation.rhs)))
        return formula

    def __repr__(self):
        return 'Formula(%s)' % str(self)

    @staticmethod
    def from_str(raw_formula: str) -> 'Formula':
        def from_str_helper(raw_formula: str) -> Tuple[Optional[SatFormula], str]:
            if not raw_formula:
                return None, ''

            # Binary case
            if raw_formula[0] == '(':
                left, s = from_str_helper(raw_formula[1:])
                op = Operator(s[0])
                right, s = from_str_helper(s[1:])
                return SatFormula(left, right, op), s[1:]

            # Unary case
            if raw_formula[0] == Operator.NEGATION.value:
                op = Operator.NEGATION
                left, s = from_str_helper(raw_formula[1:])

                if left.is_leaf:
                    left.value.negate()
                    return left, s

                return SatFormula(left, None, op), s

            # Equation case:
            else:
                t1, s = FunctionTerm.from_str(raw_formula, formula)
                t2, s = FunctionTerm.from_str(s[1:], formula)
                equation = EquationTerm(t1, t2)
                equation = formula.update_equations(equation)
                return SatFormula.create_leaf(equation.name, equation.fake_variable), s

        formula = Formula()
        formula.sat_formula = from_str_helper(raw_formula)[0]
        formula.set_literals()
        return formula

    def update_equations(self, equation: EquationTerm) -> EquationTerm:
        # Update equations mappings
        if equation not in self.equations_to_idx:
            self.equations_to_idx[str(equation)] = equation.idx
            self.equations[equation.idx] = equation
            self.var_equation_mapping[equation.fake_variable] = equation.idx

        equation = self.equations[self.equations_to_idx[equation]]
        return equation

    def set_literals(self):
        for literal in self.sat_formula.get_literals():
            equation = self.equations[self.var_equation_mapping[literal.variable]]
            equation.add_literal(literal.negated)

    def update_mappings(self, partial_assignment: Dict[Variable, bool]):
        self.equalities = set()
        self.inequalities = set()

        for v, assigned_true in partial_assignment.items():
            equation = self.equations[self.var_equation_mapping[v]]

            if assigned_true:

                if equation.fake_literals[not NEGATED]:
                    self.equalities.add(equation.idx)

                if equation.fake_literals[NEGATED]:
                    self.inequalities.add(equation.idx)

            else:

                if equation.fake_literals[NEGATED]:
                    self.equalities.add(equation.idx)

                elif equation.fake_literals[not NEGATED]:
                    self.inequalities.add(equation.idx)

    def satisfied(self, partial_assignment: Dict[Variable, bool]) -> bool:
        from smt_solver.algorithms import CongruenceClosureAlgorithm
        self.update_mappings(partial_assignment)
        return CongruenceClosureAlgorithm(self).is_legal_sets()

    def conflict(self, partial_assignment: Dict[Variable, bool]) -> List[Literal]:
        '''
        Assuming there is a conflict in the current assignment and returns the clause which casues it
        :param partial_assignment:
        :return:
        '''

        conflict = []

        # No conflicts
        if self.satisfied(partial_assignment):
            return conflict

        for v, value in partial_assignment.items():
            equation = self.equations[self.var_equation_mapping[v]]
            negated_value = value
            literal = equation.fake_literals[negated_value]
            conflict.append(literal)
        return conflict

    def propagate(self, partial_assignment: Dict[Variable, bool]) -> bool:
        '''
        Checks if the partial assignment is valid, if so, extend the assignment (propagation) and return True
        otherwise, return False
        '''
        from smt_solver.algorithms import CongruenceClosureAlgorithm

        self.update_mappings(partial_assignment)
        classes_algorithm = CongruenceClosureAlgorithm(self)

        if not classes_algorithm.is_legal_sets():
            return False

        for var, equation_idx in self.var_equation_mapping.items():
            non_assigned_var = var not in partial_assignment
            if non_assigned_var:

                assign_true = classes_algorithm.is_equation_is_true(equation_idx)
                equation = self.equations[equation_idx]

                # Conflict - same class, but literal appears with negated and with out
                if assign_true and equation.fake_literals[NEGATED] and equation.fake_literals[not NEGATED]:
                    return False

                # Assign False - same class, but literal appears with negated
                elif assign_true and equation.fake_literals[NEGATED]:
                    partial_assignment[var] = not assign_true

                # Assign True - same class and literal appears without negated
                elif assign_true and equation.fake_literals[not NEGATED]:
                    partial_assignment[var] = assign_true

        return True

    def solve(self) -> bool:
        cnf_formula = CnfFormula.from_str(str(self.sat_formula))
        dpll_algorithm = DPLL(cnf_formula, propagate_helper=self.propagate, conflict_helper=self.conflict)

        # Solve sat formula
        is_sat = dpll_algorithm.search()

        # Sat formula is unsat hence, smt formula is ussat
        if not is_sat:
            return False

        partial_assignment = dpll_algorithm.get_partial_assignment()

        # DEBUG - Sat formula is sat hence need to check smt formula
        print(partial_assignment)
        self.update_mappings(partial_assignment)
        return self.satisfied(partial_assignment)
