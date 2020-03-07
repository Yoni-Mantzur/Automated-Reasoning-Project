import copy
from collections import defaultdict
from itertools import product
from typing import cast

from smt_solver.formula import Formula, Term, FunctionTerm


class CongruenceClosureAlgorithm(object):
    def __init__(self, formula: Formula):
        self.formula = copy.deepcopy(formula)

        for equality in self.formula.equalities:
            self.process(t1=self.formula.equations[equality].lhs, t2=self.formula.equations[equality].rhs)

    def find_representative(self, t: Term) -> Term:
        t = self.formula.terms[t.next_ptr]
        while t.next_ptr != t.idx:
            t = self.formula.terms[t.next_ptr]
        return t

    def merge(self, t1: Term, t2: Term) -> bool:
        # find representative
        t1_rep = self.find_representative(t1)
        t2_rep = self.find_representative(t2)

        if t1_rep == t2_rep:
            return False

        # merge parents
        t2_rep.parents |= t1_rep.parents
        t1_rep.parents = set()

        # Change representative ptr
        t1_rep.next_ptr = t2_rep.idx
        return True

    def process(self, t1: Term, t2: Term) -> None:
        # find parents
        t1_rep = self.find_representative(t1)
        t2_rep = self.find_representative(t2)
        parents_pairs = set(product(t1_rep.parents, t2_rep.parents))

        # merge classes
        if not self.merge(t1, t2):
            return

        # merge parents
        for p1, p2 in map(lambda p: (self.formula.terms[p[0]], self.formula.terms[p[1]]), parents_pairs):
            if self.is_merging_functions(p1, p2):
                p1, p2 = cast(FunctionTerm, p1), cast(FunctionTerm, p2)
                if not self.args_function_in_same_class(p1, p2):
                    continue
            self.process(p1, p2)

    def is_equation_is_true(self, equation: int) -> bool:
        t1 = self.formula.equations[equation].lhs
        t2 = self.formula.equations[equation].rhs

        return self.is_in_same_class(t1, t2)

    def is_legal_sets(self) -> bool:
        for inequality in self.formula.inequalities:
            if self.is_equation_is_true(inequality):
                return False
        return True

    def is_in_same_class(self, t1: Term, t2: Term) -> bool:
        t1_rep = self.find_representative(t1)
        t2_rep = self.find_representative(t2)

        return t1_rep == t2_rep

    def get_sets(self):
        sets = defaultdict(set)

        for term in self.formula.terms.values():
            rep = self.find_representative(term)
            sets[rep.idx] |= {term}

        return sets

    def is_merging_functions(self, t1, t2):
        return isinstance(t1, FunctionTerm) and isinstance(t2, FunctionTerm)

    def args_function_in_same_class(self, t1: FunctionTerm, t2: FunctionTerm):
        return t1.name == t2.name and \
            all((self.is_in_same_class(arg1, arg2) for arg1, arg2 in zip(t1.input_terms, t2.input_terms)))
