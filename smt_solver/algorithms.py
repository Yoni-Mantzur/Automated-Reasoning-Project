from itertools import count
from typing import List, Set, cast

from smt_solver.formula import Formula, Term, EquationTerm

def congruence_closure_algorithm(terms: Set[Term],
                                 equalities: Set[Term],
                                 inequalities: Set[Term]) -> bool:
    class Element(object):
        def __init__(self, term):
            self.id = term.id
            self.term = term.name
            self.inputs = []  # type: List[Element]
            self.parents = []  # type: List[Element]
            self.find_ptr = self.id
            self.equivalent_class = self.id

    class SubTermsDag(object):
        def __init__(self, terms):

            self.nodes = set()
            self.roots = []  # type: List[Element]

        def build_graph(self, terms):
            pass

        def add_root(self, element: Element):
            self.nodes.add(element)

    def merge(equavilant_class_first, equavilant_class_second):
        pass

    def process():
        pass

    def is_legal_sets():
        for inequality in inequalities:
            if inequality.lhs. in equivalent_set and inequality.rhs in equivalent_set:
                    return False
        return True


    for equality in equalities:
        process()

    return is_legal_sets()


def satisfied(formula: Formula) -> bool:
    terms = formula.get_terms()
    equalities = cast(Set[EquationTerm], formula.get_equalities())
    inequalities = cast(Set[EquationTerm], formula.get_inequalities())

    return congruence_closure_algorithm(terms, equalities, inequalities)
