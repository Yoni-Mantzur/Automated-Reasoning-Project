from typing import List, Optional, Set

from smt_solver.formula import Formula, Term


def congruence_closure_algorithm(terms: List[Term]) -> bool:
    class Element(object):
        def __init__(self, term):
            self.term = term
            self.ptr = None  # type: Optional[Term]
            self.parents = None  # type: Set[Element]

    class EquavilantClass(object):
        def __init__(self, element: Element) -> None:
            self.representative = None  # type: Optional[Element]
            self.elements = {element}  # type: Set[Element]

        def add(self, elements: Set[Element]):
            self.elements += elements

    def merge(equavilant_class_first, equavilant_class_second):
        pass

    def process(equation):
        pass

    equavilant_sets = [EquavilantClass(Element(term)) for term in terms]
    return False

def satisfied(formula: Formula) -> bool:
    terms = formula.get_functions() + formula.get_terms()
    return congruence_closure_algorithm(terms)