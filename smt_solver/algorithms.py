from itertools import product
from typing import Set, Dict

from sat_solver.DPLL import DPLL
from sat_solver.cnf_formula import CnfFormula
from smt_solver.formula import Formula, Term, EquationTerm


def congruence_closure_algorithm(terms: Dict[int, Term],
                                 equations: Dict[int, EquationTerm],
                                 equalities: Set[int],
                                 inequalities: Set[int]) -> bool:
    def find_representative(t: Term) -> Term:
        next_ptr = t.next_ptr
        t = terms[next_ptr]
        while next_ptr != t.idx:
            next_ptr = t.next_ptr

        return t

    def merge(t1, t2):
        # find representative
        t1_rep = find_representative(t1)
        t2_rep = find_representative(t2)

        if t1_rep == t2_rep:
            return False

        # merge parents
        t2_rep.parents |= t1_rep.parents
        t1_rep.parents = set()

        # Change representative ptr
        t1_rep.next_ptr = t2_rep.idx
        return True

    def process(t1, t2):
        # find parents
        t1_rep = find_representative(t1)
        t2_rep = find_representative(t2)
        parents_pairs = set(product(t1_rep.parents, t2_rep.parents))

        # merge classes
        if not merge(t1, t2):
            return

        # merge parents
        for p1, p2 in map(lambda p: (terms[p[0]], terms[p[1]]), parents_pairs):
            process(p1, p2)

    def is_legal_sets():
        for inequality in inequalities:
            t1_rep = find_representative(equations[inequality].lhs)
            t2_rep = find_representative(equations[inequality].rhs)

            if t1_rep.next_ptr == t2_rep.next_ptr:
                return False
        return True

    for equality in equalities:
        process(t1=equations[equality].lhs, t2=equations[equality].rhs)

    return is_legal_sets()


def solve(formula: str) -> bool:
    formula = Formula.from_str(formula)

    while True:
        # Solve sat formula
        dpll_algorithm = DPLL(formula.cnf_formula)
        is_sat, partial_assignment = dpll_algorithm.search(formula.propagate)

        # Sat formula is unsat hence, smt formula is ussat
        if not is_sat:
            return False

        # Sat formula is sat hence need to check smt formula
        formula.update_mappings(partial_assignment)
        if formula.satisfied():
            return True

        # Smt formula is unsat, add conflict
        formula.conflict(partial_assignment)
