from random import random, sample, seed, randint
from timeit import default_timer as timer
from typing import List

from tqdm import tqdm
from z3 import Bool, Solver, Or, Not, sat

from sat_solver.DPLL import DPLL
from sat_solver.cnf_formula import CnfFormula
from sat_solver.preprocessor import preprocess
from sat_solver.sat_formula import Variable, Literal


def create_random_query(num_variables=5, num_clauses=4, clause_length=3):
    assert clause_length <= num_variables
    variables = [Variable('x{}'.format(i + 1)) for i in range(num_variables)]
    pos_l = [Literal(v, negated=False) for v in variables]
    neg_l = [Literal(v, negated=True) for v in variables]

    clauses = []
    for _ in range(num_clauses):
        clause = []
        vars = sample(range(0, num_variables), clause_length)
        for i in range(clause_length):
            if random() > 0.5:
                clause.append(neg_l[vars[i]])
            else:
                clause.append(pos_l[vars[i]])
        clauses.append(clause)

    return clauses


def get_z3_result(clauses: List[Literal], debug=False) -> bool:
    s = Solver()
    variables = {l.variable: Bool(l.name) for c in clauses for l in c}

    for c in clauses:
        z3_clause = []
        for l in c:
            v = variables[l.variable]
            if l.negated:
                v = Not(v)
            z3_clause.append(v)
        s.add(Or(*z3_clause))

    if s.check() == sat:
        if debug:
            print(s, s.model())
        return True
    return False


def perform_test(clauses: List[List[Literal]], debug=False):
    z3_time_start = timer()
    z3_res = get_z3_result(clauses, debug)
    z3_time_end = timer()

    our_time_start = timer()
    cnf = CnfFormula(clauses)
    cnf = preprocess(cnf)
    dpll = DPLL(cnf)
    search_result = dpll.search()
    if debug:
        # print(clauses)
        print(dpll.get_full_assignment())
    our_time_end = timer()

    assert search_result == z3_res, "Our: {}, Z3: {}".format(search_result, z3_res)
    res_str = 'Sat ' if search_result else 'UNSAT '
    all_vars = set([lit.variable for clause in clauses for lit in clause])
    res_str += "#var: {}, #clauses: {} #per_clause: {} ".format(len(all_vars), len(clauses), len(clauses[0]))
    res_str += "Time(sec): Our {:0.2f}, z3: {:0.2f}".format(our_time_end - our_time_start, z3_time_end - z3_time_start)
    print(res_str)


if __name__ == "__main__":
    # get_z3_result(create_random_query(5, 2, 3))
    seed(0)

    for _ in tqdm(range(30)):
        num_var = randint(1, 10) * 2
        num_clause = randint(1, 10) * 10
        var_per_clause = randint(2, min(num_var, 10))
        clauses = create_random_query(num_var, num_clause, var_per_clause)
        perform_test(clauses, False)

    clauses = create_random_query(100, 1000, 2)
    perform_test(clauses, False)

    clauses = create_random_query(200, 1000, 4)
    perform_test(clauses, False)

    # TODO: Does not work :(
    exit(-1)
    clauses = create_random_query(100, 400, 3)
    perform_test(clauses, False)
