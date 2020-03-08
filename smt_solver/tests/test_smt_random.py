from random import seed, randint, choices, choice, uniform
from timeit import default_timer as timer
from typing import List

import pytest
from z3 import Not, Or, And, sat, Solver, Real, Function, RealSort, ArithRef

from smt_solver.formula import Formula

seed(0)


class UF:
    def __init__(self, name: str, max_num_vars: int):
        self.name = name
        self.input_range = randint(1, min(max_num_vars, 5))
        z3_input = [RealSort()] * self.input_range
        self.z3 = Function(name, *z3_input, RealSort())

    def create_random_call(self, smt_vars: List['smt_var']) -> [str, ArithRef]:
        v = choices(smt_vars, k=self.input_range)
        v_str = "".join(str(v)).replace('[', '').replace(']', '')
        z3_func_call = self.z3(*[cur_v.z3 for cur_v in v])
        return "{}({})".format(self.name, v_str), z3_func_call

    def __str__(self):
        return self.name


class SmtVar:
    def __init__(self, name: str):
        self.name = name
        self.z3 = Real(name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class smt_clause:
    def __init__(self, functions: List['UF'], smt_variables: List['SmtVar']):
        self.ufs = functions
        self.smt_vars = smt_variables
        self.equality = uniform(0, 1) > 0.5
        self.op = choice(["&", "|"])

    def create_smt_literal(self):
        create_function = uniform(0, 1) > 0.5
        if create_function:
            func = choice(self.ufs)
            return func.create_random_call(self.smt_vars)
        else:
            v = choice(self.smt_vars)
            return v, v.z3

    def create_equality(self):
        lhs_str, lhs_z3 = self.create_smt_literal()
        rhs_str, rhs_z3 = self.create_smt_literal()

        smt_clause_str = "{}={}".format(lhs_str, rhs_str)
        smt_clause_z3 = lhs_z3 == rhs_z3
        if not self.equality:
            smt_clause_z3 = Not(smt_clause_z3)
            smt_clause_str = "~" + smt_clause_str

        return smt_clause_str, smt_clause_z3


def create_random_query(num_functions=4, num_variables=5, num_clauses=4):
    # '(((a=x&b=y)&f(a,b,c)=g(a,b,c))&~f(a,b,c)=g(x,y,z)')
    uf = [UF("f{}".format(i), num_variables) for i in range(num_functions)]

    smt_vars = [SmtVar("a{}".format(i)) for i in range(num_variables)]

    str_clauses = []
    z3_clauses = []
    for _ in range(num_clauses):
        clause = smt_clause(uf, smt_vars)
        str_c, z3_c = clause.create_equality()
        str_clauses.append(str_c)
        z3_clauses.append(z3_c)

    str_clause = str_clauses[0]
    z3_clause = z3_clauses[0]
    for i in range(1, len(str_clauses)):
        op = choice(['&', '|'][:1])
        if op == '&':
            z3_clause = And(z3_clause, z3_clauses[i])
        else:
            z3_clause = Or(z3_clause, z3_clauses[i])
        str_clause = f"({str_clause}{op}{str_clauses[i]})"

    return str_clause, z3_clause


def get_z3_result(query, debug=False) -> bool:
    s = Solver()

    s.add(query)
    if s.check() == sat:
        if debug:
            print(s, s.model())
        return True
    return False


def perform_test(str_query, z3_query, debug=False):
    z3_time_start = timer()
    z3_res = get_z3_result(z3_query, debug)
    z3_time_end = timer()

    our_time_start = timer()
    f = Formula.from_str(str_query.replace(' ', ''))
    res, assignment = f.solve()

    our_time_end = timer()

    assert res == z3_res, "q: {}, Our: {}, Z3: {}".format(str_query, res, z3_res)
    res_str = 'Sat ' if res else 'UNSAT '

    res_str += "Time(sec): Our {:0.2f}, z3: {:0.2f}".format(our_time_end - our_time_start, z3_time_end - z3_time_start)
    print(res_str)


random_smt = [[10, 20, 50], [20, 40, 100], [31, 40, 100], [20, 40, 120], [50, 50, 150]] \
             + choices([[f, v, c] for c in range(2, 30) for f in range(2, 10) for v in range(2, 10)], k=100)


@pytest.mark.parametrize(['num_functions', 'num_variables', 'num_clauses'], random_smt)
def test_random_smt(num_functions, num_variables, num_clauses):
    str_query, z3_query = create_random_query(num_functions=num_functions, num_variables=num_variables,
                                              num_clauses=num_clauses)
    perform_test(str_query, z3_query)
