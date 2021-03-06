import numpy as np

from lp_solver.lp_theory import LpTheory


def test_auxiliry_lp():
    # HW3 Q1
    rule = 'dantzig'
    objective = 'x1,3x2'
    constraints = '((-1x1,x2<=-1&-2x1,-2x2<=-6)|-1x1,4x2<=2)'
    lp = LpTheory(constraints, objective, rule=rule)
    feasible = lp.solve()
    assert feasible
    # assert obj == np.inf


def test_simple_lp():
    # Lecture 11 Slide 19
    rule = 'dantzig'
    objective = '5x1,4x2,3x3'
    constraints = '((0x0,2x1,3x2,x3<=5&4x1,x2,2x3<=11)&3x1,4x2,2x3<=8)'
    lp = LpTheory(constraints, objective, rule=rule)
    feasible = lp.solve()
    assert feasible



def test_simple_lp_unfesiable():
    # Lecture 11 Slide 19
    rule = 'dantzig'
    objective = '5x1,4x2'

    constraints = '((x1<=1&x2<=1)&-1x1,-1x2<=-3)'
    lp = LpTheory(constraints, objective, rule=rule)
    feasible = lp.solve()
    assert not feasible


def test_negate():
    rule = 'dantzig'
    objective = '5x1'
    constraints = '(~x1<=1&x1<=1)'
    lp = LpTheory(constraints, objective, rule=rule)
    feasible = lp.solve()
    assert not feasible


def test_negate_complex():
    rule = 'dantzig'
    objective = '5x1,x2'
    constraints = '((~x1<=1&x1<=1)|(x2,x1<=1&x2<=1))'
    lp = LpTheory(constraints, objective, rule=rule)
    feasible = lp.solve()
    assert feasible

    constraints = '((~x1<=1&x1<=1)&(x2,x1<=1&x2<=1))'
    lp = LpTheory(constraints, objective, rule=rule)
    feasible = lp.solve()
    assert not feasible