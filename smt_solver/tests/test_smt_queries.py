import pytest

from smt_solver.formula import Formula


def test_simple():
    f = Formula.from_str('((x=y&y=z)&~f(x)=f(z))')
    print(f.sat_formula)
    res = f.solve()
    assert not res
    f = Formula.from_str('((x=y&y=z)&f(x)=f(z))')
    res = f.solve()
    assert res

def test_same_literals():
    f = Formula.from_str('((x=y|~f(x)=f(z))&x=y)')
    print(f.sat_formula)
    res = f.solve()
    assert res

def test_complicated_transitions():
    f = Formula.from_str('(((x=y&x=f(x))&f(x)=r(z))&r(z)=y)')
    print(f.sat_formula)
    res = f.solve()
    assert res

def test_nested_functions():
    pytest.skip()
    f = Formula.from_str('f(f(f(g(x)))=g(x))&g(f(f(f(x)))))=g(x)')
    print(f.sat_formula)
    res = f.solve()
    assert res

def test_complex():
    f = Formula.from_str('((g(a)=c&(~f(g(a))=f(c)|g(a)=d))&~c=d)')
    print(f.sat_formula)
    res = f.solve()
    assert not res
    # f = Formula.from_str('((x=y&y=z)&f(x)=f(z))')
    # res = f.solve()
    # assert res

def test_implies():
    f = Formula.from_str('(g(a)=b&g(b)=a)->b=a')
    print(f.sat_formula)
    res = f.solve()
    assert res

    f = Formula.from_str('(~g(a)=b|~g(b)=a)->b=a')
    print(f.sat_formula)
    res = f.solve()
    assert not res

def test_iff():
    pytest.skip()
    # a=b iff f(a) = f(b), SAT for example a=b & f=identity function
    f = Formula.from_str('(f(a)=f(b)<->a=b)')
    print(f.sat_formula)
    res = f.solve()
    assert res

def test_multi_variable_function():
    # a=x & b=y & f(a,b,c) = g(a,b,c) &f(a,b,c) != g(x,y,z) -> SAT since z!=c
    f = Formula.from_str('(((a=x&b=y)&f(a,b,c)=g(a,b,c))&~f(a,b,c)=g(x,y,z)')
    print(f.sat_formula)
    res = f.solve()
    assert res

    # a=x & b=y & c=z & f(a,b,c) = g(a,b,c) &f(a,b,c) != g(x,y,z) -> UNSAT
    f = Formula.from_str('((((a=x&b=y)&c=z)&f(a,b,c)=g(a,b,c))&~f(a,b,c)=g(x,y,z)')
    print(f.sat_formula)
    res = f.solve()
    assert not res

if __name__ == "__main__":
    test_implies()