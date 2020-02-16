from smt_solver.algorithms import satisfied
from smt_solver.formula import Formula


def test_cca(debug=True):
    for s, is_sat in [('(f(a,b)=a&~f(f(a,b),b)=a)', False),
                      ('(f(x)=f(y)&~x=y)', True),
                      ('((f(f(f(a)))=a&f(f(f(f(f(a)))))=a)&f(a)=a)', True)]:
        if debug:
            print('Parsing', s, 'as a first-order formula...')
        formula = Formula.from_str(s)
        if debug:
            print('.. and got', formula)
        assert satisfied(formula) == is_sat
