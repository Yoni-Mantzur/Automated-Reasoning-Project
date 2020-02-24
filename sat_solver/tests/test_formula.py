from sat_solver.sat_formula import SatFormula
from common.operator import Operator

def test_str(debug=True):
    if debug:
        print("Testing formula 'x12'")
    assert str(SatFormula.from_str('x12')) == 'x12'
    if debug:
        print("Testing formula '(p1|p1)'")
    assert str(SatFormula(SatFormula.from_str('p1'), SatFormula.from_str('p1'), Operator.OR)) == '(p1|p1)'
    if debug:
        print("Testing formula '~(p1&q7)'")
    assert str(SatFormula(SatFormula(SatFormula.from_str('p1'), SatFormula.from_str('q7'), Operator.AND), None,
                          Operator.NEGATION)) == '~(p1&q7)'

def test_from_str(debug=True):
    for infix in ['(((x2&x5)|x3)&(x1|x5))', '~~(x1&~T)','~~p1', '~x12', '(x1&y1)', '~~(x1|~T)', '((x1&~x2)|F)']:
        if debug:
            print("Testing from str parsing of formula", infix)
            assert str(SatFormula.from_str(infix)) == infix
