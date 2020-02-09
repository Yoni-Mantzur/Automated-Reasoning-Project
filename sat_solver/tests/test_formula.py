from sat_solver.formula import Formula

def test_str(debug=True):
    if debug:
        print("Testing formula 'x12'")
    assert str(Formula.from_str('x12')) == 'x12'
    if debug:
        print("Testing formula '(p1|p1)'")
    assert str(Formula(Formula.from_str('p1'),Formula.from_str('p1'), Formula.Operator.OR)) == '(p1|p1)'
    if debug:
        print("Testing formula '~(p1&q7)'")
    assert str(Formula(Formula(Formula.from_str('p1'), Formula.from_str('q7'), Formula.Operator.AND), None,
                       Formula.Operator.NEGATION)) == '~(p1&q7)'

def test_from_str(debug=True):
    for infix in ['~~(x1&~T)','~~p1', '~x12', '(x1&y1)', '~~(x1|~T)', '((x1&~x2)|F)']:
        if debug:
            print("Testing from str parsing of formula", infix)
        assert str(Formula.from_str(infix)) == infix
