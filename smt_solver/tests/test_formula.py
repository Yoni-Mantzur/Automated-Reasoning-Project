from smt_solver.formula import Term, FunctionTerm, PureTerm, Formula


def test_term_str(debug=True):
    if debug:
        print('Testing representation of the term f(s(a),x)')
    term = FunctionTerm('f', [FunctionTerm('s', [PureTerm('0')]), PureTerm('x')])
    assert str(term) == 'f(s(0),x)'


def test_term_from_str(debug=True):
    for s in ['f(a1,g(x1))', 'a12', 's(s(s(a1)))', 'plus(x1,s(y1))']:
        if debug:
            print('Parsing', s, 'as a Term...')
        term = Term.from_str(s)
        if debug:
            print('... and got', term)
        assert str(term) == s


def test_formula_from_str(debug=True):
    for s in ['~r(r(x))=x', 'a=a', '(r(x)=x|q(y)=y)', '(a=a&x=x)', '((r(x)=x&x=a)|q(x)=a)',
              'r(r(x),y)=x', 'plus(s(x),y,s(plus(x,y)))=x', 'r(x8,x7,c)=a',
              'r(x,y)=x', '~~~q(x)=x']:
        if debug:
            print('Parsing', s, 'as a first-order formula...')
        formula = Formula.from_str(s)
        if debug:
            print('.. and got', formula)
        assert str(formula) == s
