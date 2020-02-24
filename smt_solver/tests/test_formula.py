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
        term = Term.from_str(s, Formula())
        if debug:
            print('... and got', term)
        assert str(term) == s


def test_formula_from_str(debug=True):
    for s in ['~q(x)=x', '((x=y|x=z)&(~x=y|z=r))', '~r(r(x))=x', '(a=a|a=a)', '(r(x)=x|q(y)=y)', '(a=a&x=x)',
              '((r(x)=x&x=a)|q(x)=a)', 'r(r(x),y)=x', 'plus(s(x),y,s(plus(x,y)))=x', 'r(x8,x7,c)=a', 'r(x,y)=x']:
        if debug:
            print('Parsing', s, 'as a first-order formula...')
        formula = Formula.from_str(s)
        if debug:
            print('.. and got', formula)
        assert str(formula) == s


def test_get_terms(debug=True):
    for s, term_set in [['f(f(a,b),b)=b', {'f(f(a,b),b)', 'f(a,b)', 'b', 'a'}],
                        ['~r(r(x))=r(x)|r(x)=x', {'r(x)', 'x', 'r(r(x))'}],
                        ['(r(x)=x|q(y)=y)', {'r(x)', 'x', 'q(y)', 'y'}],
                        ['plus(s(x),y,s(plus(x,y)))=x', {'plus(s(x),y,s(plus(x,y)))', 's(x)', 'y',
                                                         's(plus(x,y))', 'plus(x,y)', 'x'}]]:
        if debug:
            print('Parsing', s, 'as a first-order formula...')
        formula = Formula.from_str(s)
        if debug:
            print('.. and got', formula)
        actual_term_set = {str(term) for term in formula.terms.values()}
        assert len(actual_term_set) == len(formula.terms.values())
        assert actual_term_set == term_set


def test_get_parents(debug=True):
    for s, term_set, parents in [['f(f(a,b),b)=b', {'f(f(a,b),b)', 'f(a,b)', 'b', 'a'}, {'b': {'f(f(a,b),b)', 'f(a,b)'},
                                                                                         'a': {'f(a,b)'},
                                                                                         'f(a,b)': {'f(f(a,b),b)'},
                                                                                         'f(f(a,b),b)': set()}]]:
        if debug:
            print('Parsing', s, 'as a first-order formula...')
        formula = Formula.from_str(s)
        if debug:
            print('.. and got', formula)
        actual_term_set = {str(term) for term in formula.terms.values()}
        assert len(actual_term_set) == len(formula.terms.values())
        assert actual_term_set == term_set
        for term in term_set:
            actual_parents = set(map(lambda idx_parent: str(formula.terms[idx_parent]),
                                     formula.terms[formula.terms_to_idx[term]].parents))
            assert parents[term] == actual_parents


def test_satisfied(debug=True):
    for s, is_sat in [('(f(a,b)=a&~f(f(a,b),b)=a)', False),
                      ('(f(x)=f(y)&~x=y)', True),
                      ('((f(f(f(a)))=a&f(f(f(f(f(a)))))=a)&f(a)=a)', True)]:
        if debug:
            print('Parsing', s, 'as a first-order formula...')
    formula = Formula.from_str(s)
    if debug:
        print('.. and got', formula)
    assignment_all_true = {v: True for v in formula.var_equation_mapping.keys()}
    assert formula.satisfied(assignment_all_true) == is_sat


def test_literals():
    for s in ['(x=y&f(x)=f(y))', '(x=y&~f(x)=f(y))', '((x=y&~f(x)=f(y))|~x=y)']:
        formula = Formula.from_str(s)

        literals = formula.sat_formula.get_literals()
        for literal in literals:
            negated = literal.negated
            equation_idx = formula.var_equation_mapping[literal.variable]
            assert literal == formula.equations[equation_idx].fake_literals[negated]

            if formula.equations[equation_idx].fake_literals[not negated]:
                assert formula.equations[equation_idx].fake_literals[not negated] in literals


def test_propagation(debug=True):
    for s, before, after in [['((x=y&~f(x)=f(y))|~y=x)', {0: True}, False],
                             ['(x=y&f(x)=f(y))', {0: True}, {0: True, 1: True}],
                             ['(x=y&~f(x)=f(y))', {0: True}, {0: True, 1: False}],
                             ['((x=y&~f(x)=f(y))|z=c)', {0: True}, {0: True, 1: False}]]:

        if debug:
            print('Parsing', s, 'as a first-order formula...')
        formula = Formula.from_str(s)
        variables = list(formula.var_equation_mapping.keys())
        partial_assignment = {variables[v_idx]: value for v_idx, value in before.items()}
        if type(after) == bool:
            expected_assignment = after
        else:
            expected_assignment = {variables[v_idx]: value for v_idx, value in after.items()}

        res = formula.propagate(partial_assignment)

        if type(after) == dict:
            assert res
            assert partial_assignment == expected_assignment

        else:
            assert after == res


def test_conflict(debug=True):
    for s, assignment, conflict in [['((x=y&~f(x)=f(y))|~y=x)', {0: True}, {0: False}],
                                    ['(x=y&~f(x)=f(y))', {0: True, 1: True}, {0: False, 1: False}],
                                    ['((x=y&~f(x)=f(y))|z=c)', {0: True, 1: True}, {0: False, 1: False}],
                                    ['((x=y&~f(x)=f(y))&z=c)', {0: True, 1: False}, []]]:
        if debug:
            print('Parsing', s, 'as a first-order formula...')
        formula = Formula.from_str(s)
        variables = list(formula.var_equation_mapping.keys())
        partial_assignment = {variables[v_idx]: value for v_idx, value in assignment.items()}
        if conflict:
            expected_conflict = [formula.equations[formula.var_equation_mapping[variables[v_idx]]].fake_literals[value]
                                 for v_idx, value in assignment.items()]
        else:
            expected_conflict = []

        conflict = formula.conflict(partial_assignment)

        assert expected_conflict == conflict
