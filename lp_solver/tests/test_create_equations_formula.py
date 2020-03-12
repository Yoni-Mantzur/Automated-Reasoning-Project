import pytest

from lp_solver.lp_theory import extract_equations_from_formula

tests = [['(x1,x2<=3&2x2,-1x1<=1)', ['x1,x2<=3', '2x2,-1x1<=1']],
         ['(x1,x2<=3->2x2,-1x1<=1)', ['x1,x2<=3', '2x2,-1x1<=1']],
         ['(x1,x2<=3<->2x2,-1x1<=1)', ['x1,x2<=3', '2x2,-1x1<=1']],
         ['~(x1,x2<=3&2x2,-1x1<=1)', ['x1,x2<=3', '2x2,-1x1<=1']],
         ['~(x1,x2<=3|2x2,-1x1<=1)', ['x1,x2<=3', '2x2,-1x1<=1']],
         ['(((x1,x2<=3|2x2,-1x1<=1)&x1<=2)->x3<=1)', ['x1,x2<=3', '2x2,-1x1<=1', 'x1<=2', 'x3<=1']]
         ]


@pytest.mark.parametrize(['raw', 'expected'], tests)
def test_extract_equations_from_formula(raw, expected):
    assert sorted(extract_equations_from_formula(raw)) == sorted(expected)
