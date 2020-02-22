from lp_solver.lp_program import Equation


def test_equation_parsing():
    for raw_equation, equation in \
            [['4x1,5x2,-6x3,>=', Equation({4: 1, 5: 2, -6: 3}, Equation.Type.GE)],
             ['-4x1,5x2,6x3,<=', Equation({-4: 1, 5: 2, 6: 3}, Equation.Type.LE)],
             ['-4x1,5x2,6x3,=', Equation({-4: 1, 5: 2, 6: 3}, Equation.Type.EQ)],
             ['4.3x2,5x3,x1,=', Equation({4.3: 2, 5: 3, 1: 1}, Equation.Type.EQ)],
             ['3.1x5,5x2,0.5x1, <=', Equation({3.1: 5, 5: 2, 0.5: 1}, Equation.Type.LE)]]:

        print("testing the equation: %s" % raw_equation)
        assert Equation.from_str(raw_equation) == equation
