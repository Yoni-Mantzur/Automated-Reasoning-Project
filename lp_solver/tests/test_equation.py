from lp_solver.lp_program import Equation


def test_equation_parsing():
    for raw_equation, equation in \
            [['4x1,5x2,-6x3>=2', Equation({1: 4, 2: 5, 3: -6}, Equation.Type.GE, 2)],
             ['-4x1,5x2,6x3<=-4', Equation({1: -4, 2: 5, 3: 6}, Equation.Type.LE, -4)],
             ['-4x1,5x2,6x3=1', Equation({1: -4, 2: 5, 3: 6}, Equation.Type.EQ, 1)],
             ['4.3x2,5x3,x1=0', Equation({2: 4.3, 3: 5, 1: 1}, Equation.Type.EQ, 0)],
             ['3.1x5,5x2,0.5x1<=3', Equation({5: 3.1, 2: 5, 1: 0.5}, Equation.Type.LE, 3)],
             ['2.1x0>=-1', Equation({0: 2.1}, Equation.Type.GE, -1)]]:

        # print("testing the equation: %s" % raw_equation)
        assert Equation.get_equation(raw_equation) == equation

def test_expression_parsing():
    for raw_equation, equation in \
            [['4x1,5x2,-6x3', Equation({1: 4, 2: 5, 3: -6})],
             ['-4x1,5x2,6x3', Equation({1: -4, 2: 5, 3: 6})],
             ['-4x1,5x2,6x3', Equation({1: -4, 2: 5, 3: 6})],
             ['4.3x2,5x3,x1', Equation({2: 4.3, 3: 5, 1: 1})],
             ['3.1x5,5x2,0.5x1', Equation({5: 3.1, 2: 5, 1: 0.5})],
             ['2.1x0', Equation({0: 2.1})]]:

        # print("testing the equation: %s" % raw_equation)
        assert Equation.get_expression(raw_equation) == equation

if __name__ == "__main__":
    test_equation_parsing()