from sat_solver.cnf_formula import tseitins_transformation
from sat_solver.formula import Formula


def test_simple_tseitins_transformation():
    # Formula: (x1 & x2) || x3
    # Actual Result: tse4 & (tse4 \iff p_g1 || x3) & (tse3 \iff x1 & x2)
    # tse4 & (~tse4 || tse3 || x3) & (tse4 || ~tse3) & (tse4 || ~x3) &
    # & (tse3 || ~x1 || ~x2) & (~tse3 || x1) & (~tse3 || x2)
    x1 = Formula.create_leaf("x1")
    x2 = Formula.create_leaf("x2")
    x3 = Formula.create_leaf("x3")
    x1andx2 = Formula(x1, x2, Formula.Operator.AND)
    f = Formula(x1andx2, x3, Formula.Operator.OR)

    actual_cnf = tseitins_transformation(f)
    actual_cnf_set = [set(map(str, ls)) for ls in actual_cnf]
    expected_result = [['tse4'], ['~tse4', 'tse3', 'x3'], ['tse4', '~tse3'], ['tse4', '~x3'], ['tse3', '~x1', '~x2'],
                        ['~tse3', 'x1'], ['~tse3', 'x2']]

    # print(actual_cnf)
    # print(expected_result)
    assert len(expected_result) == len(actual_cnf)
    assert all([set(expected) in actual_cnf_set for expected in expected_result])

def test_simple_negate_tseitins_transformation():
    # Formula: (x1 & ~x2) || x3
    # Actual Result: tse4 & (tse4 \iff p_g1 || x3) & (tse3 \iff x1 & ~x2)
    # tse4 & (~tse4 || tse3 || x3) & (tse4 || ~tse3) & (tse4 || ~x3) &
    # & (tse3 || ~x1 || x2) & (~tse3 || x1) & (~tse3 || ~x2)
    x1 = Formula.create_leaf("x1")
    x2 = Formula.create_leaf("x2")
    x3 = Formula.create_leaf("x3")
    x1andx2 = Formula(x1, x2, Formula.Operator.AND)
    f = Formula(x1andx2, x3, Formula.Operator.OR)

    cnf_f = tseitins_transformation(f)
    print(cnf_f)


if __name__ == '__main__':
    test_simple_tseitins_transformation()
