from common.operator import Operator

unary_operator_pattern = Operator.NEGATION.value

operators = map(lambda op: '\|' if op == Operator.OR.value else op, [op.value for op in Operator])
binary_operator_pattern = '|'.join([op for op in operators if op != unary_operator_pattern])
variable_pattern = '(T|F|[a-z]+\\d*)'
sub_formula_or_variable_pattern = '(?P<{side}>\(.*\)|{variable}|{unary}.*)'
sub_formula_for_left_side = sub_formula_or_variable_pattern.format(
    side='left', variable=variable_pattern, unary=unary_operator_pattern)
sub_formula_for_right_side = sub_formula_or_variable_pattern.format(
    side='right', variable=variable_pattern, unary=unary_operator_pattern)
unary_formula_pattern = '(?P<op>{unary}){sub_formula}'.format(
    unary=unary_operator_pattern, sub_formula=sub_formula_for_left_side)
binary_formula_pattern = '\({left}(?P<op>{binary}){right}\)'.format(
    left=sub_formula_for_left_side, binary=binary_operator_pattern, right=sub_formula_for_right_side)
