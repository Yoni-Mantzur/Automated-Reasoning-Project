from sat_solver.patterns import variable_pattern, unary_operator_pattern, binary_operator_pattern

function_pattern = '(?P<function_name>[a-z]+\d*)\((?P<inputs>(.*))\)'
term_pattern = '((?P<function>{function_pattern})|(?P<variable>{variable_pattern}))'.format(
    function_pattern=function_pattern, variable_pattern=variable_pattern)
equation_pattern = '(?P<lhs>{term_pattern})=(?P<rhs>{term_pattern})'

unary_formula_pattern = '(?P<op>{unary})(?P<left>.+)'.format(unary=unary_operator_pattern)
binary_formula_pattern = '\\((?P<left>[.*=.*]+)(?P<op>{binary})(?P<right>[.*=.*]+)$\\)'.format(
    binary=binary_operator_pattern)
