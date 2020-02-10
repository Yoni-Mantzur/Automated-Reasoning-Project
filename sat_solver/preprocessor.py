from collections import defaultdict

from sat_solver.cnf_formula import CnfFormula


def remove_redundant_literals(formula: CnfFormula) -> CnfFormula:
    clauses = []
    literal_to_clauses = defaultdict(set)

    for clause_number, clause in enumerate(formula.clauses):
        new_clause = []

        for literal in clause:
            if clause_number not in literal_to_clauses[literal]:
                new_clause.append(literal)
                literal_to_clauses[literal].add(clause_number)

        clauses.append(new_clause)

    return CnfFormula(clauses, literal_to_clauses)


def delete_trivial_clauses(formula: CnfFormula) -> CnfFormula:
    removed_clauses = set()
    new_clauses = []

    if not formula.literal_to_clauses:
        raise Exception("This function should be called after remove_redundant_literals function")

    for clause_number, clause in enumerate(formula.clauses):

        # Assuming no literal redundancy - meaning, this function should be called after `remove_redundant_literals`
        variables = map(lambda literal: literal.idx, clause)
        if len(set(variables)) == len(clause):
            new_clauses.append(clause)
        else:
            removed_clauses.add(clause_number)

    # update literal_to_clauses
    literal_to_clauses = formula.literal_to_clauses
    for literal in literal_to_clauses.keys():
        literal_to_clauses[literal] -= removed_clauses

    return CnfFormula(new_clauses, literal_to_clauses)


def preprocess(formula: CnfFormula) -> CnfFormula:
    formula = remove_redundant_literals(formula)
    formula = delete_trivial_clauses(formula)
    return formula
