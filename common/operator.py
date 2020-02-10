from enum import Enum


class Operator(Enum):
    IMPLIES = '->'
    IFF = '<->'
    OR = '|'
    AND = '&'
    NEGATION = '~'
