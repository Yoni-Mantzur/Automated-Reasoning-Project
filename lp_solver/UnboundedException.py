class UnboundedException(Exception):
    def __str__(self):
        return 'LP Problem is unbounded'


class InfeasibleException(Exception):
    def __str__(self):
        return 'No feasible solution'
