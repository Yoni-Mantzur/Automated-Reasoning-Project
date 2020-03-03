class UnboundedException(Exception):
    def __init__(self, *args):
        pass

    def __str__(self):
        return 'LP Problem is unbounded'

