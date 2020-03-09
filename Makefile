.PHONY: sat smt lp test test-all

all:  sat smt lp

sat: 
	python3 main.py sat '(x1&x2)'
smt: 
	python3 main.py smt '(f(x1)=x0&f(x1)=f(x0))'
lp: 
	python3 main.py lp 'x1,x2<=3' 'x1<=3' 'x1,x2'

test:
	pytest-3 -m "not (gurobi or z3)"

test-all:
	pytest-3
