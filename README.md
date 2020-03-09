# Automated-Reasoning-Project

Implementation of a basic SMT/SAT solver based on DPLL & CDCL  
    SMT supports UF theory and LP  
    LP Uses revised simplex algorithm   


## Running Instructions:
Each model has a folder, and a solve / search function.  
There is a unified main that can be used to execute any of the models, the running arguments are as follows:  

SAT:  

    python3 main.py sat '(x1&x2)'  
    
SMT: 

    python3 main.py smt '(f(a)=b&f(b)=a)'  
    
LP: 

    python3 main.py lp '0x0,2x1,3x2,x3<=5' '4x1,x2,2x3<=11' '3x1,4x2,2x3<=8' '5x1,4x2,3x3'  
    

## Tests 
Tests are based on pytest (run pytest in main folder to execute all of them)  
Each model has its own tests, most of them are unit tests and one special file which is called test_random_MODEL
The random tests samples different parameters (number of variables, number of equations etc...), and runs ensures the result of our model and z3 (sat & smt) or Gurobi (LP) are the same


