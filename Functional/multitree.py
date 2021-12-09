
from gurobipy import GRB
from numpy import infty, array
from re import match
from timeit import default_timer
from Functional.problems import check_dimensions, setup_master, setup_meta_data, setup_sub,optimize, setup_feas, add_cut, setup_meta_data, check_dimensions




def MT(problem_data,tol):
    check_dimensions(problem_data)
    start = default_timer()
    iteration_counter = 0
    LB = -infty
    UB = infty
    meta_data = setup_meta_data(problem_data)
    master,y_var,dual_var,w_var = setup_master(problem_data,meta_data)
    solution = {}
    while LB + tol < UB:
        #Solve Masterproblem
        m_status,m_vars,m_val = optimize(master)
        
        if m_status != GRB.OPTIMAL:
            return ('The bilevel problem is infeasible')
        else:
            LB = m_val
        
        #Solve Subproblem
        sub = setup_sub(problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter)
        s_status,s_vars,s_val = optimize(sub)
        next_cut = s_vars
        if s_status == GRB.OPTIMAL or s_status == GRB.SUBOPTIMAL:#subproblem feasible           
            if s_val < UB:
                for v in s_vars:
                    solution[v.varName] = v.x
                for v in m_vars:
                    if match(r'x|s',v.varName) is not None:
                        solution[v.varName] = v.x
                UB = s_val
        else:#Subproblem infeasible
            feas = setup_feas(problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter)
            f_vars = optimize(feas)[1]
            next_cut = f_vars

        #Add Linearization of Strong Duality Constraint at solution of sub or feasibility
        #problem as constraint to masterproblem
        cp = []
        for var in next_cut:
            if match(r'^y',var.varName) is not None:
                cp.append(var.x)
        
        add_cut(problem_data,master,meta_data,y_var,dual_var,w_var,array(cp))
        iteration_counter += 1
    stop = default_timer()
    runtime = stop - start
    return solution,UB,runtime