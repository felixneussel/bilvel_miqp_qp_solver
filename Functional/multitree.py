
from gurobipy import GRB
from numpy import infty, array
from re import match
from timeit import default_timer
from Functional.problems import check_dimensions, setup_master, setup_meta_data, setup_sub_mt, setup_sub_mt_rem_2, setup_sub_st,setup_sub_mt_rem_1 ,optimize, setup_feas_mt, setup_feas_st, add_cut, setup_meta_data, check_dimensions,branch, setup_st_master, is_int_feasible, get_int_vars, getX_IParam

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
            return None,'None',default_timer() - start, 4
        else:
            LB = m_val
        
        #Solve Subproblem
        sub = setup_sub_mt(problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter)
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
            feas = setup_feas_mt(problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter)
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
    return solution,UB,runtime, 2

def MT_rem_1(problem_data,tol):
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
            return None,'None',default_timer() - start, 4
        else:
            LB = m_val
        
        #Solve Subproblem
        sub = setup_sub_mt_rem_1(problem_data,meta_data,getX_IParam(master))
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
            feas = setup_feas_mt(problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter)
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
    return solution,UB,runtime, 2   

def MT_rem_2(problem_data,tol):
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
            return None,'None',default_timer() - start, 4
        else:
            LB = m_val
        
        #Solve Subproblem
        sub,y_solution = setup_sub_mt_rem_2(problem_data,meta_data,getX_IParam(master))
        s_status,s_vars,s_val = optimize(sub)
        cp = y_solution
        if s_status == GRB.OPTIMAL or s_status == GRB.SUBOPTIMAL:#subproblem feasible           
            if s_val < UB:
                for v in s_vars:
                    solution[v.varName] = v.x
                for v in m_vars:
                    if match(r'x|s',v.varName) is not None:
                        solution[v.varName] = v.x
                for i,v in enumerate(y_solution):
                    solution[f"y[{i}]"] = v
                UB = s_val
        else:#Subproblem infeasible
            feas = setup_feas_mt(problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter)
            f_vars = optimize(feas)[1]
            next_cut = f_vars

            #Add Linearization of Strong Duality Constraint at solution of sub or feasibility
            #problem as constraint to masterproblem
            cp = []
            for var in next_cut:
                if match(r'^y',var.varName) is not None:
                    cp.append(var.x)
            cp = array(cp)
        
        add_cut(problem_data,master,meta_data,y_var,dual_var,w_var,cp)
        iteration_counter += 1
    stop = default_timer()
    runtime = stop - start
    return solution,UB,runtime, 2

def ST(problem_data,tol):
    start = default_timer()
    UB = infty
    iteration_counter = 0
    cut_counter = 0
    solution = {}
    z_star = None
    meta_data = setup_meta_data(problem_data)
    master,y_var,dual_var,w_var = setup_st_master(problem_data,meta_data)
    O = [master]
    while O:# and iteration_counter: #<8:
        N_p = O.pop()
        m_status,m_vars,m_val = optimize(N_p)
        int_vars = get_int_vars(m_vars)
        if m_status != GRB.OPTIMAL or m_val >= UB - tol:
            continue
        elif is_int_feasible(int_vars) and m_val < UB:
            #Solve Subproblem
            sub = setup_sub_st(problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter)
            s_status,s_vars,s_val = optimize(sub)
            next_cut = s_vars
            if s_status == GRB.OPTIMAL or s_status == GRB.SUBOPTIMAL:
                if s_val < UB:#subproblem feasible
                    for v in s_vars:
                        solution[v.varName] = v.x
                    for v in m_vars:
                        if match(r'x|s',v.varName) is not None:
                            solution[v.varName] = v.x
                    UB = s_val
            else:#Subproblem infeasible
                feas = setup_feas_st(problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter)
                f_vars = optimize(feas)[1]
                next_cut = f_vars
            O.append(N_p)
            cp = []
            for var in next_cut:
                if match(r'^y',var.varName) is not None:
                    cp.append(var.x)
            
            for pro in O:
                add_cut(problem_data,pro,meta_data,y_var,dual_var,w_var,array(cp))
            cut_counter += 1

        else:
            first,second = branch(N_p,int_vars,problem_data)
            O.append(first)
            O.append(second)
        iteration_counter += 1
    stop = default_timer()
    runtime = stop-start
    if solution != {}:
        return solution, UB, runtime,2
    else:
        return None,None,runtime,4

def ST_rem_1(problem_data,tol):
    start = default_timer()
    UB = infty
    iteration_counter = 0
    cut_counter = 0
    solution = {}
    z_star = None
    meta_data = setup_meta_data(problem_data)
    master,y_var,dual_var,w_var = setup_st_master(problem_data,meta_data)
    O = [master]
    while O:# and iteration_counter: #<8:
        N_p = O.pop()
        m_status,m_vars,m_val = optimize(N_p)
        int_vars = get_int_vars(m_vars)
        if m_status != GRB.OPTIMAL or m_val >= UB - tol:
            continue
        elif is_int_feasible(int_vars) and m_val < UB:
            #Solve Subproblem
            sub = setup_sub_mt_rem_1(problem_data,meta_data,getX_IParam(master))
            s_status,s_vars,s_val = optimize(sub)
            next_cut = s_vars
            if s_status == GRB.OPTIMAL or s_status == GRB.SUBOPTIMAL:
                if s_val < UB:#subproblem feasible
                    for v in s_vars:
                        solution[v.varName] = v.x
                    for v in m_vars:
                        if match(r'x|s',v.varName) is not None:
                            solution[v.varName] = v.x
                    UB = s_val
            else:#Subproblem infeasible
                feas = setup_feas_st(problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter)
                f_vars = optimize(feas)[1]
                next_cut = f_vars
            O.append(N_p)
            cp = []
            for var in next_cut:
                if match(r'^y',var.varName) is not None:
                    cp.append(var.x)
            
            for pro in O:
                add_cut(problem_data,pro,meta_data,y_var,dual_var,w_var,array(cp))
            cut_counter += 1

        else:
            first,second = branch(N_p,int_vars,problem_data)
            O.append(first)
            O.append(second)
        iteration_counter += 1
    stop = default_timer()
    runtime = stop-start
    if solution != {}:
        return solution, UB, runtime,2
    else:
        return None,None,runtime,4

def ST_rem_2(problem_data,tol):
    start = default_timer()
    UB = infty
    iteration_counter = 0
    cut_counter = 0
    solution = {}
    z_star = None
    meta_data = setup_meta_data(problem_data)
    master,y_var,dual_var,w_var = setup_st_master(problem_data,meta_data)
    O = [master]
    while O:# and iteration_counter: #<8:
        N_p = O.pop()
        m_status,m_vars,m_val = optimize(N_p)
        int_vars = get_int_vars(m_vars)
        if m_status != GRB.OPTIMAL or m_val >= UB - tol:
            continue
        elif is_int_feasible(int_vars) and m_val < UB:
            #Solve Subproblem
            sub,y_solution = setup_sub_mt_rem_2(problem_data,meta_data,getX_IParam(master))
            s_status,s_vars,s_val = optimize(sub)
            cp = y_solution
            if s_status == GRB.OPTIMAL or s_status == GRB.SUBOPTIMAL:
                if s_val < UB:#subproblem feasible
                    for v in s_vars:
                        solution[v.varName] = v.x
                    for v in m_vars:
                        if match(r'x|s',v.varName) is not None:
                            solution[v.varName] = v.x
                    for i,v in enumerate(y_solution):
                        solution[f"y[{i}]"] = v
                    UB = s_val
            else:#Subproblem infeasible
                feas = setup_feas_st(problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter)
                f_vars = optimize(feas)[1]
                next_cut = f_vars
                cp = []
                for var in next_cut:
                    if match(r'^y',var.varName) is not None:
                        cp.append(var.x)
                cp = array(cp)
            O.append(N_p)
            
            for pro in O:
                add_cut(problem_data,pro,meta_data,y_var,dual_var,w_var,cp)
            cut_counter += 1

        else:
            first,second = branch(N_p,int_vars,problem_data)
            O.append(first)
            O.append(second)
        iteration_counter += 1
    stop = default_timer()
    runtime = stop-start
    if solution != {}:
        return solution, UB, runtime,2
    else:
        return None,None,runtime,4