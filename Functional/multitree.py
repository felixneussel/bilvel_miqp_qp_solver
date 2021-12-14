
from gurobipy import GRB
from numpy import infty, array
from re import match
from timeit import default_timer
from Functional.problems import check_dimensions, setup_master, setup_meta_data, setup_sub_mt, setup_sub_rem_2, setup_sub_st,setup_sub_rem_1 ,optimize, setup_feas_mt, setup_feas_st, add_cut, setup_meta_data, check_dimensions,branch, setup_st_master, is_int_feasible, get_int_vars, getX_IParam, warmstart
from bisect import bisect
from operator import itemgetter

def solve_subproblem_regular(SETUP_SUB_FUNCTION,UB,solution,m_vars,problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter):
    sub = SETUP_SUB_FUNCTION(problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter)
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
    return array(cp),solution, UB

def solve_subproblem_remark_1(SETUP_SUB_FUNCTION,UB,solution,m_vars,problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter):
    sub = SETUP_SUB_FUNCTION(problem_data,meta_data,getX_IParam(master))
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
    return array(cp),solution, UB

def solve_subproblem_remark_2(SETUP_SUB_FUNCTION,UB,solution,m_vars,problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter):
    sub,y_solution = SETUP_SUB_FUNCTION(problem_data,meta_data,getX_IParam(master))
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
    return array(cp),solution, UB

def MT(problem_data,tol,iteration_limit,subproblem_mode,kelley_cuts,early_termination, use_warmstart):
    check_dimensions(problem_data)
    start = default_timer()
    iteration_counter = 0
    cut_counter = 0
    LB = -infty
    UB = infty
    meta_data = setup_meta_data(problem_data)
    master,y_var,dual_var,w_var = setup_master(problem_data,meta_data)
    solution = {}
  
    if subproblem_mode == 'regular':
        SOLVE_SUB_FUNCTION = solve_subproblem_regular
        SETUP_SUB_FUNCTION = setup_sub_mt
    elif subproblem_mode == 'remark_1':
        SOLVE_SUB_FUNCTION = solve_subproblem_remark_1
        SETUP_SUB_FUNCTION = setup_sub_rem_1
    elif subproblem_mode == 'remark_2':
        SOLVE_SUB_FUNCTION = solve_subproblem_remark_2
        SETUP_SUB_FUNCTION = setup_sub_rem_2
    else:
        raise ValueError('Keyword argument subproblem_mode must be "regular", "remark_1" or "remark_2"')
    while LB + tol < UB and iteration_counter < iteration_limit:
        #Solve Masterproblem
        if early_termination and iteration_counter > 0:
            master.setParam(GRB.Param.BestObjStop, UB - 2*tol)
        if use_warmstart:
            master = warmstart(master,solution)
        m_status,m_vars,m_val = optimize(master)
        
        if m_status not in [GRB.OPTIMAL,15]:
            return None,None,default_timer() - start, 4
        else:
            LB = m_val
        
        cut_point,solution,UB = SOLVE_SUB_FUNCTION(SETUP_SUB_FUNCTION,UB,solution,m_vars,problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter)
        
        master = add_cut(problem_data,master,meta_data,y_var,dual_var,w_var,cut_point)
        cut_counter += 1
        if kelley_cuts:
            y_p = []
            for v in m_vars:
                if match(r'^y',v.varName):
                    y_p.append(v.x)
            y_p = array(y_p)
            master = add_cut(problem_data,master,meta_data,y_var,dual_var,w_var,y_p)
            cut_counter += 1
        iteration_counter += 1
    stop = default_timer()
    runtime = stop - start
    return solution,UB,runtime, 2
""" 
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
 """

def ST(problem_data,tol,subproblem_mode,kelley_cuts,initial_cut,initial_ub):
    start = default_timer()
    UB = infty
    iteration_counter = 0
    cut_counter = 0
    solution = {}
    meta_data = setup_meta_data(problem_data)
    master,y_var,dual_var,w_var = setup_st_master(problem_data,meta_data)
    O = [(master,UB)]
    if subproblem_mode == 'regular':
        SOLVE_SUB_FUNCTION = solve_subproblem_regular
        SETUP_SUB_FUNCTION = setup_sub_st
    elif subproblem_mode == 'remark_1':
        SOLVE_SUB_FUNCTION = solve_subproblem_remark_1
        SETUP_SUB_FUNCTION = setup_sub_rem_1
    elif subproblem_mode == 'remark_2':
        SOLVE_SUB_FUNCTION = solve_subproblem_remark_2
        SETUP_SUB_FUNCTION = setup_sub_rem_2
    else:
        raise ValueError('Keyword argument subproblem_mode must be "regular", "remark_1" or "remark_2"')
    if kelley_cuts:
        non_improving_ints = []
    if initial_cut or initial_ub:
        initial_solution, initial_incumbent, initial_time, inital_status = MT(problem_data,tol,1,subproblem_mode,False,False,False)
    if initial_cut:
        y_p = []
        for v in initial_solution:
            if match(r'^y',v):
                y_p.append(initial_solution[v])
        O[0] = (add_cut(problem_data,O[0][0],meta_data,y_var,dual_var,w_var,array(y_p)),O[0][1])
        cut_counter += 1
    if initial_ub:
        UB = initial_incumbent + 2*tol
    while O:# and iteration_counter: #<8:
        N_p,N_p_ub = O.pop()
        m_status,m_vars,m_val = optimize(N_p)
        int_vars = get_int_vars(m_vars)
        if kelley_cuts and is_int_feasible(int_vars) and m_val >= UB:
            y_p = []
            for v in m_vars:
                if match(r'^y',v.varName):
                    y_p.append(v.x)
            non_improving_ints.append(array(y_p))
        if m_status != GRB.OPTIMAL or m_val >= UB - tol:
            continue
        elif is_int_feasible(int_vars) and m_val < UB:
            cut_point,solution,UB = SOLVE_SUB_FUNCTION(SETUP_SUB_FUNCTION,UB,solution,m_vars,problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter)
            O.append((N_p,UB))
            O = sorted(O,key=itemgetter(1),reverse=True)
            for pro in O:
                pro = (add_cut(problem_data,pro[0],meta_data,y_var,dual_var,w_var,cut_point),pro[1])
            cut_counter += 1
            if kelley_cuts:
                while non_improving_ints:
                    for pro in O:
                        pro = (add_cut(problem_data,pro[0],meta_data,y_var,dual_var,w_var,non_improving_ints.pop()),pro[1])
                    cut_counter += 1

        else:
            first,second = branch(N_p,int_vars,problem_data)
            O.append((first,UB))
            O.append((second,UB))
            O = sorted(O,key=itemgetter(1),reverse=True)
            #bisect(O,first,key=itemgetter(1))
            #bisect(O,second,key=itemgetter(1))
        iteration_counter += 1
    stop = default_timer()
    runtime = stop-start
    if solution != {}:
        return solution, UB, runtime,2
    else:
        return None,None,runtime,4
""" 
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
            sub = setup_sub_rem_1(problem_data,meta_data,getX_IParam(master))
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
            sub,y_solution = setup_sub_rem_2(problem_data,meta_data,getX_IParam(master))
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
"""  