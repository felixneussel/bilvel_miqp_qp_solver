
from gurobipy import GRB
from numpy import infty, array
from re import match
from timeit import default_timer
from Functional.problems import check_dimensions, getSParam, getX_IParamLazy, setup_master, setup_meta_data ,optimize,  add_cut, setup_meta_data, check_dimensions,getX_IParam, warmstart,add_lazy_constraint
from Functional.subproblem import setup_sub_rem_1,setup_sub_rem_2,setup_sub_as_fixed_nonconvex_reform,setup_sub_st_lazy
from Functional.feasibility import setup_feas_lazy,setup_feas_mt
from bisect import bisect
from operator import itemgetter
import concurrent.futures as futures
from Functional.utils import stop_process_pool, time_remaining

def solve(problem_data,tol,iteration_limit,time_limit,subproblem_mode,algorithm,big_M,optimized_binary_expansion):
    if algorithm == 'MT':
        return MT(problem_data,tol,iteration_limit,time_limit,subproblem_mode,False,False,False,big_M,optimized_binary_expansion)
    elif algorithm == 'MT-K':
        return MT(problem_data,tol,iteration_limit,time_limit,subproblem_mode,True,False,False,big_M,optimized_binary_expansion)
    elif algorithm == 'MT-K-F':
        return MT(problem_data,tol,iteration_limit,time_limit,subproblem_mode,True,True,False,big_M,optimized_binary_expansion)
    elif algorithm == 'MT-K-F-W':
        return MT(problem_data,tol,iteration_limit,time_limit,subproblem_mode,True,True,True,big_M,optimized_binary_expansion)
    elif algorithm == 'ST':
        return ST(problem_data,tol,time_limit,subproblem_mode,False,False,False,big_M,optimized_binary_expansion)
    elif algorithm == 'ST-K':
        return ST(problem_data,tol,time_limit,subproblem_mode,True,False,False,big_M,optimized_binary_expansion)
    elif algorithm == 'ST-K-C':
        return ST(problem_data,tol,time_limit,subproblem_mode,True,True,False,big_M,optimized_binary_expansion)
    elif algorithm == 'ST-K-C-S':
        return ST(problem_data,tol,time_limit,subproblem_mode,True,True,True,big_M,optimized_binary_expansion)
    else:
        raise ValueError(f"Algorithm {algorithm} is not a valid argument")

def solve_subproblem_regular(SETUP_SUB_FUNCTION,UB,solution,m_vars,problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter,start,time_limit):
    sub = SETUP_SUB_FUNCTION(problem_data,meta_data,getX_IParam(master))
    sub.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
    sub.setParam(GRB.Param.NumericFocus,3)
    sub_start = default_timer()
    s_status,s_vars,s_val = optimize(sub)
    time_in_sub = default_timer() - sub_start
    next_cut = s_vars
    if s_status in [GRB.OPTIMAL,GRB.SUBOPTIMAL,GRB.TIME_LIMIT]:#subproblem feasible           
        if s_val < UB:
            for v in s_vars:
                solution[v.varName] = v.x
            for v in m_vars:
                if match(r'x|s',v.varName) is not None:
                    solution[v.varName] = v.x
            UB = s_val
        if s_status == GRB.TIME_LIMIT:
            return array([]),solution,UB,True, time_in_sub
    else:#Subproblem infeasible
        feas = setup_feas_mt(problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter)
        feas.setParam(GRB.Param.NumericFocus,3)
        feas.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
        sub_start = default_timer()
        f_status,f_vars,f_obj = optimize(feas)
        time_in_sub = default_timer() - sub_start
        next_cut = f_vars
        if f_status == GRB.TIME_LIMIT:
            return array([]),solution,UB,True, time_in_sub
    #Add Linearization of Strong Duality Constraint at solution of sub or feasibility
    #problem as constraint to masterproblem
    cp = []
    for var in next_cut:
        if match(r'^y',var.varName) is not None:
            cp.append(var.x)
    return array(cp),solution, UB, False, time_in_sub

def solve_subproblem_remark_2(SETUP_SUB_FUNCTION,UB,solution,m_vars,problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter,start,time_limit):
    sub,y_solution = SETUP_SUB_FUNCTION(problem_data,meta_data,getX_IParam(master))
    sub.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
    sub.setParam(GRB.Param.NumericFocus,3)
    sub_start = default_timer()
    s_status,s_vars,s_val = optimize(sub)
    time_in_sub = default_timer() - sub_start
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
        if s_status == GRB.TIME_LIMIT:
            return array([cp]),solution,UB,True,time_in_sub
    else:#Subproblem infeasible
        feas = setup_feas_mt(problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter)
        feas.setParam(GRB.Param.NumericFocus,3)
        feas.setParam(GRB.Param.TimeLimit,max(time_limit - (default_timer()-start),0))
        sub_start = default_timer() 
        f_status,f_vars,f_obj = optimize(feas)
        time_in_sub = default_timer() - sub_start
        next_cut = f_vars
        if f_status == GRB.TIME_LIMIT:
            return array([]),solution,UB,True, time_in_sub
        #Add Linearization of Strong Duality Constraint at solution of sub or feasibility
        #problem as constraint to masterproblem
        cp = []
        for var in next_cut:
            if match(r'^y',var.varName) is not None:
                cp.append(var.x)
        cp = array(cp)
    return array(cp),solution, UB, False, time_in_sub


def solve_subproblem_regular_lazy(SETUP_SUB_FUNCTION,problem_data,master,meta_data,start,time_limit):
    sub = SETUP_SUB_FUNCTION(problem_data,meta_data,getX_IParamLazy(master))
    sub_start = default_timer()
    sub.setParam(GRB.Param.NumericFocus,3)
    sub.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
    s_status,s_vars,s_val = optimize(sub)
    time_in_sub = default_timer() - sub_start
    next_cut = s_vars
    if s_status not in [GRB.OPTIMAL,GRB.SUBOPTIMAL,GRB.TIME_LIMIT]:#subproblem infeasible           
        feas = setup_feas_lazy(problem_data,meta_data,getX_IParamLazy(master),getSParam(master))
        feas.setParam(GRB.Param.NumericFocus,3)
        feas.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
        sub_start = default_timer()
        f_status,f_vars,f_obj = optimize(feas)
        time_in_sub = default_timer() - sub_start
        next_cut = f_vars
    cp = []
    for var in next_cut:
        if match(r'^y',var.varName) is not None:
            cp.append(var.x)
    return array(cp),time_in_sub,s_val

def solve_subproblem_remark_2_lazy(SETUP_SUB_FUNCTION,problem_data,master,meta_data,start,time_limit):
    sub,y_solution = SETUP_SUB_FUNCTION(problem_data,meta_data,getX_IParamLazy(master))
    sub.setParam(GRB.Param.NumericFocus,3)
    sub.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
    sub_start = default_timer()
    s_status,s_vars,s_val = optimize(sub)
    time_in_sub = default_timer() - sub_start
    cp = y_solution
    if s_status not in [GRB.OPTIMAL,GRB.SUBOPTIMAL]:#subproblem infeasible           
        feas = setup_feas_lazy(problem_data,meta_data,getX_IParamLazy(master),getSParam(master))
        feas.setParam(GRB.Param.NumericFocus,3)
        feas.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
        sub_start = default_timer() 
        f_status,f_vars,f_obj = optimize(feas)
        time_in_sub = default_timer() - sub_start
        next_cut = f_vars
        #Add Linearization of Strong Duality Constraint at solution of sub or feasibility
        #problem as constraint to masterproblem
        cp = []
        for var in next_cut:
            if match(r'^y',var.varName) is not None:
                cp.append(var.x)
        cp = array(cp)
    return array(cp),time_in_sub,s_val

def MT(problem_data,tol,iteration_limit,time_limit,subproblem_mode,kelley_cuts,early_termination, use_warmstart,big_M,optimized_binary_expansion):
    check_dimensions(problem_data)
    start = default_timer()
    iteration_counter = 0
    cut_counter = 0
    LB = -infty
    UB = infty
    meta_data = setup_meta_data(problem_data,optimized_binary_expansion)
    master,y_var,dual_var,w_var = setup_master(problem_data,meta_data,big_M,optimized_binary_expansion)
    solution = {}
    TIME_LIMIT_EXCEEDED = False
    time_in_subs = 0
    num_of_subs = 0
    gap = infty
    if subproblem_mode == 'regular':
        SOLVE_SUB_FUNCTION = solve_subproblem_regular
        SETUP_SUB_FUNCTION = setup_sub_as_fixed_nonconvex_reform#setup_sub_mt
    elif subproblem_mode == 'remark_1':
        SOLVE_SUB_FUNCTION = solve_subproblem_regular
        SETUP_SUB_FUNCTION = setup_sub_rem_1
    elif subproblem_mode == 'remark_2':
        SOLVE_SUB_FUNCTION = solve_subproblem_remark_2
        SETUP_SUB_FUNCTION = setup_sub_rem_2
    else:
        raise ValueError('Keyword argument subproblem_mode must be "regular", "remark_1" or "remark_2"')
    termination = False
    while not termination:#LB + tol < UB and iteration_counter < iteration_limit and not TIME_LIMIT_EXCEEDED:
        #Solve Masterproblem
        if early_termination and iteration_counter > 0:
            master.setParam(GRB.Param.BestObjStop, UB-5*tol)
            master.setParam(GRB.Param.Cutoff, UB-tol)
        if use_warmstart:
            master = warmstart(master,solution)
        master.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
        master.setParam(GRB.Param.NumericFocus,3)
        master.setParam(GRB.Param.IntFeasTol,1e-9)
        m_status,m_vars,m_val = optimize(master)

        if m_status == GRB.CUTOFF and early_termination:
            termination = True
            continue
        
        if m_status not in [GRB.OPTIMAL,15,GRB.TIME_LIMIT]:
            return None,None,default_timer() - start,time_in_subs,num_of_subs, 4,gap
            #termination = True
        else:
            LB = m_val
        if m_status == GRB.TIME_LIMIT:
            TIME_LIMIT_EXCEEDED = True
            return solution,UB,default_timer() - start,time_in_subs,num_of_subs, GRB.TIME_LIMIT,gap
        cut_point,solution,UB, TIME_LIMIT_EXCEEDED,time_in_sub = SOLVE_SUB_FUNCTION(SETUP_SUB_FUNCTION,UB,solution,m_vars,problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter,start,time_limit)
        time_in_subs += time_in_sub
        num_of_subs += 1
        gap = UB - LB
        if TIME_LIMIT_EXCEEDED:
            continue
        
        master = add_cut(problem_data,master,meta_data,y_var,dual_var,w_var,cut_point,optimized_binary_expansion)
        cut_counter += 1
        if kelley_cuts:
            y_p = []
            for v in m_vars:
                if match(r'^y',v.varName):
                    y_p.append(v.x)
            y_p = array(y_p)
            master = add_cut(problem_data,master,meta_data,y_var,dual_var,w_var,y_p,optimized_binary_expansion)
            cut_counter += 1
        iteration_counter += 1

        if not early_termination and (LB + tol >= UB or iteration_counter >= iteration_limit or TIME_LIMIT_EXCEEDED):
            termination = True
    stop = default_timer()
    runtime = stop - start
    if TIME_LIMIT_EXCEEDED:
        return solution, UB, runtime,time_in_subs,num_of_subs, GRB.TIME_LIMIT,gap
    elif solution == {}:
        return None,None,default_timer() - start,time_in_subs,num_of_subs, 4,gap
    return solution,UB,runtime,time_in_subs,num_of_subs, 2,gap


def ST(problem_data,tol,time_limit,subproblem_mode,kelley_cuts,initial_cut,initial_ub,big_M,optimized_binary_expansion):
    start = default_timer()
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    UB = infty
    iteration_counter = 0
    cut_counter = 0
    solution = {}
    time_in_subs = []
    TIME_LIMIT_EXCEEDED = False
    meta_data = setup_meta_data(problem_data,optimized_binary_expansion)
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr,non_binary_index_sets = meta_data
    master,y_var,dual_var,w_var = setup_master(problem_data,meta_data,big_M,optimized_binary_expansion)
    if subproblem_mode == 'regular':
        SOLVE_SUB_FUNCTION = solve_subproblem_regular_lazy
        SETUP_SUB_FUNCTION = setup_sub_as_fixed_nonconvex_reform
    elif subproblem_mode == 'remark_1':
        SOLVE_SUB_FUNCTION = solve_subproblem_regular_lazy#solve_subproblem_remark_1_lazy
        SETUP_SUB_FUNCTION = setup_sub_rem_1
    elif subproblem_mode == 'remark_2':
        SOLVE_SUB_FUNCTION = solve_subproblem_remark_2_lazy
        SETUP_SUB_FUNCTION = setup_sub_rem_2
    else:
        raise ValueError('Keyword argument subproblem_mode must be "regular", "remark_1" or "remark_2"')
    if initial_cut or initial_ub:
        initial_solution, initial_incumbent, initial_time,initial_time_in_sub,initial_num_of_subs, initial_status,initial_gap = MT(problem_data,tol,1,time_limit,subproblem_mode,False,False,False,big_M,optimized_binary_expansion)
        if initial_status == GRB.TIME_LIMIT:
            return initial_solution,initial_incumbent,initial_time,initial_time_in_sub,initial_num_of_subs,initial_status,initial_gap
    if initial_cut:
        y_p = []
        for v in initial_solution:
            if match(r'^y',v):
                y_p.append(initial_solution[v])
        master = add_cut(problem_data,master,meta_data,y_var,dual_var,w_var,array(y_p),optimized_binary_expansion)
        cut_counter += 1
    if initial_ub:
        UB = initial_incumbent + 1e6*tol
        master.setParam(GRB.Param.Cutoff, UB)
        """ start_sol = {}
        for v in initial_solution:
            start_sol[v.varName] = v.x
        master = warmstart(master,start_sol) """
    
        
    master.setParam(GRB.Param.TimeLimit,max(time_limit - (default_timer()-start),0))
    master.setParam(GRB.Param.LazyConstraints,1)
    master.setParam(GRB.Param.NumericFocus,3)
    master.setParam(GRB.Param.IntFeasTol,1e-9)
    master._G_l = G_l
    master._d_l = d_l
    master._b = b
    master._y = y_var
    master._dual = dual_var
    master._w = w_var
    master._bin_coeff = bin_coeff_arr
    master._SOLVE_SUB_FUNCTION = SOLVE_SUB_FUNCTION
    master._SETUP_SUB_FUNCTION = SETUP_SUB_FUNCTION
    master._problem_data = problem_data
    master._meta_data = meta_data
    master._time_in_subs = 0
    master._num_of_subs = 0
    master._vars = master.getVars()
    master._kelley = kelley_cuts
    master._incumbent = infty
    master._tol = tol
    master._start = start
    master._time_limit = time_limit
    master._optimized_binary_expansion = optimized_binary_expansion
    master.optimize(newCut)

    solution = {}
    for v in master.getVars():
        try:
            solution[v.varName] = v.x
        except AttributeError:
            break
    try:
        obj = master.ObjVal
    except AttributeError:
        obj = infty
    try:
        gap = master.MIPGap
    except AttributeError:
        gap = infty
    runtime = default_timer() - start
    status = master.status
    return solution,obj,runtime,master._time_in_subs,master._num_of_subs,status, gap
    

def newCut(model,where):
    if where == GRB.Callback.MIPSOL:
        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        incumbent = model._incumbent
        if current_obj < incumbent:#Best Objective is updated immediately, so if current_obj is better than imcumbent, it is best objective
            cut_point, time_in_sub,sub_obj = model._SOLVE_SUB_FUNCTION(model._SETUP_SUB_FUNCTION,model._problem_data,model,model._meta_data,model._start,model._time_limit)
            model._time_in_subs += time_in_sub
            model._num_of_subs += 1
            model = add_lazy_constraint(cut_point,model)
            if sub_obj < incumbent:
                model._incumbent = sub_obj
        else:#integer feasible non improving -> potential addidtional kelley cut
            if model._kelley:
                cut_point = array(model.cbGetSolution(model._y))
                model = add_lazy_constraint(cut_point,model)
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

""" def ST_old(problem_data,tol,time_limit,subproblem_mode,kelley_cuts,initial_cut,initial_ub):
    start = default_timer()
    UB = infty
    iteration_counter = 0
    cut_counter = 0
    solution = {}
    time_in_subs = []
    TIME_LIMIT_EXCEEDED = False
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
        initial_solution, initial_incumbent, initial_time,intial_time_in_sub, inital_status = MT(problem_data,tol,1,subproblem_mode,False,False,False)
    if initial_cut:
        y_p = []
        for v in initial_solution:
            if match(r'^y',v):
                y_p.append(initial_solution[v])
        O[0] = (add_cut(problem_data,O[0][0],meta_data,y_var,dual_var,w_var,array(y_p)),O[0][1])
        cut_counter += 1
    if initial_ub:
        UB = initial_incumbent + 2*tol
    while O and not TIME_LIMIT_EXCEEDED:
        N_p,N_p_ub = O.pop()
        N_p.setParam(GRB.Param.TimeLimit,max(time_limit - (default_timer()-start),0))
        N_p.setParam(GRB.Param.DualReductions,1)
        m_status,m_vars,m_val = optimize(N_p)
        if m_status == GRB.TIME_LIMIT:
            TIME_LIMIT_EXCEEDED = True
            continue
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
            cut_point,solution,UB, TIME_LIMIT_EXCEEDED,time_in_sub = SOLVE_SUB_FUNCTION(SETUP_SUB_FUNCTION,UB,solution,m_vars,problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter,start,time_limit)
            time_in_subs.append(time_in_sub)
            if TIME_LIMIT_EXCEEDED:
                continue
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
    if solution != {} and not TIME_LIMIT_EXCEEDED:
        return solution, UB, runtime,time_in_subs,2
    elif TIME_LIMIT_EXCEEDED:
        return solution, UB, runtime,time_in_subs, GRB.TIME_LIMIT
    else:
        return None,None,runtime,time_in_subs, 4 """

""" def solve_subproblem_remark_1(SETUP_SUB_FUNCTION,UB,solution,m_vars,problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter,start,time_limit):
    sub = SETUP_SUB_FUNCTION(problem_data,meta_data,getX_IParam(master))
    sub.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
    sub.setParam(GRB.Param.NumericFocus,3)
    sub.setParam(GRB.Param.FeasibilityTol,1e-9)
    sub_start = default_timer()
    s_status,s_vars,s_val = optimize(sub)
    time_in_sub = default_timer() - sub_start
    next_cut = s_vars
    if s_status == GRB.OPTIMAL or s_status == GRB.SUBOPTIMAL:#subproblem feasible           
        if s_val < UB:
            for v in s_vars:
                solution[v.varName] = v.x
            for v in m_vars:
                if match(r'x|s',v.varName) is not None:
                    solution[v.varName] = v.x
            UB = s_val
        if s_status == GRB.TIME_LIMIT:
            return array([]),solution,UB,True, time_in_sub
    else:#Subproblem infeasible
        feas = setup_feas_mt(problem_data,master,meta_data,y_var,dual_var,w_var,iteration_counter)
        feas.setParam(GRB.Param.NumericFocus,3)
        feas.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
        sub_start = default_timer()
        f_status,f_vars,f_obj = optimize(feas)
        time_in_sub = default_timer()- sub_start
        next_cut = f_vars
        if f_status == GRB.TIME_LIMIT:
            return array([]),solution,UB,True,time_in_sub
    #Add Linearization of Strong Duality Constraint at solution of sub or feasibility
    #problem as constraint to masterproblem
    cp = []
    for var in next_cut:
        if match(r'^y',var.varName) is not None:
            cp.append(var.x)
    return array(cp),solution, UB, False,time_in_sub """

""" def solve_subproblem_remark_1_lazy(SETUP_SUB_FUNCTION,problem_data,master,meta_data,start,time_limit):
    sub = SETUP_SUB_FUNCTION(problem_data,meta_data,getX_IParamLazy(master))
    sub_start = default_timer()
    sub.setParam(GRB.Param.NumericFocus,3)
    sub.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
    s_status,s_vars,s_val = optimize(sub)
    time_in_sub = default_timer() - sub_start
    next_cut = s_vars
    if s_status not in [GRB.OPTIMAL,GRB.SUBOPTIMAL,GRB.TIME_LIMIT]:#subproblem infeasible           
        feas = setup_feas_lazy(problem_data,meta_data,getX_IParamLazy(master),getSParam(master))
        feas.setParam(GRB.Param.NumericFocus,3)
        feas.setParam(GRB.Param.TimeLimit,time_remaining(start,time_limit))
        sub_start = default_timer()
        f_status,f_vars,f_obj = optimize(feas)
        time_in_sub = default_timer()- sub_start
        next_cut = f_vars
    cp = []
    for var in next_cut:
        if match(r'^y',var.varName) is not None:
            cp.append(var.x)
    return array(cp),time_in_sub,s_val """