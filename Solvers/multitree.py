#
#This file contains the solvers for all multi-tree and single-tree based solution approaches.
#
from gurobipy import GRB
from numpy import infty, array
from re import match
from timeit import default_timer
from Solvers.problems import check_dimensions, getSParam, getX_IParamLazy, setup_master, setup_meta_data ,optimize,  add_cut, setup_meta_data, check_dimensions,getX_IParam, warmstart,add_lazy_constraint
from Solvers.subproblem import setup_sub_rem_1,setup_sub_rem_2,setup_sub_as_fixed_nonconvex_reform,setup_sub_st_lazy
from Solvers.feasibility import setup_feas_lazy,setup_feas_mt
from bisect import bisect
from operator import itemgetter
import concurrent.futures as futures
from Solvers.utils import stop_process_pool, time_remaining

def solve(problem_data,tol,iteration_limit,time_limit,subproblem_mode,algorithm,big_M,optimized_binary_expansion):
    """
    Solves an MIQP-QP with outer approximation multi-tree or single-tree solution approach from the paper
    'Outer approximation techniques for the global opimization of mixed-integer quadratic bilevel problems'
    by Kleinert, Grimm and Schmidt.

    # Parameters 

    - problem_data : List of the problem specific dimensions, vectors and matrices 
    n_I, n_R, n_y, m_u, m_l, H_u, G_u, G_l, c_u, d_u, d_l, A, B, a, x^-, x^+, C, D, b as specified in 
    'Outer approximation techniques for the global opimization of mixed-integer quadratic bilevel problems'
    by Kleinert, Grimm and Schmidt.
    - tol : Optimality tolerance, should not be smaller than 1e-4.
    - iteration_limit : Maximal number of iterations.
    - time_limit : Time limit in seconds.
    - subproblem_mode : Method to aquire subproblem solutions. Can be set to 'regular', 'remark_1' and 'remark_2'. 
    It is recommended to use 'remark_2' if the matrix G_l is strictly positive definite or else 'regular'.
    - algorithm : The solution approach that should be used. Can be set to 'MT', 'MT-K', 'MT-K-F', 'MT-K-F-W',
    'ST', 'ST-K', 'ST-K-C', 'ST-K-C-S'. It is recommended to use 'ST' for small problem instances.
    - big_M : big_M value for upper and lower bounds in the binary-expansion-related constraints. It is recommended
    to derive it from problem specific knowledge. 1e5 was used in a numerical study.
    - optimized_binary_expansion : True if an enhanced version of binary expansion which allows for negative integer 
    variables and reduces the number of additional variables should be used or else False. True is recommended.

    # Returns

    Tuple containing
    - Dictionary with optimal values for variables.
    - The optimal objective value.
    - Runtime.
    - Time spent for solving subproblems or feasibiltiy problems.
    - Number of solved subproblems and feasibility problems.
    - Optimization status code (see Gurobi Status Codes).
    - Gap between upper bound and lower bound (not for 'MT-K-F' and 'MT-K-F-W').
    """
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
        f_status,f_vars,_ = optimize(feas)
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
        f_status,f_vars,_ = optimize(feas)
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
        _,f_vars,_ = optimize(feas)
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
    _,_,_,_,_,_,_,G_l,_,_,d_l,_,_,_,_,_,_,_,b = problem_data
    UB = infty
    cut_counter = 0
    solution = {}
    meta_data = setup_meta_data(problem_data,optimized_binary_expansion)
    _,_,_,_,_,_,bin_coeff_arr,_ = meta_data
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
        if initial_status != GRB.OPTIMAL:
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
        master = warmstart(master,initial_solution)
    
        
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
    master._kelley_cut_points = []
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
    """
    Callback function for Single-Tree solution approach. Adds an outer approximation cut of the strong duality constraint
    when a new integer solution which improves upon the current best solution.
    """
    if where == GRB.Callback.MIPSOL:
        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        incumbent = model._incumbent
        if model._kelley:
            model._kelley_cut_points.append(array(model.cbGetSolution(model._y)))
        if current_obj < incumbent:#Best Objective is updated immediately, so if current_obj is better than imcumbent, it is best objective
            cut_point, time_in_sub,sub_obj = model._SOLVE_SUB_FUNCTION(model._SETUP_SUB_FUNCTION,model._problem_data,model,model._meta_data,model._start,model._time_limit)
            model._time_in_subs += time_in_sub
            model._num_of_subs += 1
            model = add_lazy_constraint(cut_point,model)
            if sub_obj < incumbent:
                model._incumbent = sub_obj
            kelley_points = model._kelley_cut_points
            for point in kelley_points:
                model = add_lazy_constraint(point,model)
            model._kelley_cut_points = []
