#
#This file contains functions that are related to setting up the master problem for the Gurobi solver 
#and adding outer approximation cuts. It also contains some functions that are required by the solver generally.
#
from gurobipy import Model,GRB, tuplelist, quicksum
from numpy import ones,log2,floor,ceil, concatenate, array, infty, zeros_like
from scipy.linalg import block_diag
from re import match,compile, sub
from operator import itemgetter

def calc_r_bar(l,u):
    """
    Calculates the number of additional binary variables for the binary expansion
    for each upper-level integer variable if the optimized binary expansion is used.
    """
    if l >= 0 and u >= 0:
        return int(floor(log2(u - l)) + 1)
    elif l < 0 and u > 0:
        return int(floor(log2(abs(l) + u)) + 1)
    elif l <= 0 and u <= 0:
        return int(floor(log2(abs(l) - abs(u))) + 1)
    else:
        raise ValueError(f"Unexpected value for bounds. l = {l} , u = {u}")

def setup_meta_data(problem_data,optimized_binary_expansion):
    """
    Generates index sets and coefficients for the binary expansion.
    """
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    if optimized_binary_expansion:
        r_bar = list(map(calc_r_bar,int_lb,int_ub))
    else:
        r_bar = (floor(log2(int_ub)) + ones(int_ub.shape)).astype(int)
    jr = tuplelist([(a,b) for a in range(0,n_I) for b in range(0, r_bar[a])])
    non_binary_index_set = tuplelist([(a,b) for a in range(0,n_I) if int_lb[a] != 0 or int_ub[a] != 1 for b in range(0,r_bar[a])])
    I = tuplelist([a for a in range(0,n_I)])
    R = tuplelist([a for a in range(0,n_R)])
    J = tuplelist([a for a in range(0,n_y)])
    ll_constr = tuplelist([a for a in range(0,m_l)])
    bin_coeff_dict = getBinaryCoeffsDict(jr)
    bin_coeff_arr = getBinaryCoeffsArray(jr)
    return jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr, non_binary_index_set

def setup_master(problem_data,meta_data,big_M,optimized_binary_expansion):
    """
    Creates a Gurobi model of the master problem.
    """
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    model = Model('Masterproblem')
    model.Params.LogToConsole = 0
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr,non_binary_index_set = meta_data
    x_I = model.addVars(I, vtype=GRB.INTEGER,lb=int_lb, ub=int_ub,name='x_I')
    return mp_common(problem_data,meta_data,model,x_I,big_M,optimized_binary_expansion)

def mp_common(problem_data,meta_data,model,x_I,big_M,optimized_binary_expansion):
    """
    Creates a Gurobi model for the master problem.
    """
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr, non_binary_index_set = meta_data

    x_R = model.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
    y = model.addVars(J, vtype=GRB.CONTINUOUS,name='y')
    dual = model.addVars(ll_constr,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
    w = model.addVars(jr,vtype=GRB.CONTINUOUS, name="w")
    s = model.addVars(non_binary_index_set,vtype= GRB.BINARY,name='s')
    model.update()
    model._x_I = x_I
    model._s = s
    s_x_vector = get_s_x_vector(x_I,s,jr)
    #setObjective(model)
    HG = block_diag(H,G_u)
    cd = concatenate((c,d_u))
    primalvars = x_I.select() + x_R.select() + y.select()
    model.setMObjective(Q=HG/2,c=cd,constant=0.0,xQ_L=primalvars,xQ_R=primalvars,xc=primalvars,sense=GRB.MINIMIZE)
    #setPConstraint(model)
    AB = concatenate((A,B),1)
    model.addMConstr(A=AB,x=primalvars,sense='>=',b=a)

    CD = concatenate((C,D),1)
    lower_level_vars = x_I.select() + y.select()
    model.addMConstr(A=CD,x=lower_level_vars,sense='>=',b=b)
    #setDualFeasiblityConstraint(model)
    GD = concatenate((D.T,-G_l),1)
    y_lambda = dual.select() + y.select()
    model.addMConstr(A=GD,x=y_lambda,sense='=',b=d_l)
    #setStrongDualityLinearizationConstraint(model)
    is_non_binary = list(map(lambda l,u: l != 0 or u != 1,int_lb,int_ub))
    if optimized_binary_expansion:
        bin_exp_constant = int_lb
    else:
        bin_exp_constant = zeros_like(int_lb)
    model.addConstrs((s.prod(bin_coeff_dict,j,'*') + bin_exp_constant[j] == x_I[j] for j,r in jr if is_non_binary[j]),'binary expansion')
    ub = big_M
    lb = -big_M
    model.addConstrs((w[j,r] <= ub*s_x_vector[j,r] for j,r in jr),'13a')
    model.addConstrs((w[j,r] <= quicksum([C[i,j]*dual[i] for i in ll_constr]) + lb*(s_x_vector[j,r] - 1) for j,r in jr),'13b')
    model.addConstrs((w[j,r] >= lb*s_x_vector[j,r] for j,r in jr),'13c')
    model.addConstrs((w[j,r] >= quicksum([C[(i,j)]*dual[i] for i in ll_constr]) + ub*(s_x_vector[j,r] - 1) for j,r in jr),'13d')
    return model,y.select(),dual.select(),w.select()

def get_s_x_vector(x,s,jr):
    """
    Generates a dictionary of the binary variables for the linearization of the strong duality constraint.
    This is required because we only need to introduce new binary variables s_jr if x_j is not already binary.
    The function returns a dictionary of the additional binary variables s and the x_js that are already binary
    with meaningful keys that can be used for setting up constraints.
    """
    res = {}
    for j,r in jr:
        try:
            res[(j,r)] = s[(j,r)]
        except KeyError:
            res[(j,r)] = x[j]
    return res




def getX_IParam(model):
    """
    Retrieves the solutions of the upper-level integer variables from a solved master problem
    so that they can be used as parameters for the subproblem or the feasibility problem in the multi-tree approach.
    """
    res = []
    for v in model.getVars():
        if match(r'^x_I',v.varName):
            res.append(v.x)
    return array(res)

def getX_IParamLazy(model):
    """
    Retrieves the solutions of the upper-level integer variables from a solved master problem
    so that they can be used as parameters for the subproblem or the feasibility problem in the single-tree approach.
    """
    res = []
    sol = model.cbGetSolution(model._x_I)
    for v in sol:
        res.append(sol[v])
    return array(res)

def getSParam(model):
    """
    Retrieves the solutions of the additional binary variables from the binary expansion from a solved master problem
    so that they can be used as parameters for the subproblem or the feasibility problem in the single-tree approach.
    """
    name_exp = compile(r'^s')
    index_exp = compile(r'(?<=\[)\d+(?=,)|(?<=,)\d+(?=\])')
    s = {}
    for var in model._vars:
        if name_exp.match(var.varName) is not None:
            indices = list(map(int,index_exp.findall(var.varName)))
            if len(indices) != 2:
                raise ValueError('Regex did not find exactly two indices')
            s[indices[0],indices[1]] = model.cbGetSolution(var)
    return s


def check_dimensions(problem_data):
    """
    Check if the all matrices and vectors fit the problem dimensions.
    """
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    if H.shape != (n_I+n_R,n_I+n_R):
        raise ValueError('Dimension of H is not n_I+n_R x n_I+n_R.')
    elif G_u.shape != (n_y, n_y):
        raise ValueError('Dimension of G_u is not n_y x n_y.')
    elif G_l.shape != (n_y, n_y):
        raise ValueError('Dimension of G_l is not n_y x n_y.')
    elif c.shape != (n_I+n_R,):
        raise ValueError('Dimension of c is not n_I+n_R')
    elif d_u.shape != (n_y,):
        raise ValueError('Dimension of d_u is not n_y')
    elif d_l.shape != (n_y,):
        raise ValueError('Dimension of d_l is not n_y')
    elif A.shape != (m_u, n_I+n_R):
        raise ValueError('Dimension of A is not m_u x n_I+n_R')
    elif B.shape != (m_u, n_y):
        raise ValueError('Dimension of B is not m_u x n_y')
    elif a.shape != (m_u,):
        raise ValueError('Dimension of a is not m_u')
    elif int_lb.shape != (n_I,):
        raise ValueError('Dimension of int_lb is not n_I')
    elif int_ub.shape != (n_I,):
        raise ValueError('Dimension of int_ub is not n_I')
    elif C.shape != (m_l, n_I,):
        raise ValueError('Dimension of C is not m_l x n_I')
    elif D.shape != (m_l, n_y):
        raise ValueError('Dimension of D is not m_l x n_y')
    elif b.shape != (m_l,):
        raise ValueError('Dimension of b is not m_l')
    else:
        pass

def optimize(model):
    """
    Optimizes a Gurobi Model and returns status, variables and objective value.
    """
    model.optimize()
    status = model.status
    if status == GRB.OPTIMAL or status == GRB.SUBOPTIMAL or status == 15:
        return model.status, model.getVars(),model.ObjVal
    else:
        return model.status, None, infty

def getBinaryCoeffsArray(index_set):
    """
    Generates the coefficients 1,2,4 etc. for the binary expansion in the form of a numpy array.
    """
    bi_c_arr = []
    for (j,r) in index_set:
        bi_c_arr.append(2**r)
    return array(bi_c_arr)

def getBinaryCoeffsDict(index_set):
    """
    Generates the coefficients 1,2,4 etc. for the binary expansion in the form of a dictionary.
    """
    bin_coeff = {}
    for (j,r) in index_set:
        bin_coeff[(j,r)] = 2**r
    return bin_coeff


def add_cut(problem_data,model,meta_data,y_var,dual_var,w_var,p,optimized_binary_expansion):
    """
    Takes a point \bar{y}, linearizes strong duality constraint at this point and adds the constraint to the model.
    This function is used in the multi-tree approach.
    """
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr,non_binary_index_set = meta_data
    cut_point = p
    
    yTGy = cut_point.T @ G_l @ cut_point
    term1 = 2*cut_point.T @ G_l @ y_var 
    term2 = d_l.T @ y_var 
    term3 = -b.T @ dual_var 
    term4 = bin_coeff_arr@w_var 
    if optimized_binary_expansion:
        bin_exp_constant = int_lb.T @ C.T @ dual_var
    else:
        bin_exp_constant = 0
    
    model.addConstr((term1+term2+term3+term4-yTGy + bin_exp_constant <= 0),'Strong duality linearization')
    return model

def add_lazy_constraint(cut_point,model):
    """
    Adds an outer approximation cut of the strong duality constraint to the model
    as a Gurobi lazy constraint. This function is invoked during callbacks of the single-tree approach.
    """
    _,_,_,_,_,_,_,_,_,_,_,_,_,_,int_lb,_,C,_,_ = model._problem_data
    yTGy = cut_point.T @ model._G_l @ cut_point
    term1 = 2*cut_point.T @ model._G_l @ model._y
    term2 = model._d_l.T @ model._y
    term3 = -model._b.T @ model._dual
    term4 = model._bin_coeff@ model._w
    if model._optimized_binary_expansion:
        bin_exp_constant = int_lb.T @ C.T @ model._dual
    else:
        bin_exp_constant = 0
    model.cbLazy(term1+term2+term3+term4-yTGy + bin_exp_constant <= 0)
    return model


def warmstart(model,solution):
    """
    Sets the start attribute of model variables according to a given solution to warmstart a problem.
    """
    if solution == {}:
        return model
    variables = model.getVars()
    for v in variables:
        try:
            v_name = v.varName
            start = solution[v_name]
            v.setAttr(GRB.Attr.Start, start)
        except KeyError:#Generally not good style to handle exceptions like this, but this means that no start sol is available, so it cannot be set.
            continue
    model.update()
    return model


