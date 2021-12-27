import enum
from gurobipy import Model,GRB, tuplelist
from numpy import ones,log2,floor,ceil, concatenate, array, infty
from scipy.linalg import block_diag
from re import match
from operator import itemgetter

def setup_meta_data(problem_data):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    r_bar = (floor(log2(int_ub)) + ones(int_ub.shape)).astype(int)
    jr = tuplelist([(a,b) for a in range(0,n_I) for b in range(0, r_bar[a])])#Caution, r_bar[a] was changed from r_bar[a-1]
    I = tuplelist([a for a in range(0,n_I)])
    R = tuplelist([a for a in range(0,n_R)])
    J = tuplelist([a for a in range(0,n_y)])
    ll_constr = tuplelist([a for a in range(0,m_l)])
    bin_coeff_dict = getBinaryCoeffsDict(jr)
    bin_coeff_arr = getBinaryCoeffsArray(jr)
    return jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr

def mp_common(problem_data,meta_data,model,x_I):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data

    x_R = model.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
    y = model.addVars(J, vtype=GRB.CONTINUOUS,name='y')
    dual = model.addVars(ll_constr,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
    w = model.addVars(jr,vtype=GRB.CONTINUOUS, name="w")
    s = model.addVars(jr,vtype= GRB.BINARY,name='s')
    model.update()
    model._x_I = x_I
    model._s = s
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
    model.addConstrs((s.prod(bin_coeff_dict,j,'*') == x_I[j] for j,r in jr),'binary expansion')
    ub = 1e5
    lb = -1e5
    model.addConstrs((w[j,r] <= ub*s[j,r] for j,r in jr),'13a')
    model.addConstrs((w[j,r] <= sum([C[i,j]*dual[i] for i in ll_constr]) + lb*(s[j,r] - 1) for j,r in jr),'13b')
    model.addConstrs((w[j,r] >= lb*s[j,r] for j,r in jr),'13c')
    model.addConstrs((w[j,r] >= sum([C[(i,j)]*dual[i] for i in ll_constr]) + ub*(s[j,r] - 1) for j,r in jr),'13d')
    return model,y.select(),dual.select(),w.select()

def setup_st_master(problem_data,meta_data):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    model = Model('Masterproblem')
    model.Params.LogToConsole = 0
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
    x_I = model.addVars(I, vtype=GRB.CONTINUOUS,lb=int_lb, ub=int_ub,name='x_I')
    return mp_common(problem_data,meta_data,model,x_I)

def getX_IParam(model):
    res = []
    sol = model.cbGetSolution(model._x_I)
    for v in sol:
        res.append(sol[v])
    return array(res)

def getSParam(model):
    res = []
    sol = model.cbGetSolution(model._s)
    for v in sol:
        res.append(sol[v])
    return array(res)

def setup_master(problem_data,meta_data):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    model = Model('Masterproblem')
    model.Params.LogToConsole = 0
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
    x_I = model.addVars(I, vtype=GRB.INTEGER,lb=int_lb, ub=int_ub,name='x_I')
    return mp_common(problem_data,meta_data,model,x_I)

def setup_sub(problem_data,master,meta_data,y_var,dual_var,w_var):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
    model = master.fixed()
    linear_vector = concatenate((d_l, - b, bin_coeff_arr))
    y_lam_w = y_var + dual_var + w_var
    model.addMQConstr(Q = G_l, c = linear_vector, sense="<", rhs=0, xQ_L=y_var, xQ_R=y_var, xc=y_lam_w, name="Strong Duality Constraint" )
    return model

def setup_sub_st_lazy(problem_data,meta_data,x_I_param,s_param):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
    x_I_param = x_I_param
    s_param = s_param
    model = Model('Subproblem')
    model.Params.LogToConsole = 0
    #add variables
    x_R = model.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
    y = model.addVars(J, vtype=GRB.CONTINUOUS,name='y')
    dual = model.addVars(ll_constr,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
    w = model.addVars(jr,vtype=GRB.CONTINUOUS, name="w")
    #set objective
    #Slice H into quadrants corresponding to terms with x_I, x_R or and x_I - x_R-mixed-term
    H_II = H[:n_I,:n_I]
    H_RR = H[n_I:,n_I:]
    H_IR = H[:n_I,n_I:]
    #slice c into vectors corresponding to x_I and x_R
    c_I = c[:n_I]
    c_R = c[n_I:]

    quad_matrix = block_diag(H_RR,G_u)
    lin_vec = concatenate((c_R.T+x_I_param.T@H_IR,d_u.T)).T
    constant_term = 0.5*x_I_param@H_II@x_I_param + c_I@x_I_param
    vars = x_R.select() + y.select()
    model.setMObjective(Q=quad_matrix/2,c=lin_vec,constant=constant_term,xQ_L=vars,xQ_R=vars,xc=vars,sense=GRB.MINIMIZE)
    #set P-constraint
    A_I = A[:,:n_I]
    A_R = A[:,n_I:]
    AB = concatenate((A_R,B),1)
    primalvars = x_R.select() + y.select()
    model.addMConstr(A=AB,x=primalvars,sense='>=',b=a-A_I@x_I_param)
    model.addMConstr(A=D,x=y.select(),sense='>=',b=b - C@x_I_param)
    #set dual feasibility constriant
    GD = concatenate((D.T,-G_l),1)
    y_lambda = dual.select() + y.select()
    model.addMConstr(A=GD,x=y_lambda,sense='=',b=d_l)
    #set strong duality linearization constraint
    model.addConstrs((w[j,r] == s_param[j,r]*sum([C[i,j]*dual[i] for i in ll_constr]) for j,r in jr), 'binary_expansion')
    #set strong duality constraint
    linear_vector = concatenate((d_l, - b, bin_coeff_arr))
    y_lam_w = y.select() + dual.select() + w.select()
    model.addMQConstr(Q = G_l, c = linear_vector, sense="<", rhs=0, xQ_L=y.select(), xQ_R=y.select(), xc=y_lam_w, name="Strong Duality Constraint" )
    return model


def setup_sub_st(problem_data,master,meta_data,y_var,dual_var,w_var):
    model = setup_sub(problem_data,master,meta_data,y_var,dual_var,w_var)
    #model = removeMasterLinearizations(model,cut_counter)
    return model

def setup_sub_mt(problem_data,master,meta_data,y_var,dual_var,w_var):
    model = setup_sub(problem_data,master,meta_data,y_var,dual_var,w_var)
    #model = removeBinaryExpansion(model)
    #model = removeMasterLinearizations(model,cut_counter)
    return model

def setup_sub_rem_1(problem_data,meta_data,x_I_param):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
    lower = setup_lower(n_y,m_l,G_l,d_l,C,D,b,x_I_param)
    lower_status,lower_vars,lower_obj = optimize(lower)
    model = Model('Subproblem')
    model.Params.LogToConsole = 0
    x_I = model.addVars(I, vtype=GRB.CONTINUOUS,lb=int_lb, ub=int_ub,name='x_I')
    x_R = model.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
    y = model.addVars(J, vtype=GRB.CONTINUOUS,name='y')
    #Objective
    #Slice H into quadrants corresponding to terms with x_I, x_R or and x_I - x_R-mixed-term
    H_II = H[:n_I,:n_I]
    H_RR = H[n_I:,n_I:]
    H_IR = H[:n_I,n_I:]
    #slice c into vectors corresponding to x_I and x_R
    c_I = c[:n_I]
    c_R = c[n_I:]
    quad_matrix = block_diag(H_RR,G_u)
    lin_vec = concatenate((c_R.T+x_I_param.T@H_IR,d_u.T)).T
    constant_term = 0.5*x_I_param@H_II@x_I_param + c_I@x_I_param
    vars = x_R.select() + y.select()
    model.setMObjective(Q=quad_matrix/2,c=lin_vec,constant=constant_term,xQ_L=vars,xQ_R=vars,xc=vars,sense=GRB.MINIMIZE)
    #P Constraint
    A_I = A[:,:n_I]
    A_R = A[:,n_I:]
    AB = concatenate((A_R,B),1)
    primalvars = x_R.select() + y.select()
    model.addMConstr(A=AB,x=primalvars,sense='>=',b=a-A_I@x_I_param)
    model.addMConstr(A=D,x=y.select(),sense='>=',b=b - C@x_I_param)
    #Lower Level Optimality constraint
    model.addMQConstr(Q = G_l/2, c = d_l, sense="<", rhs=lower_obj, xQ_L=y.select(), xQ_R=y.select(), xc=y.select(), name="Lower Level Optimality" )
    return model

def setup_sub_rem_2(problem_data,meta_data,x_I_param):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
    model = Model('Subproblem')
    lower = setup_lower(n_y,m_l,G_l,d_l,C,D,b,x_I_param)
    lower_status,lower_vars,lower_obj = optimize(lower)
    y_param = []
    for v in lower_vars:
        y_param.append(v.x)
    y_param = array(y_param)
    model.Params.LogToConsole = 0
    x_R = model.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
    #Objective
    #Slice H into quadrants corresponding to terms with x_I, x_R or and x_I - x_R-mixed-term
    H_II = H[:n_I,:n_I]
    H_RR = H[n_I:,n_I:]
    H_IR = H[:n_I,n_I:]
    #slice c into vectors corresponding to x_I and x_R
    c_I = c[:n_I]
    c_R = c[n_I:]
    lin_vec = (c_R.T+x_I_param.T@H_IR).T
    constant_term = 0.5*x_I_param@H_II@x_I_param + c_I@x_I_param + 0.5 * y_param.T @ G_u @ y_param + d_u @ y_param 
    vars = x_R.select()
    model.setMObjective(Q=H_RR/2,c=lin_vec,constant=constant_term,xQ_L=vars,xQ_R=vars,xc=vars,sense=GRB.MINIMIZE)
    A_I = A[:,:n_I]
    A_R = A[:,n_I:]
    primalvars = x_R.select()
    model.addMConstr(A=A_R,x=primalvars,sense='>=',b=a-A_I@x_I_param-B@y_param)
    return model, y_param


def setup_lower(n_y,m_l,G_l,d_l,C,D,b,x_I_param):
    model = Model('Lower_Level')
    model.Params.LogToConsole = 0
    y = model.addMVar(shape=n_y,vtype = GRB.CONTINUOUS,name = 'y')
    model.setMObjective(Q=G_l/2, c = d_l, constant=0, xQ_L=y, xQ_R=y, xc=y, sense=GRB.MINIMIZE )
    model.addMConstr(A=D, x=y, sense='>', b=b - C@x_I_param, name="Lower Level Constraints" )
    return model

def setup_feas(problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
    model = master.fixed()
    #set new Objective
    linear_vector = concatenate((d_l, - b, bin_coeff_arr))
    y_lam_w = y_var + dual_var + w_var
    model.setMObjective(Q=G_l,c=linear_vector,constant=0,xQ_L=y_var,xQ_R=y_var,xc=y_lam_w,sense=GRB.MINIMIZE)
    return model

def setup_feas_st(problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter):
    model = setup_feas(problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter)
    #model = removeMasterLinearizations(model,cut_counter)
    return model

def setup_feas_mt(problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter):
    model = setup_feas(problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter)
    #model = removeMasterLinearizations(model,cut_counter)
    #model = removeBinaryExpansion(model)
    return model

def setup_feas_lazy(problem_data,meta_data,x_I_param,s_param):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
    x_I_param = x_I_param
    s_param = s_param
    model = Model('Feasiblity-Problem')
    model.Params.LogToConsole = 0
    #add variables
    x_R = model.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
    y = model.addVars(J, vtype=GRB.CONTINUOUS,name='y')
    dual = model.addVars(ll_constr,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
    w = model.addVars(jr,vtype=GRB.CONTINUOUS, name="w")
    #set objective
    linear_vector = concatenate((d_l, - b, bin_coeff_arr))
    y_lam_w = y.select() + dual.select() + w.select()
    model.setMObjective(Q=G_l,c=linear_vector,constant=0,xQ_L=y.select(),xQ_R=y.select(),xc=y_lam_w,sense=GRB.MINIMIZE)
    #set P-constraint
    A_I = A[:,:n_I]
    A_R = A[:,n_I:]
    AB = concatenate((A_R,B),1)
    primalvars = x_R.select() + y.select()
    model.addMConstr(A=AB,x=primalvars,sense='>=',b=a-A_I@x_I_param)
    model.addMConstr(A=D,x=y.select(),sense='>=',b=b - C@x_I_param)
    #set dual feasibility constriant
    GD = concatenate((D.T,-G_l),1)
    y_lambda = dual.select() + y.select()
    model.addMConstr(A=GD,x=y_lambda,sense='=',b=d_l)
    #set strong duality linearization constraint
    model.addConstrs((w[j,r] == s_param[j,r]*sum([C[i,j]*dual[i] for i in ll_constr]) for j,r in jr), 'binary_expansion')
    return model

def branch(model,int_vars,problem_data):
    m1 = model
    m2 = model
    x_I_m1 = list(filter(lambda v: match(r'^x_I',v.varName),m1.getVars()))
    x_I_m2 = list(filter(lambda v: match(r'^x_I',v.varName),m2.getVars()))
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    candidates = []
    for i,var in enumerate(int_vars):
        candidates.append((i,var,int_ub[i]-int_lb[i]))
    candidates = sorted(candidates,key=itemgetter(2),reverse=True)
    branch_index = 0
    for c in candidates:
        if not c[1].is_integer():
            branch_index = c[0]
            break
    int_ub[branch_index] = floor(int_vars[branch_index])
    int_lb[branch_index] = ceil(int_vars[branch_index])
    for i,var in enumerate(x_I_m1):
            var.setAttr('ub',int_ub[i])
    for i,var in enumerate(x_I_m2):
            var.setAttr('lb',int_lb[i])
    return m1,m2

def check_dimensions(problem_data):
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
    model.optimize()
    status = model.status
    if status == GRB.OPTIMAL or status == GRB.SUBOPTIMAL or status == 15:
        return model.status, model.getVars(),model.ObjVal
    else:
        return model.status, None, infty

def getBinaryCoeffsArray(index_set):
    bi_c_arr = []
    for (j,r) in index_set:
        bi_c_arr.append(2**r)
    return array(bi_c_arr)

def getBinaryCoeffsDict(index_set):
    bin_coeff = {}
    for (j,r) in index_set:
        bin_coeff[(j,r)] = 2**r
    return bin_coeff

def removeBinaryExpansion(model):
        constr = model.getConstrs()
        filtered_cons = list(filter(lambda c: match(r'^binary expansion',c.ConstrName) is not None,constr))
        for con in filtered_cons:
            model.remove(con)
        return model

def removeMasterLinearizations(model,cut_counter):
    constr = model.getConstrs()
    for i in range(cut_counter):
        model.remove(constr.pop())
    return model

def add_cut(problem_data,model,meta_data,y_var,dual_var,w_var,p):
    """
    Takes a point \bar{y}, linearizes strong duality constraint at this point and adds the constraint to the model.
    """
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
    cut_point = p
    #twoyTG = 2*cut_point.T @ G_l
    yTGy = cut_point.T @ G_l @ cut_point
    term1 = 2*cut_point.T @ G_l @ y_var #sum([twoyTG[i]*y_var(i)[0] for i in J])
    term2 = d_l.T @ y_var #sum([d_l[i]*y_var(i)[0] for i in J])
    term3 = -b.T @ dual_var #-sum([b[j]*dual_var(j)[0] for j in ll_constr])
    term4 = bin_coeff_arr@w_var #w_var.prod(bin_coeff_dict)
    
    model.addConstr((term1+term2+term3+term4-yTGy <= 0),'Strong duality linearization')
    return model

def get_int_vars(vars):
    if not vars:
        return
    int_vars = []
    for v in vars:
        if match(r'^x_I',v.varName):
            int_vars.append(v.x)
    return int_vars

def is_int_feasible(vars):
    is_int = list(map(lambda x: x.is_integer(),vars))
    return all(is_int)

def warmstart(model,solution):
    if solution == {}:
        return model
    for v in model.getVars():
        try:
            v.setAttr(GRB.Attr.Start, solution[v.varName])
        except KeyError:
            continue
    return model


