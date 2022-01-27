from numpy import concatenate,array
from gurobipy import Model,GRB
from scipy.linalg import block_diag
from Functional.problems import optimize

def setup_sub_as_fixed_nonconvex_reform(problem_data,meta_data,x_I_param):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr,non_binary_index_set = meta_data
    model = Model('Subproblem')
    model.Params.LogToConsole = 0
    #Variables
    x_R = model.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
    y = model.addVars(J, vtype=GRB.CONTINUOUS,name='y')
    dual = model.addVars(ll_constr,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
    #set objective
    H_II = H[:n_I,:n_I]
    H_RR = H[n_I:,n_I:]
    H_IR = H[:n_I,n_I:]
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
    #Strong duality constraint
    linear_vector = concatenate((d_l, - b, x_I_param.T @ C.T))
    y_lam_w = y.select() + dual.select() + dual.select()
    model.addMQConstr(Q = G_l, c = linear_vector, sense="<", rhs=0, xQ_L=y.select(), xQ_R=y.select(), xc=y_lam_w, name="Strong Duality Constraint" )
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

def setup_sub_rem_1(problem_data,meta_data,x_I_param):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr,_ = meta_data
    lower = setup_lower(n_y,m_l,G_l,d_l,C,D,b,x_I_param)
    lower_status,lower_vars,lower_obj = optimize(lower)
    model = Model('Subproblem')
    model.Params.LogToConsole = 0
    #x_I = model.addVars(I, vtype=GRB.CONTINUOUS,lb=int_lb, ub=int_ub,name='x_I')
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
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr,_ = meta_data
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

""" def setup_sub(problem_data,master,meta_data,y_var,dual_var,w_var):
n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
model = master.fixed()
linear_vector = concatenate((d_l, - b, bin_coeff_arr))
y_lam_w = y_var + dual_var + w_var
model.addMQConstr(Q = G_l, c = linear_vector, sense="<", rhs=0, xQ_L=y_var, xQ_R=y_var, xc=y_lam_w, name="Strong Duality Constraint" )
return model """

""" def setup_sub_st(problem_data,master,meta_data,y_var,dual_var,w_var):
    model = setup_sub(problem_data,master,meta_data,y_var,dual_var,w_var)
    #model = removeMasterLinearizations(model,cut_counter)
    return model """

""" def setup_sub_mt(problem_data,master,meta_data,y_var,dual_var,w_var):
    model = setup_sub(problem_data,master,meta_data,y_var,dual_var,w_var)
    #model = removeBinaryExpansion(model)
    #model = removeMasterLinearizations(model,cut_counter)
    return model """