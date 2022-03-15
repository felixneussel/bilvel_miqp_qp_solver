#
#This file contains functions to set up the feasibility problem in different contexts.
#
from numpy import concatenate
from gurobipy import Model, GRB 

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
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr,_ = meta_data
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