from Functional.problems import setup_meta_data, setup_master
from gurobipy import GRB,tuplelist,gp
from numpy import concatenate,diag,array,identity
from scipy.linalg import block_diag

def setup_kkt_miqp(problem_data,M):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    model = gp.Model("KKT-MIQP")
    model.Params.LogToConsole = 0
    model.setParam(GRB.Param.DualReductions,0)
    x_I = model.addMVar(shape=n_I,lb=int_lb,ub=int_ub,vtype=GRB.INTEGER,name="x_I")
    x_R = model.addMVar(shape=n_R, vtype=GRB.CONTINUOUS,name='x_R')
    y = model.addMVar(shape=n_y, vtype=GRB.CONTINUOUS,name='y')
    dual = model.addMVar(shape=m_l,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
    v = model.addMVar(shape=m_l,vtype=GRB.BINARY,name="v")

    HG = block_diag(H,G_u)
    cd = concatenate((c,d_u))
    obj_vars = x_I.tolist() + x_R.tolist() + y.tolist()
    model.setMObjective(Q=HG/2,c=cd,constant=0.0,xQ_L=obj_vars,xQ_R=obj_vars,xc=obj_vars,sense=GRB.MINIMIZE)

    AB = concatenate((A,B),1)
    model.addMConstr(A=AB,x=obj_vars,sense='>=',b=a)

    CD = concatenate((C,D),1)
    lower_level_vars = x_I.tolist() + y.tolist()
    model.addMConstr(A=CD,x=lower_level_vars,sense='>=',b=b)

    GD = concatenate((D.T,-G_l),1)
    y_lambda = dual.tolist() + y.tolist()
    model.addMConstr(A=GD,x=y_lambda,sense='=',b=d_l)

    CDM = concatenate((C,D,-diag(array([M]*m_l))),axis=1)
    #model.addMConstr(A=CDM,x=x_I.tolist()+y.tolist()+v.tolist(),sense='<=',b=b)

    IM = concatenate((identity(m_l),diag(array([M]*m_l))),axis=1)
    #model.addMConstr(A=IM,x=dual.tolist()+v.tolist(),sense="<=",b=array([M]*m_l))
    model.addMConstr(concatenate((identity(m_l),-C,-D),axis=1),v.tolist()+x_I.tolist()+y.tolist(),"=",-b)
    v_list = v.tolist()
    for i, var in enumerate(dual.tolist()):
        model.addSOS(GRB.SOS_TYPE1,[var,v_list[i]])
    return model

def setup_sd_miqcpcp(problem_data,big_M,optimized_binary_expansion):
    _,_,_,_,_,_,_,G_l,_,_,d_l,_,_,_,_,_,_,_,b = problem_data
    meta_data = setup_meta_data(problem_data,optimized_binary_expansion)
    _,_,_,_,_,_,bin_coeff_arr, _ = meta_data
    model,y,dual,w = setup_master(problem_data,meta_data,big_M,optimized_binary_expansion)
    #setup strong duality constraint
    linear_vector = concatenate((d_l, - b, bin_coeff_arr))
    y_lam_w = model.y.select() + dual.select() + w.select()
    model.addMQConstr(Q = G_l, c = linear_vector, sense="<", rhs=0, xQ_L=y.select(), xQ_R=y.select(), xc=y_lam_w, name="Strong Duality Constraint" )
    return model

def optimize_benchmark(model,time_limit):
    model.setParam(GRB.Param.TimeLimit,time_limit)
    model.optimize()
    return model.status,model.ObjVal,model.Runtime,model.MIPGap

