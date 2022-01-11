from gurobipy import GRB,tuplelist
import gurobipy as gp
from numpy import concatenate,diag,array,identity
from scipy.linalg import block_diag

def setup_kkt_miqp(problem_data,M):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    model = gp.Model("KKT-MIQP")
    x_I = model.addMVar(shape=n_I,lb=int_lb,ub=int_ub,vtype=GRB.INTEGER,name="x_I")
    x_R = model.addMVar(shape=n_R, vtype=GRB.CONTINUOUS,name='x_R')
    y = model.addMVar(shape=n_y, vtype=GRB.CONTINUOUS,name='y')
    dual = model.addMVar(shape=m_l,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
    v = model.addMVar(shape=m_l,vtype=GRB.BINARY,name="v")

    """ I = tuplelist([a for a in range(0,n_I)])
    R = tuplelist([a for a in range(0,n_R)])
    J = tuplelist([a for a in range(0,n_y)])
    ll_constr = tuplelist([a for a in range(0,m_l)])

    x_I = model.addVars(I, vtype=GRB.INTEGER,lb=int_lb, ub=int_ub,name='x_I')
    x_R = model.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
    y = model.addVars(J, vtype=GRB.CONTINUOUS,name='y')
    dual = model.addVars(ll_constr,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
    v = model.addVars(ll_constr,vtype=GRB.BINARY,name="v") """

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

    CDM = concatenate((C,D,diag(array([M]*m_l))),axis=1)
    model.addMConstr(A=CDM,x=x_I.tolist()+y.tolist()+v.tolist(),sense='<=',b=b)

    IM = concatenate((identity(m_l),diag(array([M]*m_l))),axis=1)
    model.addMConstr(A=IM,x=dual.tolist()+v.tolist(),sense="<=",b=array([M]*m_l))
    return model