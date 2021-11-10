import gurobipy as gp
from gurobipy import GRB
import matrix_operations as mo
import numpy as np

def getIndexSet(N):
    if len(N) == 1:
        return gp.tuplelist([a for a in range(N[0])])
    elif len(N) == 2:
        return gp.tuplelist([(a,b) for a in range(N[0]) for b in range(N[1])])
    else:
        raise ValueError('Too many entries in list')

def addVariables(mp,int_lb,int_ub,I,R,J,ll_constr,jr):
    x_I = mp.addVars(I, vtype=GRB.INTEGER,lb=int_lb, ub=int_ub,name='x_I')
    x_R = mp.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
    y = mp.addVars(J, vtype=GRB.CONTINUOUS,name='y')
    dual = mp.addVars(ll_constr,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
    s = mp.addVars(jr,vtype= GRB.BINARY,name='s')
    w = mp.addVars(jr,name="w")
    mp.update()
    return x_I, x_R,y,dual,s,w

def setObjective(mp,H,G,c,d,x_I,x_R,y):
    HG = mo.concatenateDiagonally(H,G)
    cd = np.concatenate((c,d))
    primalvars = x_I.select() + x_R.select() + y.select()

    mp.setMObjective(Q=HG/2,c=cd,constant=0.0,xQ_L=primalvars,xQ_R=primalvars,xc=primalvars,sense=GRB.MINIMIZE)
    mp.update()

def setPConstraint(mp,A,B,a,C,D,b,x_I,x_R,y):
    AB = mo.concatenateHorizontally(A,B)
    primalvars = x_I.select() + x_R.select() + y.select()
    mp.addMConstr(A=AB,x=primalvars,sense='>=',b=a)

    CD = mo.concatenateHorizontally(C,D)
    lower_level_vars = x_I.select() + y.select()
    mp.addMConstr(A=CD,x=lower_level_vars,sense='>=',b=b)
    mp.update()

def setDualFeasiblityConstraint(mp,D,G,dual,y,d):
    GD = mo.concatenateHorizontally(D.T,-G)
    y_lambda = dual.select() + y.select()
    mp.addMConstr(A=GD,x=y_lambda,sense='=',b=d)

def setStrongDualityLinearizationConstraint(mp,x_I,dual,s,w,jr,ll_constr,C):
    #Note, since our r indices start at zero, we write 2**r instead of 2**(r-1)
    bin_coeff = {}
    for (j,r) in jr:
        bin_coeff[(j,r)] = 2**r
    #Note, since our r indices start at zero, we write 2**r instead of 2**(r-1)
    #master.addConstrs((sum(2**r*s[j,r] for r in jr[j,'*']) == x_I[j] for j in I),'binary')
    mp.addConstrs((s.prod(bin_coeff,j,'*') == x_I[j] for j,r in jr),'binary')

    ub = getUpperBound()
    lb = getLowerBound()
    mp.addConstrs((w[j,r] <= ub*s[j,r] for j,r in jr),'13a')
    mp.addConstrs((w[j,r] <= sum([C[i,j]*dual[i] for i in ll_constr]) + lb*(s[j,r] - 1) for j,r in jr),'13b')
    #Possible refactor: replace lam_coeff with C and get rid of lam_coeff

    mp.addConstrs((w[j,r] >= lb*s[j,r] for j,r in jr),'13c')
    mp.addConstrs((w[j,r] >= sum([C[(i,j)]*dual[i] for i in ll_constr]) + ub*(s[j,r] - 1) for j,r in jr),'13d')

def getLowerBound():
    """
    Returns a lower bound for the w_jr constraints.
    """
    return -100

def getUpperBound():
    """
    Returns an upper bound for the w_jr constraints.
    """
    return 100
