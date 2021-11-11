from Internal_Lib import optimization_modelling as opt
import gurobipy as gp
from gurobipy import GRB
def setupSub(n_I,n_R,n_y,m_u,m_l,H,G,c,d,A,B,a,int_lb,int_ub,C,D,b,x_I_sol,s_sol):
    """
    This function takes all the Input data and the solution from the 
    previously solved Masterproblem for the MIQP-QP (1) in Kleinert et al 
    and creates the Gurobi model of the Subproblem (S^p) used in Algorithm 1 

    Parameters
    
    n_I : Number of upper-level integer variables
    n_R : Number of upper-level continuous variables
    n_y : Number of lower-level variables
    m_u : Number of upper-level constraints
    m_l : Number of lower-level constraints
    H : Numpy-Matrix with dim n_I+n_R x n_I+n_R
    G : Numpy-Matrix with dim n_y x n_y
    c : Numpy-vector with dim n_I+n_R
    d : Numpy-Vector with dim n_y
    A : Numpy-Matrix with dim m_u x n_I+n_R
    B : Numpy-Matrix with dim m_u x n_y
    a : Numpy-Vector with dim m_u
    int_lb : Numpy-Vector with dim n_I containing lower bounds for integer variables
    int_ub : Numpy-Vector with dim n_I containing upper bounds for integer variables
    C : Numpy-Matrix with dim m_l x n_I
    D : Numpy-Matrix with dim m_l x n_y
    b : Numpy-Vector with dim m_l
    x_I_sol : Numpy-Vector with dim n_I containing the solution for x_I from the masterproblem
    s_sol : Numpy-Vector with dim ?? containing the solution for s from the masterproblem
    """

    r_bar = opt.getNumOfBinaryDigits(int_ub)

    I = opt.getIndexSet([n_I])
    R = opt.getIndexSet([n_R])
    J = opt.getIndexSet([n_y])
    ll_constr = opt.getIndexSet([m_l])
    jr = opt.getIndexSet([n_I,r_bar])

    mp = gp.Model('Subproblem')
    x_I, x_R, y, dual, s, w = addSubVariables(mp,int_lb,int_ub,I,R,J,ll_constr)

    setSubObjective(mp,H,G,c,d,x_I,x_R,y)


def addSubVariables(mp,int_lb,int_ub,I,R,J,ll_constr,jr):
    
    x_R = mp.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
    y = mp.addVars(J, vtype=GRB.CONTINUOUS,name='y')
    dual = mp.addVars(ll_constr,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
    w = mp.addVars(jr,name="w")
    mp.update()
    return x_R,y,dual,w
  
def setSubObjective(mp,H,G,c,d,x_I,x_R,y):
    pass