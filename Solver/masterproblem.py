
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib import RankWarning
from . Internal_Lib.matrix_operations import *

def getNumOfBinaryDigits(x_plus):
    """
    Takes the numpy array of integer-variable upper bounds as an input and
    returns an np array with the numbers of binary digits needed to represent
    each upper bound
    """
    return (np.floor(np.log2(x_plus)) + np.ones(x_plus.shape)).astype(int)


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

def addCut(model,point,G,d,b,y_var,dual_var,w_var,y_ind,dual_ind,w_ind, binary_coeff):
    """
    Takes a point \bar{y}, linearizes strong duality constraint at this point and adds the constraint to the model.
    """
    twoyTG = 2*point.T @ G
    yTGy = point.T @ G @ point
    term1 = sum([twoyTG[i]*y_var[i] for i in y_ind])
    term2 = sum([d[i]*y[i] for i in y_ind])
    term3 = -sum([b[j]*dual_var[j] for j in dual_ind])
    term4 = w_var.prod(binary_coeff)
    
    model.addConstr((term1+term2+term3+term4-yTGy <= 0),'Strong duality relaxation')

def setupMaster(n_I,n_R,n_y,m_u,m_l,H,G,c,d,A,B,a,int_lb,in_ub,C,D,b):
    """
    This function takes all the Input data for the MIQP-QP (1) in Kleinert et al and creates the Gurobi model of the Masterproblem (M^p)
    used in Algorithm 1 

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
    """
    checkDimensions(n_I,n_R,n_y,m_u,m_l,H,G,c,d,A,B,a,int_lb,in_ub,C,D,b)

def checkDimensions(n_I,n_R,n_y,m_u,m_l,H,G,c,d,A,B,a,int_lb,in_ub,C,D,b):
    """
    This function takes all the Input data for the MIQP-QP (1) in 
    Kleinert et al and makes sure that it has the right dimensions

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
    """
    if H.shape != (n_I+n_R,n_I+n_R):
        raise ValueError('Dimension of H is not n_I+n_R x n_I+n_R.')
    if G.shape != (n_y, n_y):
        raise ValueError('Dimension of G is not n_y x n_y.')
    

#Dimensions
#Number of Integer upper-level variables
n_I = 1
#Number of Continuous upper-level variables
n_R = 1
#Number of lower-level variables
n_y = 2
#Number of upper level constraints
m_u = 2
#Number of lower level constaints
m_l = 2

#Input data
H = np.array([[1,2],[3,4]])
G = np.array([[1,0,],[0,1]])
c = np.array([1,0])
d = np.array([1,0])

A = np.array([[2,0,],[3,1]])
B = np.array([[1,0,],[0,1]])
a = np.array([1,0])

int_lb = np.array([0])
int_ub = np.array([5])

C = np.array([[5],[0]])
D = np.array([[4,2],[7,2]])
b = np.array([1,0])

r_bar = getNumOfBinaryDigits(int_ub)


#Create tuplelist all possible indices j \in I, r \in[\bar{r_j}]
jr = gp.tuplelist([(a,b) for a in range(0,n_I) for b in range(0, r_bar[a])])#Caution, r_bar[a] was changed from r_bar[a-1]
I = gp.tuplelist([a for a in range(0,n_I)])
R = gp.tuplelist([a for a in range(0,n_R)])
J = gp.tuplelist([a for a in range(0,n_y)])
ll_constr = gp.tuplelist([a for a in range(0,m_l)])
ij = gp.tuplelist([(a,b) for a in range(0,m_l) for b in range(0,n_I)])


#Note, since our r indices start at zero, we write 2**r instead of 2**(r-1)
bin_coeff = {}
for (j,r) in jr:
    bin_coeff[(j,r)] = 2**r



lam_coeff ={}
for i,j in ij:
    lam_coeff[(i,j)] = C[i,j]


#Create model
master = gp.Model('Masterproblem')

#Add variables
#x_I = master.addMVar(shape=n_I, vtype=GRB.INTEGER,lb=int_lb, ub=int_ub, name='x_I')
#x_R = master.addMVar(shape= n_R, vtype = GRB.CONTINUOUS, name='x_R')
#y = master.addMVar(shape = n_y, vtype = GRB.CONTINUOUS, name='y')
#dual = master.addMVar(shape = m_l, vtype=GRB.CONTINUOUS, lb=0, name='lambda')

x_I = master.addVars(I, vtype=GRB.INTEGER,lb=int_lb, ub=int_ub,name='x_I')
x_R = master.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
y = master.addVars(J, vtype=GRB.CONTINUOUS,name='y')
dual = master.addVars(ll_constr,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
s = master.addVars(jr,vtype= GRB.BINARY,name='s')
w = master.addVars(jr,name="w")
master.update()




#Objective function
HG = mo.concatenateDiagonally(H,G)
cd = np.concatenate((c,d))
primalvars = x_I.select() + x_R.select() + y.select()

master.setMObjective(Q=HG/2,c=cd,constant=0.0,xQ_L=primalvars,xQ_R=primalvars,xc=primalvars,sense=GRB.MINIMIZE)
master.update()

#Constraints
AB = mo.concatenateHorizontally(A,B)
master.addMConstr(A=AB,x=primalvars,sense='>=',b=a)

CD = mo.concatenateHorizontally(C,D)
lower_level_vars = x_I.select() + y.select()
master.addMConstr(A=CD,x=lower_level_vars,sense='>=',b=b)

GD = mo.concatenateHorizontally(D.T,-G)
y_lambda = dual.select() + y.select()
master.addMConstr(A=GD,x=y_lambda,sense='=',b=d)

#Note, since our r indices start at zero, we write 2**r instead of 2**(r-1)
#master.addConstrs((sum(2**r*s[j,r] for r in jr[j,'*']) == x_I[j] for j in I),'binary')
master.addConstrs((s.prod(bin_coeff,j,'*') == x_I[j] for j,r in jr),'binary')

ub = getUpperBound()
lb = getLowerBound()
master.addConstrs((w[j,r] <= ub*s[j,r] for j,r in jr),'13a')
master.addConstrs((w[j,r] <= sum([lam_coeff[i,j]*dual[i] for i in ll_constr]) + lb*(s[j,r] - 1) for j,r in jr),'13b')
#Possible refactor: replace lam_coeff with C and get rid of lam_coeff

master.addConstrs((w[j,r] >= lb*s[j,r] for j,r in jr),'13c')
master.addConstrs((w[j,r] >= sum([lam_coeff[(i,j)]*dual[i] for i in ll_constr]) + ub*(s[j,r] - 1) for j,r in jr),'13d')

res = master.optimize()

#Test Strong duality relaxation
rel_point = np.array([master.getVarByName(f'y[{i}]').x for i in range(n_y)])

addCut(master,rel_point,G,d,b,y,dual,w,J,ll_constr,jr,bin_coeff)


setupMaster(n_I,n_R,n_y,m_u,m_l,H,G,c,d,A,B,a,int_lb,int_ub,C,D,b)