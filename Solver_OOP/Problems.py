import numpy as np
from miqp_qp_solver import MIQP_QP


#Number of Integer upper-level variables
n_I = 1
#Number of Continuous upper-level variables
n_R = 0
#Number of lower-level variables
n_y = 1
#Number of upper level constraints
m_u = 2
#Number of lower level constaints
m_l = 3

#Input data
H = np.array([[2]])
G_u = np.array([[2]])
G_l = np.array([[2]])
c = np.array([-6])
d_u = np.array([-4])
d_l = np.array([-10])

A = np.array([[-1],[1]])
B = np.array([[0],[0]])
a = np.array([-8,0])

int_lb = np.array([0])
int_ub = np.array([8])

C = np.array([[2],[-1],[-1]])
D = np.array([[-1],[2],[-2]])
b = np.array([-1,2,-14])

ClarkWesterberg1990a = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]




#Number of Integer upper-level variables
n_I = 1
#Number of Continuous upper-level variables
n_R = 0
#Number of lower-level variables
n_y = 1
#Number of upper level constraints
m_u = 5
#Number of lower level constaints
m_l = 2

#Input data
H = np.array([[2]])
G_u = np.array([[2]])
G_l = np.array([[2]])
c = np.array([-6])
d_u = np.array([-4])
d_l = np.array([-10])

A = np.array([[1],[-1],[2],[-1],[-1]])
B = np.array([[0],[0],[-1],[2],[-2]])
a = np.array([0,-8,-1,2,-14])

int_lb = np.array([0])
int_ub = np.array([8])

C = np.array([[0],[0]])
D = np.array([[1],[-1]])
b = np.array([0,-10])

GumusFloudas2001Ex4 = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]

problems = {}
problems['GumusFloudas2001Ex4'] = GumusFloudas2001Ex4
problems['ClarkWesterberg1990a'] = ClarkWesterberg1990a




def loadProblem(name):
    return problems[name]