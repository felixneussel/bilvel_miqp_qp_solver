import numpy as np
from Solver_OOP.miqp_qp_solver import MIQP_QP


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


#Number of Integer upper-level variables
n_I = 4
#Number of Continuous upper-level variables
n_R = 2
#Number of lower-level variables
n_y = 1
#Number of upper level constraints
m_u = 5
#Number of lower level constaints
m_l = 5

#Input data
np.random.seed(100)
H = np.random.normal(size=(n_I+n_R,n_I+n_R))
H = H.T@H
G_u = np.random.normal(size=(n_y,n_y))
G_u = G_u.T@G_u
G_l = np.random.normal(size=(n_y,n_y))
G_l = G_l.T@G_l
c = np.random.normal(loc = 0.5,size=n_I+n_R)
d_u = np.random.normal(loc = 0.5,size=n_y)
d_l = np.random.normal(loc = 0.5,size=n_y)

A = np.random.normal(loc=1,size=(m_u,n_I+n_R))
B = np.random.normal(loc=1,size=(m_u,n_y))
a = np.random.normal(loc = -0.5,size=m_u)

int_lb = low=np.zeros(n_I)
int_ub = np.random.randint(low = np.ones(n_I),high=np.repeat(4,n_I),size=n_I)

C = np.random.normal(loc=1,size=(m_l,n_I))
D = np.random.normal(loc=1,size=(m_l,n_y))
b = np.random.normal(loc = -0.5,size=m_l)

random1 = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]

#Number of Integer upper-level variables
n_I = 4
#Number of Continuous upper-level variables
n_R = 2
#Number of lower-level variables
n_y = 4
#Number of upper level constraints
m_u = 5
#Number of lower level constaints
m_l = 5

#Input data
np.random.seed(3)
H = np.random.normal(loc = 1,size=(n_I+n_R,n_I+n_R))
H = H.T@H
G_u = np.random.normal(loc = 1,size=(n_y,n_y))
G_u = G_u.T@G_u
G_l = np.random.normal(loc = 1,size=(n_y,n_y))
G_l = G_l.T@G_l
c = np.random.normal(loc = 0.5,size=n_I+n_R)
d_u = np.random.normal(loc = 0.5,size=n_y)
d_l = np.random.normal(loc = 0.5,size=n_y)

A = np.random.normal(loc=1,size=(m_u,n_I+n_R))
B = np.random.normal(loc=1,size=(m_u,n_y))
a = np.random.normal(loc = -100,size=m_u)

int_lb = low=np.zeros(n_I)
int_ub = np.random.randint(low = np.ones(n_I),high=np.repeat(10,n_I),size=n_I)

C = np.random.normal(loc=1,size=(m_l,n_I))
D = np.random.normal(loc=1,size=(m_l,n_y))
b = np.random.normal(loc = -100,size=m_l)

random2 = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]

#Number of Integer upper-level variables
n_I = 4
#Number of Continuous upper-level variables
n_R = 2
#Number of lower-level variables
n_y = 2
#Number of upper level constraints
m_u = 5
#Number of lower level constaints
m_l = 5

#Input data
np.random.seed(0)
H = np.random.normal(loc = 1,size=(n_I+n_R,n_I+n_R))
H = H.T@H
G_u = np.random.normal(loc = 1,size=(n_y,n_y))
G_u = G_u.T@G_u
G_l = np.random.normal(loc = 1,size=(n_y,n_y))
G_l = G_l.T@G_l
c = np.random.normal(loc = 0.5,size=n_I+n_R)
d_u = np.random.normal(loc = 0.5,size=n_y)
d_l = np.random.normal(loc = 0.5,size=n_y)

A = np.random.normal(loc=1,size=(m_u,n_I+n_R))
B = np.random.normal(loc=1,size=(m_u,n_y))
a = np.random.normal(loc = -1,size=m_u)

int_lb = low=np.zeros(n_I)
int_ub = np.random.randint(low = np.ones(n_I),high=np.repeat(10,n_I),size=n_I)

C = np.random.normal(loc=1,size=(m_l,n_I))
D = np.random.normal(loc=1,size=(m_l,n_y))
b = np.random.normal(loc = -1,size=m_l)

random3 = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]



#Number of Integer upper-level variables
n_I = 4
#Number of Continuous upper-level variables
n_R = 2
#Number of lower-level variables
n_y = 3
#Number of upper level constraints
m_u = 5
#Number of lower level constaints
m_l = 5

#Input data
np.random.seed(5)
H = np.random.normal(loc = 1,size=(n_I+n_R,n_I+n_R))
H = H.T@H
G_u = np.random.normal(loc = 1,size=(n_y,n_y))
G_u = G_u.T@G_u
G_l = np.random.normal(loc = 1,size=(n_y,n_y))
G_l = G_l.T@G_l
c = np.random.normal(loc = 0.5,size=n_I+n_R)
d_u = np.random.normal(loc = 0.5,size=n_y)
d_l = np.random.normal(loc = 0.5,size=n_y)

A = np.random.normal(loc=1,size=(m_u,n_I+n_R))
B = np.random.normal(loc=1,size=(m_u,n_y))
a = np.random.normal(loc = -1,size=m_u)

int_lb = low=np.zeros(n_I)
int_ub = np.random.randint(low = np.ones(n_I),high=np.repeat(10,n_I),size=n_I)

C = np.random.normal(loc=1,size=(m_l,n_I))
D = np.random.normal(loc=1,size=(m_l,n_y))
b = np.random.normal(loc = -1,size=m_l)

random4 = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]


problems = {}
problems['GumusFloudas2001Ex4'] = GumusFloudas2001Ex4
problems['ClarkWesterberg1990a'] = ClarkWesterberg1990a
problems['random1'] = random1
problems['random2'] = random2
problems['random3'] = random3
problems['random4'] = random4



def loadProblem(name):
    return problems[name]

def randomProblem(seed):
    #Number of Integer upper-level variables
    n_I = 4
    #Number of Continuous upper-level variables
    n_R = 2
    #Number of lower-level variables
    n_y = 3
    #Number of upper level constraints
    m_u = 5
    #Number of lower level constaints
    m_l = 5

    #Input data
    np.random.seed(seed)
    H = np.random.normal(loc = 1,size=(n_I+n_R,n_I+n_R))
    H = H.T@H
    G_u = np.random.normal(loc = 1,size=(n_y,n_y))
    G_u = G_u.T@G_u
    G_l = np.random.normal(loc = 1,size=(n_y,n_y))
    G_l = G_l.T@G_l
    c = np.random.normal(loc = 0.5,size=n_I+n_R)
    d_u = np.random.normal(loc = 0.5,size=n_y)
    d_l = np.random.normal(loc = 0.5,size=n_y)

    A = np.random.normal(loc=1,size=(m_u,n_I+n_R))
    B = np.random.normal(loc=1,size=(m_u,n_y))
    a = np.random.normal(loc = -1,size=m_u)

    int_lb = low=np.zeros(n_I)
    int_ub = np.random.randint(low = np.ones(n_I),high=np.repeat(10,n_I),size=n_I)

    C = np.random.normal(loc=1,size=(m_l,n_I))
    D = np.random.normal(loc=1,size=(m_l,n_y))
    b = np.random.normal(loc = -1,size=m_l)

    return [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]