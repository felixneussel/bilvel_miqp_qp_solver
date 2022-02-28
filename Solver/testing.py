import masterproblem as mp
import numpy as np

#This is example Bard1988Ex1 from Bilevel Optimization, Chapter 19
#But an integer condition and bounds are added to the singel 
#upper-level variable and the constant term is removed from the objective

if __name__ == '__main__':
    #Dimensions
    #Number of Integer upper-level variables
    n_I = 1
    #Number of Continuous upper-level variables
    n_R = 0
    #Number of lower-level variables
    n_y = 1
    #Number of upper level constraints
    m_u = 1
    #Number of lower level constaints
    m_l = 4

    #Input data
    H = np.array([[2]])
    G = np.array([[8]])
    c = np.array([-10])
    d = np.array([4])

    A = np.array([[1]])
    B = np.array([[0]])
    a = np.array([0])

    int_lb = np.array([0])
    int_ub = np.array([20])

    C = np.array([[3],[-2],[-1],[0]])
    D = np.array([[-1],[0.5],[-1],[1]])
    b = np.array([3,-4,-7,0])

    Bard1988Ex1 = mp.setupMaster(n_I,n_R,n_y,m_u,m_l,H,G,c,d,A,B,a,int_lb,int_ub,C,D,b)
    Bard1988Ex1.optimize()
    for var in Bard1988Ex1.getVars():
        print(f'{var.varName} = {var.x}')

   
