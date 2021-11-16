import gurobipy as gp
from gurobipy import GRB
from gurobipy import QuadExpr
import numpy as np
from models import OptimizationModel
from masterproblem import Master
from subproblem import Sub
from feasibility_problem import Feas
from matrix_operations import concatenateDiagonally, concatenateHorizontally, getUpperBound, getLowerBound
import re


class MIQP_QP():
    def __init__(self,n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b):
        
        self.problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
        self.LB = -np.infty
        self.UB = np.infty
        self.master = Master(n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b)

    def solve(self):
        while self.LB < self.UB:
            print('LB : ', self.LB, 'UB: ', self.UB)
            print('\n\n\n\n')
            status,vars,val = self.master.optimize()
            if status != GRB.OPTIMAL:
                return ('The bilevel problem is infeasible')
            else:
                self.LB = val
            x_I_p = self.master.getParamX_IForSub()
            s_p = self.master.getParamSForSub()
            self.feas = Feas(*self.problem_data,x_I_p,s_p)
            status,vars,val = self.feas.optimize()
            if val <= 0:#subproblem feasible
                self.sub = Sub(*self.problem_data,x_I_p,s_p)
                status,vars,val = self.sub.optimize()
                if val < self.UB:
                    self.current_sln = vars
                    self.UB = val
            name_exp = re.compile(r'^y')
            cp = []
            for var in vars:
                if name_exp.match(var.varName) is not None:
                    cp.append(var.x)
            self.master.setCuttingPoint(np.array(cp))
            self.master.addCut()
        return self.current_sln



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
    G_u = np.array([[8]])
    G_l = np.array([[1]])
    c = np.array([-10])
    d_u = np.array([4])
    d_l = np.array([1])

    A = np.array([[1]])
    B = np.array([[0]])
    a = np.array([0])

    int_lb = np.array([0])
    int_ub = np.array([20])

    C = np.array([[3],[-2],[-1],[0]])
    D = np.array([[-1],[0.5],[-1],[1]])
    b = np.array([3,-4,-7,0])

    m = MIQP_QP(n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b)
    print(m.solve())