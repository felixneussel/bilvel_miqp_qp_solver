import gurobipy as gp
from gurobipy import GRB
from gurobipy import QuadExpr
import numpy as np
from models import OptimizationModel
from matrix_operations import concatenateDiagonally, concatenateHorizontally, getUpperBound, getLowerBound

class Feas(OptimizationModel):
    
    def __init__(self,n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b,x_I_param,s_param):
        super().__init__(n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b)
        self.x_I_param = x_I_param
        self.s_param = s_param
        self.model = gp.Model('Feasiblity-Problem')
        self.model.Params.LogToConsole = 0
        self.addVariables()
        self.setObjective()
        self.setPConstraint()
        self.setDualFeasiblityConstraint()
        self.setStrongDualityLinearizationConstraint()

    def setObjective(self):
        """ expr = QuadExpr()
        expr.addTerms(self.G_l,self.y.select(),self.y.select())
        expr.addTerms(self.d_l,self.y.select())
        expr.addTerms(-self.b,self.dual.select())
        self.model.setObjective(expr + self.w.prod(self.bin_coeff),sense=GRB.MINIMIZE) """

        linear_vector = np.concatenate((self.d_l, - self.b, self.bin_coeff_vec))
        y_lam_w = self.y.select() + self.dual.select() + self.w.select()
        #elf.model.addMQConstr(Q = self.G_l, c = linear_vector, sense="<", rhs=0, xQ_L=self.y.select(), xQ_R=self.y.select(), xc=y_lam_w, name="Strong Duality Constraint" )
        self.model.setMObjective(Q=self.G_l,c=linear_vector,constant=0,xQ_L=self.y.select(),xQ_R=self.y.select(),xc=y_lam_w,sense=GRB.MINIMIZE)

    def setPConstraint(self):
        A_I = self.A[:,:self.n_I]
        A_R = self.A[:,self.n_I:]
        AB = concatenateHorizontally(A_R,self.B)
        primalvars = self.x_R.select() + self.y.select()
        self.model.addMConstr(A=AB,x=primalvars,sense='>=',b=self.a-A_I@self.x_I_param)

    def setStrongDualityLinearizationConstraint(self):
        self.model.addConstrs((self.w[j,r] == self.s_param[j,r]*sum([self.C[i,j]*self.dual[i] for i in self.ll_constr]) for j,r in self.jr), 'binary_expansion')

    

if __name__ == '__main__':
    #Dimensions
    #Number of Integer upper-level variables
    n_I = 1
    #Number of Continuous upper-level variables
    n_R = 1
    #Number of lower-level variables
    n_y = 1
    #Number of upper level constraints
    m_u = 1
    #Number of lower level constaints
    m_l = 4

    #Input data
    H = np.array([[2,0],[0,2]])
    G_u = np.array([[8]])
    G_l = np.array([[1]])
    c = np.array([-10,0])
    d_u = np.array([4])
    d_l = np.array([1])
    A = np.array([[1,2]])
    B = np.array([[0]])
    a = np.array([0])

    int_lb = np.array([0])
    int_ub = np.array([20])

    C = np.array([[3],[-2],[-1],[0]])
    D = np.array([[-1],[0.5],[-1],[1]])
    b = np.array([3,-4,-7,0])

    x_I_param = np.array([5])

    s_param = {(0, 0): 0.0, (0, 1): 1.0, (0, 2): 0.0, (0, 3): 0.0, (0, 4): 0.0}

    s = feas(n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b,x_I_param,s_param)
    s.optimize()