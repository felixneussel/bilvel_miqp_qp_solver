import gurobipy as gp
from gurobipy import GRB
from gurobipy import QuadExpr,MQuadExpr
import numpy as np
from Solver_OOP.models import OptimizationModel
from Solver_OOP.matrix_operations import concatenateDiagonally, concatenateHorizontally, getUpperBound, getLowerBound
from Solver_OOP.lower_level import Lower_Level
import re

class Sub(OptimizationModel):
    
    def __init__(self,mp,n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b,x_I_param,s_param,cut_counter,mode):
        super().__init__(n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b)
        self.cut_counter = cut_counter
        self.mode = mode
        if self.mode == 'new':
            self.x_I_param = x_I_param
            self.s_param = s_param
            self.model = gp.Model('Subproblem')
            self.model.Params.LogToConsole = 0
            self.addVariables()
            self.setObjective()
            self.setPConstraint()
            self.setDualFeasiblityConstraint()
            self.setStrongDualityLinearizationConstraint()
            self.setStrongDualityConstraint(mp.y.select(),mp.dual.select(),mp.w.select())
        elif self.mode == 'fixed_master':
            self.model = mp.model.fixed()
            self.removeMasterLinearizations()
            self.removeBinaryExpansion()
            self.setStrongDualityConstraint(mp.y.select(),mp.dual.select(),mp.w.select())
        elif self.mode == 'remark_1':
            self.lower = Lower_Level(self.n_y,self.m_l,self.G_l,self.d_l,self.C,self.D,self.b,self.x_I_param)
            self.lower.optimize()
            self.ll_obj,self.ll_vars = self.lower.ObjVal, self.lower.solution
            self.model = gp.Model('Subproblem')
            self.model.Params.LogToConsole = 0
            self.addVariables()
            self.setObjective()
            self.setPConstraint()
            self.setLowerLevelOptimalityConstraint()
        else:
            raise ValueError('Subproblem creation mode is not new or fixed_master or remark_1')

    def setLowerLevelOptimalityConstraint(self):
        self.model.addMQConstr(Q = self.G_l/2, c = self.d_l, sense="<", rhs=self.ll_obj, xQ_L=self.y.select(), xQ_R=self.y.select(), xc=self.y.select(), name="Lower Level Optimality" )
     
    def setObjective(self):
        #Slice H into quadrants corresponding to terms with x_I, x_R or and x_I - x_R-mixed-term
        H_II = self.H[:self.n_I,:self.n_I]
        H_RR = self.H[self.n_I:,self.n_I:]
        H_IR = self.H[:self.n_I,self.n_I:]
        #slice c into vectors corresponding to x_I and x_R
        c_I = self.c[:self.n_I]
        c_R = self.c[self.n_I:]

        quad_matrix = concatenateDiagonally(H_RR,self.G_u)
        lin_vec = np.concatenate((c_R.T+self.x_I_param.T@H_IR,self.d_u.T)).T
        constant_term = 0.5*self.x_I_param@H_II@self.x_I_param + c_I@self.x_I_param
        vars = self.x_R.select() + self.y.select()
        self.model.setMObjective(Q=quad_matrix/2,c=lin_vec,constant=constant_term,xQ_L=vars,xQ_R=vars,xc=vars,sense=GRB.MINIMIZE)

    def setPConstraint(self):
        A_I = self.A[:,:self.n_I]
        A_R = self.A[:,self.n_I:]
        AB = concatenateHorizontally(A_R,self.B)
        primalvars = self.x_R.select() + self.y.select()
        self.model.addMConstr(A=AB,x=primalvars,sense='>=',b=self.a-A_I@self.x_I_param)
        self.model.addMConstr(A=self.D,x=self.y.select(),sense='>=',b=self.b - self.C@self.x_I_param)
        self.model.update()
   
    def setStrongDualityLinearizationConstraint(self):
        self.model.addConstrs((self.w[j,r] == self.s_param[j,r]*sum([self.C[i,j]*self.dual[i] for i in self.ll_constr]) for j,r in self.jr), 'binary_expansion')
    

    def setStrongDualityConstraint(self,y_var,dual_var,w_var):
        """
        twoyTG = 2*point.T @ G
        yTGy = point.T @ G @ point
        term1 = sum([twoyTG[i]*y_var[i] for i in y_ind])
        term2 = sum([d[i]*y[i] for i in y_ind])
        term3 = -sum([b[j]*dual_var[j] for j in dual_ind])
        term4 = w_var.prod(binary_coeff)
        model.addConstr((term1+term2+term3+term4-yTGy <= 0),'Strong duality relaxation')
        """
        #Not sure if this works
        #term1 = QuadExpr(self.y@self.G@self.y)#sum(sum([self.y[i]*self.G[i,j]*self.y[j] for i in self.J for j in self.J]))
        #term2 = QuadExpr(self.d@self.y)#sum([self.d[i]*self.y[i] for i in self.J])
        #term3 = QuadExpr(-self.b@self.dual)#-sum([self.b[i]*self.dual[i] for i in self.ll_constr])
        #term4 = self.w.prod(self.bin_coeff)
        #self.model.addQConstr((term2+term3+term4 <= 0),'Strong duality constraint')

        #Doesn't produce an error but I'm suspicious if it really works
        #Indeed: Does only work if dim(y) = 1
        """ expr = QuadExpr()
        expr.addTerms(self.G_l,self.y.select(),self.y.select())
        expr.addTerms(self.d_l,self.y.select())
        expr.addTerms(-self.b,self.dual.select()) """
        
        #expr = self.y.select @ self.G_l @self.y.select() #+ self.d_l@self.y.select() - self.b@self.dual.select()
        #self.model.addQConstr((expr + self.w.prod(self.bin_coeff) <= 0),'Strong Duality Constraint')
        if self.mode == 'new':
            linear_vector = np.concatenate((self.d_l, - self.b, self.bin_coeff_vec))
            y_lam_w = self.y.select() + self.dual.select() + self.w.select()
            self.model.addMQConstr(Q = self.G_l, c = linear_vector, sense="<", rhs=0, xQ_L=self.y.select(), xQ_R=self.y.select(), xc=y_lam_w, name="Strong Duality Constraint" )
        else:
            linear_vector = np.concatenate((self.d_l, - self.b, self.bin_coeff_vec))
            y_lam_w = y_var + dual_var + w_var
            self.model.addMQConstr(Q = self.G_l, c = linear_vector, sense="<", rhs=0, xQ_L=y_var, xQ_R=y_var, xc=y_lam_w, name="Strong Duality Constraint" )

    def removeMasterLinearizations(self):
        
        constraints = self.model.getConstrs()
        for i in range(self.cut_counter):
            self.model.remove(constraints.pop())

    def removeBinaryExpansion(self):
        constr = self.model.getConstrs()
        filtered_cons = list(filter(lambda c: re.match(r'^binary expansion',c.ConstrName) is not None,constr))
        for con in filtered_cons:
            self.model.remove(con)

     
    """ def removeMasterLinearizations(self):
        constraints = self.model.getConstrs()
        #filter(lambda c: True if c.ConstrName == 'Strong Duality Linearization' else False, constraints)
        constr_remover = lambda c: self.model.remove(c) if c.ConstrName == 'Strong duality linearization' else 0
        map(constr_remover,constraints) """
            
        
    
    def getPointForMasterCut(self):
        name_exp = re.compile(r'^y')
        x = []
        for var in self.model.getVars():
            if name_exp.match(var.varName) is not None:
                x.append(var.x)
        return np.array(x) 

    

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

    s = sub(n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b,x_I_param,s_param)
    s.optimize()
    #s.setPConstraint()