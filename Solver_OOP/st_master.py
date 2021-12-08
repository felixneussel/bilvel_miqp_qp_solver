import gurobipy as gp
from gurobipy import GRB
import numpy as np
from Solver_OOP.models import OptimizationModel
from Solver_OOP.matrix_operations import concatenateDiagonally, concatenateHorizontally, getUpperBound, getLowerBound
import re

class SingleTree(OptimizationModel):
    
    def __init__(self,n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b):
        super().__init__(n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b)
        self.model = gp.Model('SingleTree')
        self.model.Params.LogToConsole = 0
        self.addVariables()
        self.setObjective()
        self.setPConstraint()
        super().setDualFeasiblityConstraint()
        self.setStrongDualityLinearizationConstraint()

    def addVariables(self):
        super().addVariables()
        self.x_I = self.model.addVars(self.I, vtype=GRB.CONTINUOUS,lb=self.int_lb, ub=self.int_ub,name='x_I')
        self.s = self.model.addVars(self.jr,vtype= GRB.BINARY,name='s')

    def setObjective(self):
        HG = concatenateDiagonally(self.H,self.G_u)
        cd = np.concatenate((self.c,self.d_u))
        primalvars = self.x_I.select() + self.x_R.select() + self.y.select()

        self.model.setMObjective(Q=HG/2,c=cd,constant=0.0,xQ_L=primalvars,xQ_R=primalvars,xc=primalvars,sense=GRB.MINIMIZE)
        self.model.update()

    def setPConstraint(self):
        AB = concatenateHorizontally(self.A,self.B)
        primalvars = self.x_I.select() + self.x_R.select() + self.y.select()
        self.model.addMConstr(A=AB,x=primalvars,sense='>=',b=self.a)

        CD = concatenateHorizontally(self.C,self.D)
        lower_level_vars = self.x_I.select() + self.y.select()
        self.model.addMConstr(A=CD,x=lower_level_vars,sense='>=',b=self.b)
        self.model.update()


    def setStrongDualityLinearizationConstraint(self):
        
        #Note, since our r indices start at zero, we write 2**r instead of 2**(r-1)
        #master.addConstrs((sum(2**r*s[j,r] for r in jr[j,'*']) == x_I[j] for j in I),'binary')
        self.model.addConstrs((self.s.prod(self.bin_coeff,j,'*') == self.x_I[j] for j,r in self.jr),'binary')

        ub = getUpperBound()
        lb = getLowerBound()
        self.model.addConstrs((self.w[j,r] <= ub*self.s[j,r] for j,r in self.jr),'13a')
        self.model.addConstrs((self.w[j,r] <= sum([self.C[i,j]*self.dual[i] for i in self.ll_constr]) + lb*(self.s[j,r] - 1) for j,r in self.jr),'13b')
        #Possible refactor: replace lam_coeff with C and get rid of lam_coeff

        self.model.addConstrs((self.w[j,r] >= lb*self.s[j,r] for j,r in self.jr),'13c')
        self.model.addConstrs((self.w[j,r] >= sum([self.C[(i,j)]*self.dual[i] for i in self.ll_constr]) + ub*(self.s[j,r] - 1) for j,r in self.jr),'13d')

    def getParamSForSub(self):
        name_exp = re.compile(r'^s')
        index_exp = re.compile(r'(?<=\[)\d+(?=,)|(?<=,)\d+(?=\])')
        s = {}
        for var in self.model.getVars():
            if name_exp.match(var.varName) is not None:
                indices = list(map(int,index_exp.findall(var.varName)))
                if len(indices) != 2:
                    raise ValueError('Regex did not find exactly two indices')
                s[indices[0],indices[1]] = var.x
        return s

    def getParamX_IForSub(self):
        name_exp = re.compile(r'^x_I')
        x = []
        for var in self.model.getVars():
            if name_exp.match(var.varName) is not None:
                x.append(var.x)
        return np.array(x)

    

    def addCut(self,p):
        """
        Takes a point \bar{y}, linearizes strong duality constraint at this point and adds the constraint to the model.
        """
        self.cut_point = p
        twoyTG = 2*self.cut_point.T @ self.G_l
        yTGy = self.cut_point.T @ self.G_l @ self.cut_point
        term1 = sum([twoyTG[i]*self.y.select(i)[0] for i in self.J])
        term2 = sum([self.d_l[i]*self.y.select(i)[0] for i in self.J])
        term3 = -sum([self.b[j]*self.dual.select(j)[0] for j in self.ll_constr])
        term4 = self.w.prod(self.bin_coeff)
        
        self.model.addConstr((term1+term2+term3+term4-yTGy <= 0),'Strong duality relaxation')

    def addBounds(self,l,u):
        """
        Bounds for Single-Tree
        """
        A = np.zeros((self.n_I,self.n_I))
       
        if l is not None:
            self.model.addMConstr(A=A,x=self.x_I.select(),sense = '>=',b=l)
        if u is not None:
            self.model.addMConstr(A=A,x=self.x_I.select(),sense = '<=',b=u)

    def getBranchIndex(self):
        #Branch on longest box axis whose upper bound is not integer
        differences = self.int_ub - self.int_lb
        max_index = 0
        for i,dif in enumerate(differences):
            if dif > differences[max_index] and self.int_ub[i] - int(self.int_ub) !=0:
                max_index = i
        return i

    def addUpperBound(self):
        self.int_ub[self.getBranchIndex]=np.floor(self.int_ub[self.getBranchIndex])
        self.x_I.setAttr('ub',self.int_ub)

    def addLowerBound(self):
        self.int_lb[self.getBranchIndex]=np.ceil(self.int_lb[self.getBranchIndex])
        self.x_I.setAttr('lb',self.int_lb)

    def is_int_feasible(self):
        int_vars = []
        for v in self.solution:
            if re.match(r'^x_I',v.varName):
                int_vars.append(v.x)
        
        int_vars_casted = list(map(int,int_vars))
        self.int_vars = np.array(int_vars)
        #print('truth array',self.int_vars - np.array(int_vars_casted) == np.zeros(self.n_I))
        if all(self.int_vars - np.array(int_vars_casted) == np.zeros(self.n_I)):
            return True
        else:
            return False




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
    """ m = Master(n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b)
    m.setCuttingPoint(np.array([3]))
    m.optimize()
    m.addCut()
    print(m.model.getVars()) """