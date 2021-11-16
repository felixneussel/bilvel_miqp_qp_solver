import gurobipy as gp
from gurobipy import GRB
import numpy as np
from matrix_operations import concatenateHorizontally
import re

class OptimizationModel:

    def __init__(self,n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b):
        self.n_I = n_I
        self.n_R = n_R
        self.n_y = n_y
        self.m_u = m_u
        self.m_l = m_l
        self.H = H
        self.G_u = G_u
        self.G_l = G_l
        self.c = c
        self.d_u = d_u
        self.d_l = d_l
        self.A = A
        self.B = B
        self.a = a
        self.int_lb = int_lb
        self.int_ub = int_ub
        self.C = C
        self.D = D
        self.b = b
        self.r_bar = (np.floor(np.log2(self.int_ub)) + np.ones(self.int_ub.shape)).astype(int)
        self.I = self.getIndexSet([self.n_I])
        self.R = self.getIndexSet([self.n_R])
        self.J = self.getIndexSet([self.n_y])
        self.ll_constr = self.getIndexSet([self.m_l])
        self.jr = self.getIndexSet([n_I,self.r_bar])
        self.setBinaryCoeffs()  
        self.checkDimensions()

    def checkDimensions(self):

        if self.H.shape != (self.n_I+self.n_R,self.n_I+self.n_R):
            raise ValueError('Dimension of H is not n_I+n_R x n_I+n_R.')
        elif self.G_u.shape != (self.n_y, self.n_y):
            raise ValueError('Dimension of G_u is not n_y x n_y.')
        elif self.G_l.shape != (self.n_y, self.n_y):
            raise ValueError('Dimension of G_l is not n_y x n_y.')
        elif self.c.shape != (self.n_I+self.n_R,):
            raise ValueError('Dimension of c is not n_I+n_R')
        elif self.d_u.shape != (self.n_y,):
            raise ValueError('Dimension of d_u is not n_y')
        elif self.d_l.shape != (self.n_y,):
            raise ValueError('Dimension of d_l is not n_y')
        elif self.A.shape != (self.m_u, self.n_I+self.n_R):
            raise ValueError('Dimension of A is not m_u x n_I+n_R')
        elif self.B.shape != (self.m_u, self.n_y):
            raise ValueError('Dimension of B is not m_u x n_y')
        elif self.a.shape != (self.m_u,):
            raise ValueError('Dimension of a is not m_u')
        elif self.int_lb.shape != (self.n_I,):
            raise ValueError('Dimension of int_lb is not n_I')
        elif self.int_ub.shape != (self.n_I,):
            raise ValueError('Dimension of int_ub is not n_I')
        elif self.C.shape != (self.m_l, self.n_I,):
            raise ValueError('Dimension of C is not m_l x n_I')
        elif self.D.shape != (self.m_l, self.n_y):
            raise ValueError('Dimension of D is not m_l x n_y')
        elif self.b.shape != (self.m_l,):
            raise ValueError('Dimension of b is not m_l')
        else:
            pass

    def getIndexSet(self,N):
        if len(N) == 1:
            return gp.tuplelist([a for a in range(N[0])])
        elif len(N) == 2:
            return gp.tuplelist([(a,b) for a in range(N[0]) for b in range(N[1][a])])
        else:
            raise ValueError('Can only create 1 or 2 dimensional index sets') 

    def addVariables(self):
        
        self.x_R = self.model.addVars(self.R, vtype=GRB.CONTINUOUS,name='x_R')
        self.y = self.model.addVars(self.J, vtype=GRB.CONTINUOUS,name='y')
        self.dual = self.model.addVars(self.ll_constr,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
        self.w = self.model.addVars(self.jr,name="w")

    def setDualFeasiblityConstraint(self):
        GD = concatenateHorizontally(self.D.T,-self.G_l)
        y_lambda = self.dual.select() + self.y.select()
        self.model.addMConstr(A=GD,x=y_lambda,sense='=',b=self.d_l)

    def printSolution(self):
        m1vars = self.model.getVars()
        for i in range(len(m1vars)):
            print(m1vars[i].varName, m1vars[i].x)

    def setBinaryCoeffs(self):
        self.bin_coeff = {}
        for (j,r) in self.jr:
            self.bin_coeff[(j,r)] = 2**r

    def optimize(self):
        self.model.optimize()
        print(self.model.ModelName)
        print('func : ', self.model.ObjVal)
        vars = self.model.getVars()
        for v in vars:
            if re.match(r'x_I|x_R|y', v.varName):
                print(v.varName, v.x)
        print('\n\n')

        
        return self.model.status,self.model.getVars(), self.model.ObjVal
        
        
