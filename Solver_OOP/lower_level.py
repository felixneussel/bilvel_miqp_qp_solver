import gurobipy as gp
from gurobipy import GRB
import numpy as np
from Solver_OOP.matrix_operations import concatenateHorizontally
from Solver_OOP.models import OptimizationModel

class Lower_Level():
    def __init__(self,n_y,m_l,G_l,d_l,C,D,b,x_I_param):
        self.n_y = n_y
        self.m_l = m_l
        self.G_l = G_l
        self.d_l = d_l
        self.C = C
        self.D = D 
        self.b = b
        self.x_I_param = x_I_param
        self.model = gp.Model('Lower_Level')
        self.y = self.model.addMVar(shape=n_y,vtype = GRB.CONTINUOUS,name = 'y')
        self.model.setMObjective(Q=self.G_l/2, c = self.d_l, constant=None, xQ_L=self.y, xQ_R=self.y, xc=self.y, sense=GRB.MINIMIZE )
        self.model.addMConstr(A=self.D, x=self.y, sense='>', b=self.b - self.C@self.x_I_param, name="Lower Level Constraints" )

    def optimize(self):
        self.model.optimize()
        self.status = self.model.status
        if self.status == GRB.OPTIMAL:
            self.ObjVal = self.model.ObjVal
            self.solution = self.model.getVars().copy()
        else:
            self.ObjVal = None
            self.solution = None