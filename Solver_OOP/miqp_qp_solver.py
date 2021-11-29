#from typing import OrderedDict
import gurobipy as gp
from gurobipy import GRB
#from gurobipy import QuadExpr
import numpy as np
#from models import OptimizationModel
from masterproblem import Master
from subproblem import Sub
from feasibility_problem import Feas
from st_master import SingleTree
#from matrix_operations import concatenateDiagonally, concatenateHorizontally, getUpperBound, getLowerBound
import re
#from collections import OrderedDict
import timeit


class MIQP_QP():
    def __init__(self,n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b):
        
        self.problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
        self.LB = -np.infty
        self.UB = np.infty
        self.master = Master(*self.problem_data)
        
        self.tol = 1e-5
        self.solution = {}

    def solve(self):
        start = timeit.default_timer()
        self.iteration_counter = 0
        while self.LB + self.tol < self.UB:
            #print('LB : ', self.LB, 'UB: ', self.UB)
            #print('\n\n\n\n')

            #Solve Masterproblem
            self.master.optimize()
            m_status,m_vars,m_val = self.master.status,self.master.solution,self.master.ObjVal
            
            if m_status != GRB.OPTIMAL:
                return ('The bilevel problem is infeasible')
            else:
                self.LB = m_val
            #Retrieve parameters for Sub or Feasiblity Problem from Masterproblem
            x_I_p = self.master.getParamX_IForSub()
            s_p = self.master.getParamSForSub()
            #Solve Subproblem
            print(self.master.y)
            self.sub = Sub(*self.problem_data,x_I_p,s_p)
            
            self.sub.optimize()
            s_status,s_vars,s_val = self.sub.status,self.sub.solution,self.sub.ObjVal
            next_cut = s_vars
            if s_status == GRB.OPTIMAL:#subproblem feasible
                #self.sub = Sub(*self.problem_data,x_I_p,s_p)
                #s_status,s_vars,s_val = self.sub.optimize()
                #next_cut = s_vars
                if s_val < self.UB:
                    for v in s_vars:
                        self.solution[v.varName] = v.x
                    for v in m_vars:
                        if re.match(r'x|s',v.varName) is not None:
                            self.solution[v.varName] = v.x
                    self.UB = s_val
            else:#Subproblem infeasible
                self.feas = Feas(*self.problem_data,x_I_p,s_p)
                self.feas.optimize()
                f_status,f_vars,f_val = self.feas.status,self.feas.solution,self.feas.ObjVal
                next_cut = f_vars

            #Add Linearization of Strong Duality Constraint at solution of sub or feasibility
            #problem as constraint to masterproblem
            name_exp = re.compile(r'^y')
            cp = []
            for var in next_cut:
                if name_exp.match(var.varName) is not None:
                    cp.append(var.x)
            
            self.master.addCut(np.array(cp))
            self.master.model.update()
            self.iteration_counter += 1
        stop = timeit.default_timer()
        self.runtime = stop - start
        self.getBilevelSolution()
        return self.solution

    def solve_ST(self):
        start = timeit.default_timer()
        self.UB = np.infty
        self.iteration_counter = 0
        #self.l = self.int_lb
        #self.u = self.int_ub
        self.z_star = None
        self.O = [SingleTree(*self.problem_data)]
        while self.O:# and self.iteration_counter: #<8:
            """ print(f'O = {self.O}')
            print(f'upper bound = {self.UB}')
            print() """
            N_p = self.O.pop()
            N_p.optimize()
            m_status,m_vars,m_val = N_p.status,N_p.solution,N_p.ObjVal
            if m_status != GRB.OPTIMAL or m_val >= self.UB - self.tol:
                continue
            elif N_p.is_int_feasible() and N_p.ObjVal < self.UB:
                """ print(f'Int vars : {N_p.int_vars}')
                print(f'Integer feasiblity: {N_p.is_int_feasible()}')
                print(f'obj val = {N_p.ObjVal}')
                print(f'UB = {self.UB}')
                print() """

                #Retrieve parameters for Sub or Feasiblity Problem from Masterproblem
                x_I_p = N_p.getParamX_IForSub()
                s_p = N_p.getParamSForSub()
                #Solve Subproblem
                
                #self.sub = Sub(*self.problem_data,x_I_p,s_p)
                self.sub = self.master.model.FixedModel()
                self.sub.optimize()
                s_status,s_vars,s_val = self.sub.status,self.sub.solution,self.sub.ObjVal
                next_cut = s_vars
                if s_status == GRB.OPTIMAL:
                    if s_val < self.UB:#subproblem feasible
                        for v in s_vars:
                            self.solution[v.varName] = v.x
                        for v in m_vars:
                            if re.match(r'x|s',v.varName) is not None:
                                self.solution[v.varName] = v.x
                        self.UB = s_val
                else:#Subproblem infeasible
                  
                    self.feas = Feas(*self.problem_data,x_I_p,s_p)
                    self.feas.optimize()
                    f_status,f_vars,f_val = self.feas.status,self.feas.solution,self.feas.ObjVal
                    next_cut = f_vars
                self.O.append(N_p)
                name_exp = re.compile(r'^y')
                cp = []
                for var in next_cut:
                    if name_exp.match(var.varName) is not None:
                        cp.append(var.x)
                
                for pro in self.O:
                    pro.addCut(np.array(cp))
                    pro.model.update()

            else:
                first = N_p
                first.addUpperBoun()
                second = N_p
                second.addLowerBound()
                #print(f'O before : {self.O}')
                self.O.append(first)
                self.O.append(second)
                """ print(f'O after : {self.O}')
                print() """
            
            self.iteration_counter += 1
        stop = timeit.default_timer()
        self.runtime = stop-start
        self.getBilevelSolution()

    def getBilevelSolution(self):
        self.bilevel_solution = {}
        name_exp = re.compile(r'^x|^y')
        for key in self.solution:
            if name_exp.match(key) is not None:
                self.bilevel_solution[key] = self.solution[key]



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
    sln = m.solve()
    print(sln)



    