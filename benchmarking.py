from Functional.multitree import MT,ST, solve
from Parsers.file_reader import mps_aux_reader
import numpy as np
import re
from numpy import infty
from gurobipy import GRB
from numpy.linalg import norm
from Solver_OOP.Problems import loadProblem
from Benchmarks.KKT_MIQP import setup_kkt_miqp

if __name__ == '__main__':
    big_M = 1e5
    #Paths of mps and aux file
    name = "stein45-0.100000"
    mps_pa = f'/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/{name}.mps'
    aux_pa = f'/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/{name}.aux'

    

    #n_I,n_R,n_y,m_u,m_l,c_u,d_u,A,B,a,int_lb,int_ub,d_l,C,D,b = mps_aux_reader(mps_pa,aux_pa)
    
    """ np.random.seed(3)
    sigma_u = max(norm(c_u,np.infty),norm(d_u,np.infty))
    sigma_l = norm(d_l,np.infty)
    H = np.random.uniform(low=-np.sqrt(sigma_u),high = np.sqrt(sigma_u),size=(n_I+n_R,n_I+n_R))
    H = H.T@H
    G_u = np.random.uniform(low=-np.sqrt(sigma_u),high = np.sqrt(sigma_u),size=(n_y,n_y))
    G_u = G_u.T@G_u
    G_l = np.random.uniform(low=-np.sqrt(sigma_u),high = np.sqrt(sigma_l),size=(n_y,n_y))
    G_l = G_l.T@G_l 
    addtodiag = np.diag(np.random.uniform(low=1,high=np.sqrt(sigma_l),size=n_y))
    G_l = G_l + addtodiag """

    name = "ClarkWesterberg1990a"
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = loadProblem(name)
    problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
    model = setup_kkt_miqp(problem_data,big_M)
    model.optimize()
    for v in model.getVars():
        print(f"{v.varName} = {v.x}")