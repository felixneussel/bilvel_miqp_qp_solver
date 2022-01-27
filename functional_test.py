import timeit
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
    
    #Paths of mps and aux file
    name = "p0201-0.900000"
    mps_pa = f'/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/{name}.mps'
    aux_pa = f'/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/{name}.aux'

    n_I,n_R,n_y,m_u,m_l,c_u,d_u,A,B,a,int_lb,int_ub,d_l,C,D,b = mps_aux_reader(mps_pa,aux_pa)
    #int_ub[:10] = np.random.randint(2,7,size=10)
    #Input data
    np.random.seed(n_I)
    sigma_u = max(norm(c_u,np.infty),norm(d_u,np.infty))
    sigma_l = norm(d_l,np.infty)
    max_u = np.power(sigma_u,1/4)
    max_l = np.power(sigma_l,1/4)

    H = np.random.uniform(low=-max_u,high = max_u,size=(n_I+n_R,n_I+n_R))
    H = H.T@H
    G_u = np.random.uniform(low=-max_u,high = max_u,size=(n_y,n_y))
    G_u = G_u.T@G_u
    G_l = np.random.uniform(low=-max_l,high = max_l,size=(n_y,n_y))
    G_l = G_l.T@G_l 
    addtodiag = np.diag(np.random.uniform(low=1,high=max_l,size=n_y))
    G_l = G_l + addtodiag

    #name = "ClarkWesterberg1990a"
    #n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = loadProblem(name)
    problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
    start = timeit.default_timer()
    solution,obj,runtime,times_in_sub,num_of_subs, status,gap = solve(problem_data,1e-5,infty,300,'remark_2','MT',1e5,True)
    print(f"Time : {timeit.default_timer() - start}")
    if status in [2,GRB.TIME_LIMIT]:
        print()
        print(name)
        print(f"Status : {status}")
        """ print('All variables')
        print()
        for key in solution:
            print(key,'=', solution[key])
        print() """
        print()
        print('Variables of the Bilevel problem')
        print()
        for key in solution:
            if re.match(r'^x',key):
                print(key,'=', solution[key])
        for key in solution:
            if re.match(r'^y',key):
                print(key,'=', solution[key])
        print()

        print('Objective Function : ', obj)
        # print()
        print('Runtime : ',runtime, 's')
        print(f"Num of subproblems : {num_of_subs}")
        print(f"Time in Subproblems : {times_in_sub}")
        print(f"Gap : {gap}")
    else:
        print('Problem infeasible')
    
