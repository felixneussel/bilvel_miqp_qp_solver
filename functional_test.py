from Functional.multitree import MT,ST,MT_rem_1, MT_rem_2
from Parsers.file_reader import mps_aux_reader
import numpy as np
import re

if __name__ == '__main__':
    #Paths of mps and aux file
    mps_pa = '/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/stein45-0.100000.mps'
    aux_pa = '/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/stein45-0.100000.aux'

    n_I,n_R,n_y,m_u,m_l,c_u,d_u,A,B,a,int_lb,int_ub,d_l,C,D,b = mps_aux_reader(mps_pa,aux_pa)
    #Input data
    np.random.seed(3)
    H = np.random.normal(loc = 1,size=(n_I+n_R,n_I+n_R))
    H = H.T@H
    G_u = np.random.normal(loc = 1,size=(n_y,n_y))
    G_u = G_u.T@G_u
    G_l = np.random.normal(loc = 1,size=(n_y,n_y))
    G_l = G_l.T@G_l

    problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]

    solution,obj,runtime, status = MT_rem_2(problem_data,1e-5)
   
    if status == 2:
        """ print('All variables')
        print()
        for key in solution:
            print(key,'=', solution[key])
        print() """
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
        print()
        print('Runtime : ',runtime, 's')
    else:
        print('Problem infeasible')
    
