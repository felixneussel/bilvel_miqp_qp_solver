from Solver_OOP.Problems import loadProblem, randomProblem
from Solver_OOP.miqp_qp_solver import MIQP_QP
import re
import numpy as np
from Parsers.file_reader import mps_aux_reader
from Data_Analysis.method_comparison import run_test
from Data_Analysis.method_comparison import create_dataframe

""" data = create_dataframe("MIPLIB_RESULTS/remark_2_results_15_min.txt")
for p in ["enigma-0.100000","enigma-0.500000","enigma-0.900000","lseu-0.900000","p0033-0.100000","p0201-0.900000","p0282-0.900000","stein45-0.100000"]:
    df = data[data["problem"]==p]
    print(df.sort_values(by="runtime")) """



#Paths of mps and aux file
""" mps_pa = '/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/stein45-0.900000.mps'
aux_pa = '/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/stein45-0.900000.aux'

n_I,n_R,n_y,m_u,m_l,c_u,d_u,A,B,a,int_lb,int_ub,d_l,C,D,b = mps_aux_reader(mps_pa,aux_pa)
#Input data
np.random.seed(3)
H = np.random.normal(loc = 1,size=(n_I+n_R,n_I+n_R))
H = H.T@H
G_u = np.random.normal(loc = 1,size=(n_y,n_y))
G_u = G_u.T@G_u
G_l = np.random.normal(loc = 1,size=(n_y,n_y))
G_l = G_l.T@G_l
 """
name = 'ClarkWesterberg1990a'
#name = 'GumusFloudas2001Ex4'
#name = 'random3'
p = loadProblem(name)
#print(p)
print('\n\n\n')
print()
print()
print(f'Results for {name}')
m = MIQP_QP(*p)
for f in ["MT"]:
    for mode in ['fixed_master']:
        if f == 'MT':
            m.solve(mode)
        elif f == 'ST':
            m.solve_ST(mode)
        print()
        print(f'method {f} sub_feas_creation_mode {mode}')
        print()
        #print('All variables')
        #print()
        #for key in m.solution:
            #print(key,'=', m.solution[key])
        #print()
        print('Variables of the Bilevel problem')
        print()
        for key in m.bilevel_solution:
            if re.match(r'^x',key):
                print(key,'=', m.solution[key])
        for key in m.bilevel_solution:
            if re.match(r'^y',key):
                print(key,'=', m.solution[key])
        print()

        print('Objective Function : ', m.UB)
        print()
        print('Runtime : ',m.runtime, 's')
        print()
        print('Iterations : ', m.iteration_counter)