#
#This file can be used to solve a specific MIQP-QP.
#
from Solvers.multitree import solve
from Solvers.file_reader import mps_aux_reader
import numpy as np
from numpy import infty
import re


if __name__ == "__main__":
    #Number of Integer upper-level variables
    n_I = 2
    #Number of Continuous upper-level variables
    n_R = 0
    #Number of lower-level variables
    n_y = 1
    #Number of upper level constraints
    m_u = 2
    #Number of lower level constaints
    m_l = 2

    #Objective matrices
    H = np.array([[2,0],[0,2]])
    G_u = np.array([[2]])
    G_l = np.array([[2]])

    #objective vectors
    c_u = np.array([0,0])
    d_u = np.array([0])
    d_l = np.array([-1.142])


    #Leader Constraints
    A = np.array([[1,0],[0,1]])
    B = np.array([[-0.459],[-0.297]])
    a = np.array([0,0])

    int_lb = np.array([0,2])
    int_ub = np.array([1,6])


    #Follower Constraints
    C = np.array([[0,0],[0,0]])
    D = np.array([[1],[-1]])
    b = np.array([0,-1])

    input_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]

    #Specifications for the algorithm
    tolerance = 1e-5
    iteration_limit = infty
    time_limit = 300
    subproblem_mode = "regular"
    algorithm = "ST-K-C-S"  
    big_m = 1e5
    optimized_binary_expansion = True

    #Solve the problem
    solution,obj,runtime,times_in_sub,num_of_subs, status,gap = solve(input_data,tolerance,infty,infty,subproblem_mode,algorithm,big_m,optimized_binary_expansion)

    if status == 2:
        print('Solution')
        print()
        for key in solution:
            if re.match(r'^x',key):
                print(key,'=', solution[key])
        for key in solution:
            if re.match(r'^y',key):
                print(key,'=', solution[key])
        print()
        print('Objective Function : ', obj)
        print('Runtime : ',runtime, 's')
    else:
        print('Problem infeasible')


    data = mps_aux_reader('/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/stein45-0.100000.mps','/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/stein45-0.100000.aux')
    