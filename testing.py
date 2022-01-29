from Solver_OOP.Problems import loadProblem, randomProblem
from Solver_OOP.miqp_qp_solver import MIQP_QP
import re
import numpy as np
from Parsers.file_reader import mps_aux_reader
from Data_Analysis.method_comparison import run_test
from Data_Analysis.method_comparison import create_dataframe
from gurobipy import GRB
from numpy import infty
from Functional.multitree import solve
import pandas as pd
import matplotlib.pyplot as plt
from Functional.benchmarks import optimize_benchmark



def make_problem():
    #Number of Integer upper-level variables
    n_I = 3
    #Number of Continuous upper-level variables
    n_R = 0
    #Number of lower-level variables
    n_y = 1
    #Number of upper level constraints
    m_u = 0
    #Number of lower level constaints
    m_l = 2

    #Input data
    H = 2*np.eye(3)
    G_u = 2*np.eye(1)
    G_l = 2*np.eye(1)
    c_u = np.array([0,0,0])
    d_u = np.array([0])
    d_l = np.array([-2])

    A = np.zeros((m_u,n_I+n_R))
    B = np.zeros((m_u,n_y))
    a = np.zeros(m_u)

    int_lb = np.array([3,-1,-4])
    int_ub = np.array([4,5,-1])

    C = np.zeros((m_l,n_I))
    D = np.array([[1],[-1]])
    b = np.array([-5,-5])
    save_to = f"Problem_Data.nosync/Non_Binary/non_bin_1.npz"
    np.savez(save_to,n_I=n_I,n_R=n_R,n_y=n_y,m_u=m_u,m_l=m_l,H=H,G_u=G_u,G_l=G_l,c_u=c_u,d_u=d_u,d_l=d_l,A=A,B=B,a=a,int_lb=int_lb,int_ub=int_ub,C=C,D=D,b=b)

def solve_problem(path):
    problem = np.load(path)
    problem_data = [problem[key] for key in problem]
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    n_I = int(n_I)
    n_R = int(n_R)
    n_y = int(n_y)
    m_u = int(m_u)
    m_l = int(m_l)
    problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
    solution,obj,runtime,times_in_sub,num_of_subs, status,gap = solve(problem_data,1e-5,infty,300,'remark_1','ST-K-C-S',1e5,True)

    if status in [2,GRB.TIME_LIMIT]:
        print()
        print(path)
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

def shift_problem(path,shift_by):
    """
    Shifts feasible set of problem away from zero by specified distance

    Input:

    path: file path of npz file with problem data

    shift_by: Distance to shift problem from zero
    """
    problem = np.load(path)
    problem_data = [problem[key] for key in problem]
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    n_I = int(n_I)
    n_R = int(n_R)
    n_y = int(n_y)
    m_u = int(m_u)
    m_l = int(m_l)
    problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
    a = a +  A @ np.array([shift_by]*n_I)
    b = b + C @ np.array([shift_by]*n_I) + D @ np.array([shift_by]*n_y)
    int_lb = int_lb + shift_by* np.ones(n_I)
    int_ub = int_ub + shift_by * np.ones(n_I)
    return n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b

def test_binary_optimization(n):

    #Number of Integer upper-level variables
    n_I = 1000
    #Number of Continuous upper-level variables
    n_R = 0
    #Number of lower-level variables
    n_y = 1
    #Number of upper level constraints
    m_u = 0
    #Number of lower level constaints
    m_l = 2

    #Input data
    H = 2*np.eye(n_I)
    G_u = 2*np.eye(1)
    G_l = 2*np.eye(1)
    c_u = np.zeros(n_I)
    d_u = np.array([0])
    d_l = np.array([-2])

    A = np.zeros((m_u,n_I+n_R))
    B = np.zeros((m_u,n_y))
    a = np.zeros(m_u)

    C = np.zeros((m_l,n_I))
    D = np.array([[1],[-1]])
    b = np.array([-5,-5])

    lower_bounds = []
    runtimes_normal = []
    runtimes_optimized = []
    obj_normal = []
    obj_optimized = []
    for i in range(n):
        int_lb = np.array([2**i -1]*n_I)
        int_ub = np.array([2**i]*n_I)
        lower_bounds.append(int_lb[0])
        problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
        _,obj,runtime,_,_, _,_ = solve(problem_data,1e-5,infty,300,'remark_2','ST-K-C-S',1e5,False)
        runtimes_normal.append(runtime)
        obj_normal.append(obj)
        _,obj,runtime,_,_, _,_  = solve(problem_data,1e-5,infty,300,'remark_2','ST-K-C-S',1e5,True)
        runtimes_optimized.append(runtime)
        obj_optimized.append(obj)
    df = pd.DataFrame({"lower_bound":lower_bounds,"runtimes_normal":runtimes_normal,"runtimes_optimized":runtimes_optimized,"obj_normal":obj_normal,"obj_optimized":obj_optimized})
    print(df)
    plt.plot(lower_bounds,runtimes_normal,label="Normal")
    plt.plot(lower_bounds,runtimes_optimized,label="Optimized")
    plt.xlabel("Lower Bound")
    plt.ylabel("Runtimes")
    plt.legend()
    plt.show()

def test_benchmark():
    name = "enigma-0.100000"
    problem = np.load(f"Problem_Data.nosync/Sub_1000_Vars/{name}.npz")
    problem_data = [problem[key] for key in problem]
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    n_I = int(n_I)
    n_R = int(n_R)
    n_y = int(n_y)
    m_u = int(m_u)
    m_l = int(m_l)
    problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
    status, obj, time, gap = optimize_benchmark("SD-MIQCQP",10,problem_data,1e5,True)
    print(f"Status : {status}")
    print(f"Obj : {obj}")
    print(f"Time : {time}")
    print(f"Gap : {gap}")


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
""" name = 'ClarkWesterberg1990a'
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
        print('Iterations : ', m.iteration_counter) """

if __name__ == "__main__":
    """ data = create_dataframe("MIPLIB_RESULTS/remark_2_results_15_min.txt")
    for p in ["enigma-0.100000","enigma-0.500000","enigma-0.900000","lseu-0.900000","p0033-0.100000","p0201-0.900000","p0282-0.900000","stein45-0.100000"]:
        df = data[data["problem"]==p]
        print(df.sort_values(by="runtime")) """
    test_benchmark()
        

    