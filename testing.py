import re
import numpy as np
from Solvers.file_reader import mps_aux_reader
from gurobipy import GRB
from numpy import infty
from Solvers.multitree import solve
import pandas as pd
import matplotlib.pyplot as plt
from Solvers.benchmarks import optimize_benchmark




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




def test_binary_optimization(n):
    

    #Number of Integer upper-level variables
    n_I = 100
    #Number of Continuous upper-level variables
    n_R = 0
    #Number of lower-level variables
    n_y = 1
    #Number of upper level constraints
    m_u = 0
    #Number of lower level constaints
    m_l = 2

    #Input data
    np.random.seed(200)
    c_u = np.random.uniform(-10,10,n_I)
    d_u = np.random.uniform(-0,0,n_y)
    d_l = np.random.uniform(-0,0,n_y)
    H = np.random.normal(0,0.25,(n_I,n_I))
    H = H.T@H
    H = H + np.diag(np.random.uniform(0.5,2,n_I))
    G_u = 2*np.diag(np.random.uniform(0.1,0.5,n_y))
    G_l = 2*np.diag(np.random.uniform(0.1,0.5,n_y))
    

    A = np.zeros((m_u,n_I+n_R))
    B = np.zeros((m_u,n_y))
    a = np.zeros(m_u)

    C = np.random.uniform(0,1,(m_l,n_I))
    D = np.array([[1],[-1]])
    b = np.array([-5,-5])

    lower_bounds = []
    runtimes_normal = []
    runtimes_optimized = []
    obj_normal = []
    obj_optimized = []
    status_normal = []
    status_optimized = []
    for i in range(n):
        int_lb = np.array([2**i -1]*n_I)
        int_ub = np.array([2**i]*n_I)
        lower_bounds.append(int_lb[0])
        problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
        _,obj,runtime,_,_, status,_ = solve(problem_data,1e-5,infty,300,'remark_2','ST-K-C-S',1e5,False)
        runtimes_normal.append(runtime)
        obj_normal.append(obj)
        status_normal.append(status)
        _,obj,runtime,_,_, status,_  = solve(problem_data,1e-5,infty,300,'remark_2','ST-K-C-S',1e5,True)
        runtimes_optimized.append(runtime)
        obj_optimized.append(obj)
        status_optimized.append(status)
    df = pd.DataFrame({"lower_bound":lower_bounds,"runtimes_normal":runtimes_normal,"runtimes_optimized":runtimes_optimized,"obj_normal":obj_normal,"obj_optimized":obj_optimized})
    df = pd.DataFrame({("Lower Bound",""):lower_bounds,("Normal","Status"):status_normal,("Normal","Time"):runtimes_normal,("Normal","Objective"):obj_normal,
    ("Optimized","Status"):status_optimized, ("Optimized","Time"):runtimes_optimized,("Optimized","Objective"):obj_optimized})
    df.set_index("Lower Bound",inplace=True)
    df.to_pickle("MIPLIB_RESULTS/Testing/Bin_exp/Random_problem_multi_index.pkl")
    table = df.to_latex()
    with open("MIPLIB_RESULTS/Latex/Tables.txt","a") as out:
        out.write(f"Binary_Expansion\n")
        out.write(table)
        out.write("\n")
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

def test_reduction():
    UB_I = 50
    name = "p0548-0.500000"
    path = f"Problem_Data.nosync/Sub_1000_Vars/{name}.npz"
    problem = np.load(f"Problem_Data.nosync/Sub_1000_Vars/{name}.npz")
    problem_data = [problem[key] for key in problem]
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    n_I = int(n_I)
    n_R = int(n_R)
    n_y = int(n_y)
    m_u = int(m_u)
    m_l = int(m_l)
    seed = n_I
    np.random.seed(seed)
    if n_I > UB_I:
        n_I = 100
    n_x = n_I + n_R

    c_u = c_u[:n_x]
    A = A[:m_u,:n_x]
    int_lb = int_lb[:n_I]
    int_ub = int_ub[:n_I]
    C = C[:m_l,:n_I]
    D = D[:m_l,:n_y]
    b = b[:m_l]
    H = H[:n_x,:n_x]
    problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
    #status, _,_,_ = optimize_benchmark("KKT-MIQP",10,problem_data,1e5,True)
    _,obj,runtime,_,_, status,_  = solve(problem_data,1e-5,infty,300,'remark_2','ST-K',1e5,True)
    print(status)
    





if __name__ == "__main__":
    test_binary_optimization(5)
    df = pd.read_pickle("MIPLIB_RESULTS/Testing/Bin_exp/Random_problem_multi_index.pkl")
    print(df)
    