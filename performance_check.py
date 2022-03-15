#
#This file contains functions to perform test runs.
#
from Solvers.file_reader import mps_aux_reader
import numpy as np
from numpy.linalg import norm
from Solvers.multitree import MT, solve
import traceback
from Solvers.benchmarks import optimize_benchmark




def shift_problem(problem_data,shift_by):
    """
    Shifts feasible set of problem away from zero by specified distance

    # Parameters

    - path: file path of npz file with problem data
    - shift_by: Distance to shift problem from zero
    """
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data

    constant = 0.5*np.array([shift_by]*n_I).T @ H @ np.array([shift_by]*n_I) + 0.5*np.array([shift_by]*n_y).T @ G_u @ np.array([shift_by]*n_y) - c_u.T@np.array([shift_by]*n_I) - d_u.T@np.array([shift_by]*n_y)
    a = a +  A @ np.array([shift_by]*n_I) + B @ np.array([shift_by]*n_y)
    b = b + C @ np.array([shift_by]*n_I) + D @ np.array([shift_by]*n_y)
    int_lb = int_lb + shift_by* np.ones(n_I)
    int_ub = int_ub + shift_by * np.ones(n_I)
    c_u = c_u - H@ np.array([shift_by]*n_I)
    d_u = d_u - G_u @ np.array([shift_by]*n_y)
    d_l = d_l - G_l @ np.array([shift_by]*n_y)

    
    return [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b], constant

def shift_x(problem_data,shift_by):
    """
    Shifts x variables of problem away from zero by specified distance

    # Parameters

    - path: file path of npz file with problem data
    - shift_by: Distance to shift problem from zero
    """
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data

    constant = 0.5*np.array([shift_by]*n_I).T @ H @ np.array([shift_by]*n_I)  - c_u.T@np.array([shift_by]*n_I)
    a = a +  A @ np.array([shift_by]*n_I) 
    b = b + C @ np.array([shift_by]*n_I) 
    int_lb = int_lb + shift_by* np.ones(n_I)
    int_ub = int_ub + shift_by * np.ones(n_I)
    c_u = c_u - H@ np.array([shift_by]*n_I)
    return [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b], constant

def getProblemData(DIRECTORY,problem):
    """
    Creates an MIQP-QP based on a MILP-LP specified by a .mps and .aux file and random quadratic terms.

    # Parameters

    - DIRECTORY : Path to the directory containing the mps and aux files as string.
    - problem : Name of the problem in the directory as string.
    """
    mps_pa = f"{DIRECTORY}/{problem}.mps"
    aux_pa = f"{DIRECTORY}/{problem}.aux"
    n_I,n_R,n_y,m_u,m_l,c_u,d_u,A,B,a,int_lb,int_ub,d_l,C,D,b = mps_aux_reader(mps_pa,aux_pa)
    H,G_u,G_l = quadraticTerms(n_I,n_R,n_y,c_u,d_u,d_l,n_I)
    return [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
     
def quadraticTerms(n_I,n_R,n_y,c_u,d_u,d_l,seed):
    """
    Generates random quadratic terms for an MIQP-QP.

    # Parameters

    - n_I : Number of linking variables.
    - n_R : Number of upper level continuous variables.
    - n_y : Number of lower level variables.
    - c_u : Upper-level objective function linear term for upper-level variables.
    - d_u : Upper-level objective function linear term for lower-level variables.
    - d_l : Lower-level objective function linear term for lower-level variables.
    - seed : Seed for the random number generator.
    """
    np.random.seed(seed)
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
    D = np.diag(np.random.uniform(low=1,high=max_l,size=n_y))
    G_l = G_l + D
    return H, G_u, G_l




def problem_data_from_npz(path):
    """
    Reads MIQP-QP problem data from an npz archive.

    # Parameters

    - path : Path of the npz archive.

    # Returns

    - List of problem dimensions and problem specific vectors and matrices.
    """
    problem = np.load(path)
    problem_data = [problem[key] for key in problem]
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    n_I = int(n_I)
    n_R = int(n_R)
    n_y = int(n_y)
    m_u = int(m_u)
    m_l = int(m_l)
    return [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]

def test_optimized_binary_expansion(names,SHIFTS,TIME_LIMIT,SUBPROBLEM_MODE,ALGORITHM,BIG_M,OUTPUT_FILE,num_of_seeds):
    """
    Solves different problems with different shifts away from the origin both with and without the optimized binary expansion and stores the results in a file.

    # Parameters

    - names : List of problem names.
    - SHIFTS : List of shifts.
    - TIME_LIMIT : Time limit for each solving process in seconds.
    - SUBPROBLEM_MODE : Method to aquire subproblem solutions. Can be set to 'regular', 'remark_1' and 'remark_2'. 
    It is recommended to use 'remark_2' if the matrix G_l is strictly positive definite or else 'regular'.
    - ALGORITHM : The solution approach that should be used. Can be set to 'MT', 'MT-K', 'MT-K-F', 'MT-K-F-W',
    'ST', 'ST-K', 'ST-K-C', 'ST-K-C-S'.
    - BIG_M : Big-M that the solver should use for upper and lower bounds.
    - OUTPUT_FILE : Path of the file where the results should be stored.
    - num_of_seeds : Number of different seeds and thus different random terms for each Problem.
    """
    for name in names:
        path = f"Problem_Data.nosync/{name}.npz"
        n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data_from_npz(path)
        for seed in range(num_of_seeds):
            H,G_u,G_l = quadraticTerms(n_I,n_R,n_y,c_u,d_u,d_l,seed)
            problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
            for s in SHIFTS:
                problem_data,constant = shift_x(problem_data,s)
                for opt_bin_exp in [True,False]:
                    print(f"Solving {name} shifted by {s}, opt_bin_exp : {opt_bin_exp}")
                    _,obj,runtime,times_in_sub,num_of_subs, status,gap = solve(problem_data,1e-5,np.infty,TIME_LIMIT,SUBPROBLEM_MODE,ALGORITHM,BIG_M,opt_bin_exp)
                    with open(OUTPUT_FILE,'a') as out:
                        out.write(f'name {name}__{seed} submode {SUBPROBLEM_MODE} algorithm {ALGORITHM} shift {s} opt_bin_exp {opt_bin_exp} status {status} obj {obj+constant} time {runtime} subtime {times_in_sub} subnum {num_of_subs} gap {gap}\n')
                    




def run_tests_final(problems,algos,submodes,file):
    """
    Solves different problems with different shifts away from the origin both with and without the optimized binary expansion and stores the results in a file.

    # Parameters

    - problems : List of problem names.
    - algos : List of solution approaches that should be tested. List entries be set to 'MT', 'MT-K', 'MT-K-F', 'MT-K-F-W',
    'ST', 'ST-K', 'ST-K-C', 'ST-K-C-S', 'KKT-MIQP' and 'SD-MIQCQP'.
    - submodes : List of methods to aquire subproblem solutions that should be tested. List entries can be set to 'regular', 'remark_1' and 'remark_2'. 
    - OUTPUT_FILE : Path of the file where the results should be stored.
    """
    for p in problems:
        path = f"Problem_Data.nosync/{p}.npz"
        problem_data = problem_data_from_npz(path)
        for a in algos:
            if a in ['KKT-MIQP','SD-MIQCQP']:
                print(f"Solving {p} with {a}.")
                try:
                    status,obj,runtime,gap = optimize_benchmark(a,CONTEXT["time_limt"],problem_data,CONTEXT["big_m"],CONTEXT["optimized_binary_expansion"])
                    with open(file,'a') as out:
                        out.write(f'algorithm {a} submode - name {p} status {status} time {runtime} obj {obj} gap {gap} subtime -1 subnum -1\n')
                except:
                    with open('test_exceptions.txt',"a") as out:
                        out.write(f"exception occured in name {p} algorithm {a}\n")
                        out.write(traceback.format_exc())
                        out.write("\n")
                        continue
                continue
            for s in submodes:
                print(f"Solving {p} with {a} and {s}.")
                try:
                    _,obj,runtime,times_in_sub,num_of_subs, status,gap = solve(problem_data,CONTEXT["tol"],CONTEXT["iteration_limit"],CONTEXT["time_limt"],s,a,CONTEXT["big_m"],CONTEXT["optimized_binary_expansion"])
                    with open(file,'a') as out:
                        out.write(f'algorithm {a} submode {s} name {p} status {status} time {runtime} obj {obj} gap {gap} subtime {times_in_sub} subnum {num_of_subs}\n')
                except:
                    with open('test_exceptions.txt',"a") as out:
                        out.write(f"exception occured in name {p} algorithm {a}\n")
                        out.write(traceback.format_exc())
                        out.write("\n")
                        continue



CONTEXT = {
    'tol':1e-5,
    'time_limt':300,
    'big_m':1e5,
    'iteration_limit':np.infty,
    'optimized_binary_expansion':True
}

FINAL_TEST_DATA = {
    'algos' : ["ST","ST-K","ST-K-C","ST-K-C-S","MT","MT-K","MT-K-F","MT-K-F-W","KKT-MIQP","SD-MIQCQP"],
    'submodes' : ['regular','remark_2'],
    'problems' : ["lseu-0.100000","enigma-0.900000","enigma-0.500000","stein45-0.100000","p0282-0.500000","stein27-0.100000","p0201-0.900000","p0033-0.100000","lseu-0.900000","stein45-0.500000","enigma-0.100000","p0033-0.500000","stein27-0.500000","stein27-0.900000","p0033-0.900000","lseu-0.500000","stein45-0.900000","p0282-0.900000"]
}

BINARY_EXP_TEST_DATA = {
    'problems':['stein27-0.100000','enigma-0.100000','stein27-0.900000']
}

REFACTOR_TEST_DATA = ["lseu-0.100000","enigma-0.900000","stein45-0.100000","stein27-0.100000","p0033-0.100000","lseu-0.900000","stein45-0.500000","enigma-0.100000","p0033-0.500000","stein27-0.500000","stein27-0.900000","p0033-0.900000","lseu-0.500000","stein45-0.900000"]

if __name__ == '__main__':
    run_tests_final(FINAL_TEST_DATA["problems"],FINAL_TEST_DATA["algos"],FINAL_TEST_DATA["submodes"],'Results/refactor_check_2.txt')