import os
import re
from Parsers.file_reader import mps_aux_reader
import numpy as np
from numpy.linalg import norm
from Functional.multitree import solve
import traceback
import concurrent.futures as futures
from datetime import datetime
from Benchmarks.KKT_MIQP import optimize_benchmark, setup_kkt_miqp

def getProblems(directory,solved_problems):
    files = os.listdir(directory)
    files = list(filter(lambda x: True if re.match(r'.*\.mps$',x) else False,files))
    files = list(map(lambda x: re.sub(r'.mps','',x),files))
    all_approaches = []
    for f in files:
        for algo in ['MT','MT-K','MT-K-F','MT-K-F-W','ST','ST-K','ST-K-C','ST-K-C-S']:
            all_approaches.append(f"{f} {algo}")
    solved = []
    with open(solved_problems,'r') as f:
        for line in f:
            solved.append(re.sub(r'\n','',line))
    unsolved = all_approaches[len(solved):]
    return list(unsolved)

def runPerformanceTest(DIRECTORY,PROBLEMS,TIME_LIMIT):
    for problem in PROBLEMS:
        problem, algo = problem.split()
        problem_data = getProblemData(DIRECTORY,problem)

def getProblemData(DIRECTORY,problem):
    mps_pa = f"{DIRECTORY}/{problem}.mps"
    aux_pa = f"{DIRECTORY}/{problem}.aux"
    n_I,n_R,n_y,m_u,m_l,c_u,d_u,A,B,a,int_lb,int_ub,d_l,C,D,b = mps_aux_reader(mps_pa,aux_pa)
    H,G_u,G_l = quadraticTerms(n_I,n_R,n_y,c_u,d_u,d_l)
    return [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
     
def quadraticTerms(n_I,n_R,n_y,c_u,d_u,d_l):
    np.random.seed(3)
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
    
def stop_process_pool(executor):
    for pid, process in executor._processes.items():
        process.terminate()
    executor.shutdown()

def MIPLIB():
    res = []
    for name in ["enigma-0.100000 MT","enigma-0.500000 MT","enigma-0.900000 MT","lseu-0.900000 MT","p0033-0.100000 MT","p0201-0.900000 MT","p0282-0.900000 MT","stein45-0.100000 MT"]:
        for algo in ['MT-K','MT-K-F','MT-K-F-W','ST','ST-K','ST-K-C','ST-K-C-S']:
            name = name.split()[0]
            res.append(f"{name} {algo}")
    return res

def Test_Run(description):
    DIRECTORY = '/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv'
    SOLVED_FILE = f"MIPLIB_RESULTS/{description}_solved.txt"
    PROBLEMS_TO_SOLVE = getProblems(DIRECTORY, SOLVED_FILE)
    TIME_LIMIT = 300
    SUBPROBLEM_MODE = "remark_2"
    OUTPUT_FILE = f"MIPLIB_RESULTS/{description}_results.txt"
    EXCEPTION_REPORT = f"MIPLIB_RESULTS/{description}_exceptions.txt"
    BIG_M = 1e5
    with open(OUTPUT_FILE,"a") as out:
        out.write(f"\nRun on {datetime.now()}\n")
    with open(EXCEPTION_REPORT,"a") as out:
        out.write(f"\nRun on {datetime.now()}\n")
    for problem in PROBLEMS_TO_SOLVE:
        name, algorithm = problem.split()
        problem_data = getProblemData(DIRECTORY,name)
        n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
        print(f"Trying to solve {name} with {algorithm}")
        try:
            solution,obj,runtime,times_in_sub,num_of_subs, status,gap = solve(problem_data,1e-5,np.infty,TIME_LIMIT,SUBPROBLEM_MODE,algorithm,BIG_M)
        except Exception:
            with open(EXCEPTION_REPORT,"a") as out:
                out.write(f"exception occured in name {name} submode {SUBPROBLEM_MODE} algorithm {algorithm}\n")
                out.write(traceback.format_exc())
                out.write("\n")
            continue
        with open(OUTPUT_FILE,'a') as out:
            out.write(f'name {name} n_I {n_I} n_R {n_R} n_y {n_y} m_u {m_u} m_l {m_l} submode {SUBPROBLEM_MODE} algorithm {algorithm} status {status} obj {obj} time {runtime} subtime {times_in_sub} subnum {num_of_subs} gap {gap}\n')
        with open(SOLVED_FILE,"a") as out:
            out.write(f"{name} {algorithm}\n")

def benchmarking():
    DIRECTORY = '/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv'
    PROBLEMS_TO_SOLVE = ["enigma-0.100000","enigma-0.500000","enigma-0.900000","lseu-0.900000","p0033-0.100000","p0201-0.900000","p0282-0.900000","stein45-0.100000"]
    TIME_LIMIT = 300
    OUTPUT_FILE = "MIPLIB_RESULTS/kkt_miqp_results.txt"
    EXCEPTION_REPORT = "MIPLIB_RESULTS/kkt_miqp_exceptions.txt"
    APPROACH = "KKT_MIQP"
    if APPROACH == "KKT_MIQP":
        ALGORITHM = setup_kkt_miqp
    BIG_M = 1e5
    with open(OUTPUT_FILE,"a") as out:
        out.write(f"\nRun on {datetime.now()}\n")
    with open(EXCEPTION_REPORT,"a") as out:
        out.write(f"\nRun on {datetime.now()}\n")
    for name in PROBLEMS_TO_SOLVE:
        problem_data = getProblemData(DIRECTORY,name)
        model = ALGORITHM(problem_data,BIG_M)
        try:
            status,ObjVal,Runtime,MIPGap = optimize_benchmark(model,TIME_LIMIT)
            with open(OUTPUT_FILE,"a") as out:
                out.write(f"name {name} algorithm {APPROACH} status {status} obj {ObjVal} time {Runtime} gap {MIPGap}\n")
        except Exception:
            with open(EXCEPTION_REPORT,"a") as out:
                    out.write(f"exception occured in name {name} algorithm {APPROACH}\n")
                    out.write(traceback.format_exc())
                    out.write("\n")



if __name__ == '__main__':
    Test_Run("rem_2_5_min")
    
            

            
         

   


    

    