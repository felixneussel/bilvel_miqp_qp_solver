import os
import re
from Parsers.file_reader import mps_aux_reader
import numpy as np
from numpy.linalg import norm
from Functional.multitree import solve
import traceback

def getProblems(directory):
    files = os.listdir(directory)
    files = list(filter(lambda x: True if re.match(r'.*\.mps$',x) else False,files))
    files = list(map(lambda x: re.sub(r'.mps','',x),files))
    all_approaches = set([])
    for f in files:
        for algo in ['MT','MT-K','MT-K-F','MT-K-F-W','ST','ST-K','ST-K-C','ST-K-C-S']:
            all_approaches.add(f"{f} {algo}")
    solved = set([])
    with open('/Users/felixneussel/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Vertiefung/Bachelorarbeit/Implementierung/MIQP_QP_Solver/Results/solved.txt','r') as f:
        for line in f:
            solved.add(re.sub(r'\n','',line))
    unsolved = all_approaches - solved
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
    H = np.random.uniform(low=-np.sqrt(sigma_u),high = np.sqrt(sigma_u),size=(n_I+n_R,n_I+n_R))
    H = H.T@H
    G_u = np.random.uniform(low=-np.sqrt(sigma_u),high = np.sqrt(sigma_u),size=(n_y,n_y))
    G_u = G_u.T@G_u
    G_l = np.random.uniform(low=-np.sqrt(sigma_u),high = np.sqrt(sigma_l),size=(n_y,n_y))
    G_l = G_l.T@G_l 
    D = np.diag(np.random.uniform(low=1,high=np.sqrt(sigma_l),size=n_y))
    G_l = G_l + D
    return H, G_u, G_l

if __name__ == '__main__':
    DIRECTORY = '/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv'
    PROBLEMS_TO_SOLVE = getProblems(DIRECTORY)
    TIME_LIMIT = 10
    SUBPROBLEM_MODE = "remark_2"
    OUTPUT_FILE = "Results/remark_2_test.txt"
    EXCEPTION_REPORT = "Results/remark_2_exceptions.txt"
    for problem in PROBLEMS_TO_SOLVE:
        name, algorithm = problem.split()
        problem_data = getProblemData(DIRECTORY,name)
        n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
        try:
            solution,obj,runtime,time_in_subs, status= solve(problem_data,1e-5,np.infty,TIME_LIMIT,SUBPROBLEM_MODE,algorithm)
            with open(OUTPUT_FILE,'a') as out:
                out.write(f'name {name} n_I {n_I} n_R {n_R} n_y {n_y} m_u {m_u} m_l {m_l} submode {SUBPROBLEM_MODE} algorithm {algorithm} status {status} solution ')
                for key in solution:
                    if re.match(r'x|y',key):
                        out.write(f'{key} {solution[key]} ')
                out.write(f'obj {obj} time {runtime} subtime {time_in_subs}\n')
        except Exception:
            with open(EXCEPTION_REPORT,"a") as out:
                out.write(f"exception occured in name {name} submode {SUBPROBLEM_MODE} algorithm {algorithm}\n")
                out.write(traceback.format_exc())
                out.write("\n")
         

   


    

    