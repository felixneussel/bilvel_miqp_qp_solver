import os
import re
import timeit
from venv import create
from Parsers.file_reader import mps_aux_reader
import numpy as np
from numpy.linalg import norm
from Functional.multitree import solve
import traceback
import concurrent.futures as futures
from datetime import datetime
from Benchmarks.KKT_MIQP import optimize_benchmark, setup_kkt_miqp
import signal
import pandas as pd
from Data_Analysis.performance_profiles import create_dataframe

def signal_handler(signum, frame):
    raise Exception("Timed out!")

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
    H,G_u,G_l = quadraticTerms(n_I,n_R,n_y,c_u,d_u,d_l,n_I)
    return [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
     
def quadraticTerms(n_I,n_R,n_y,c_u,d_u,d_l,seed):
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

def prepareQuadraticTerms(in_dir,out_dir):
    files = os.listdir(in_dir)
    files = list(filter(lambda x: True if re.match(r'.*\.mps$',x) else False,files))
    names = list(map(lambda x: re.sub(r'.mps','',x),files))
    for i,name in enumerate(names):
        if name in ["seymour-0.500000","fast0507-0.900000","nw04-0.900000","nw04-0.100000","nw04-0.500000","fast0507-0.500000"]:
            continue
        print(f"Name {i} : {name}")
        mps_pa = f"{in_dir}/{name}.mps"
        aux_pa = f"{in_dir}/{name}.aux"
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(10)
        try:
            n_I,n_R,n_y,m_u,m_l,c_u,d_u,A,B,a,int_lb,int_ub,d_l,C,D,b = mps_aux_reader(mps_pa,aux_pa)
        except Exception:
            print("Timed out!")
            continue
        #save_to = f"{out_dir}/{f}.npz"
        with open(f"{out_dir}/dimensions.txt","a") as out:
            out.write(f"name {name} n_I {n_I} n_R {n_R} n_y {n_y} m_u {m_u} m_l {m_l}\n")
        print(f"{name} read")
        #np.savez(save_to,H=H,G_u=G_u,G_l=G_l)

def analyze_data(file):
    df = create_dataframe(file,["name" , "n_I" , "n_R" , "n_y" , "m_u" , "m_l" ],[str,int,int,int,int,int])
    df = df.sort_values("n_I")
    return df
    
def make_data_set(df,in_dir,out_dir):
    #names = df.loc[27:,"name"]
    names = ["air03-0.500000","harp2-0.100000"]
    names = ["enigma-0.100000"]
    for i,name in enumerate(names):
        print(f"Name {i} : {name}")
        n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = getProblemData(in_dir,name,i)
        save_to = f"{out_dir}/{name}.npz"
        np.savez(save_to,n_I=n_I,n_R=n_R,n_y=n_y,m_u=m_u,m_l=m_l,H=H,G_u=G_u,G_l=G_l,c_u=c_u,d_u=d_u,d_l=d_l,A=A,B=B,a=a,int_lb=int_lb,int_ub=int_ub,C=C,D=D,b=b)
        print(f"{name} succesful")

def reduce(dir,name,n_I,n_R,n_y,m_u,m_l):
    problem = np.load(f"Problem_Data/{name}.npz")
    n_x = n_I + n_R
    H = problem["H"][:n_x,:n_x]
    G_u = problem["G_u"][:n_y,:n_y]
    G_l = problem["G_l"][:n_y,:n_y]
    c_u = problem["c_u"][:n_x]
    d_u = problem["d_u"][:n_y]
    d_l = problem["d_l"][:n_y]
    A = problem["A"][:m_u,:n_x]
    B = problem["B"][:m_u,:n_y]
    a = problem["a"][:m_u]
    int_lb = problem["int_lb"][:m_u]
    int_ub = problem["int_ub"][:m_u]
    C = problem["C"][:m_l,:n_I]
    D = problem["D"][:m_l,:n_y]
    b = problem["b"][:m_l]
    save_to = f"{dir}/{name}_reduced.npz"
    np.savez(save_to,n_I=n_I,n_R=n_R,n_y=n_y,m_u=m_u,m_l=m_l,H=H,G_u=G_u,G_l=G_l,c_u=c_u,d_u=d_u,d_l=d_l,A=A,B=B,a=a,int_lb=int_lb,int_ub=int_ub,C=C,D=D,b=b)

    
def reduction():
    path = "Problem_Data"
    files = os.listdir(path)
    files = list(filter(lambda x: x != "dimensions.txt",files))
    UB_I, UB_y, UB_m = 150, 250, 500
    for name in files:
        problem = np.load(f"{path}/{name}.npz")
        n_I,n_R,n_y,m_u,m_l = problem["n_I"], problem["n_R"], problem["n_y"], problem["m_u"], problem["m_l"]
        if n_I <= UB_I and n_y <= UB_y and m_l <= UB_m:
            continue
        np.random.seed(n_I)
        if n_I > UB_I:
            n_I = np.random.randint(50,UB_I)
        if n_y > UB_y:
            n_y = np.random.randint(50,UB_y)
        if m_l > UB_m:
            m_l = np.random.randint(100,UB_m)
        reduce(path,name,n_I,n_R,n_y,m_u,m_l)

def create_reduced_data(name):
    UB_I, UB_y, UB_m = 150, 250, 1000
    in_dir = "/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv"
    out_dir = "Problem_Data.nosync"
    mps_pa = f"{in_dir}/{name}.mps"
    aux_pa = f"{in_dir}/{name}.aux"
    n_I,n_R,n_y,m_u,m_l,c_u,d_u,A,B,a,int_lb,int_ub,d_l,C,D,b = mps_aux_reader(mps_pa,aux_pa)
    reduced = False
    appendix = ""
    seed = n_I
    np.random.seed(seed)
    if n_I > UB_I:
        n_I = np.random.randint(50,UB_I)
        reduced = True
    if n_y > UB_y:
        n_y = np.random.randint(50,UB_y)
        reduced = True
    if m_l > UB_m:
        m_l = np.random.randint(100,UB_m)
        reduced = True
    n_x = n_I + n_R

    if reduced:
        c_u = c_u[:n_x]
        d_u = d_u[:n_y]
        d_l = d_l[:n_y]
        A = A[:m_u,:n_x]
        B = B[:m_u,:n_y]
        a = a[:m_u]
        int_lb = int_lb[:n_I]
        int_ub = int_ub[:n_I]
        C = C[:m_l,:n_I]
        D = D[:m_l,:n_y]
        b = b[:m_l]
        appendix = "_reduced"

    H,G_u,G_l = quadraticTerms(n_I,n_R,n_y,c_u,d_u,d_l,seed)
    save_to = f"{out_dir}/{name}{appendix}.npz"
    np.savez(save_to,n_I=n_I,n_R=n_R,n_y=n_y,m_u=m_u,m_l=m_l,H=H,G_u=G_u,G_l=G_l,c_u=c_u,d_u=d_u,d_l=d_l,A=A,B=B,a=a,int_lb=int_lb,int_ub=int_ub,C=C,D=D,b=b)
        
    
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

def Test_Run_npz(out_dir,PROBLEMS_TO_SOLVE,ALGORITHM,SUBPROBLEM_MODE):
    DIRECTORY = "Problem_Data.nosync/Sub_1000_Vars"
    #SOLVED_FILE = f"MIPLIB_RESULTS/{description}_solved.txt"
    #PROBLEMS_TO_SOLVE = set(os.listdir(DIRECTORY)) - {"lseu-0.100000.npz","enigma-0.900000.npz","enigma-0.500000.npz","stein45-0.100000.npz","p0282-0.500000.npz","stein27-0.100000.npz","p0201-0.900000.npz","p0033-0.100000.npz","lseu-0.900000.npz","stein45-0.500000.npz","enigma-0.100000.npz","p0033-0.500000.npz","stein27-0.500000.npz","stein27-0.900000.npz","p0033-0.900000.npz","lseu-0.500000.npz","stein45-0.900000.npz"}
    TIME_LIMIT = 300
    #SUBPROBLEM_MODE = "remark_2"
    OUTPUT_FILE = f"{out_dir}_results.txt"
    EXCEPTION_REPORT = f"{out_dir}_exceptions.txt"
    BIG_M = 1e5
    #ALGORITHM = "ST-K-C-S"
    with open(OUTPUT_FILE,"a") as out:
        out.write(f"Run algorithm {ALGORITHM} submode {SUBPROBLEM_MODE} on {datetime.now()}\n")
    with open(EXCEPTION_REPORT,"a") as out:
        out.write(f"Run on {datetime.now()}\n")
    for file in PROBLEMS_TO_SOLVE:
        problem = np.load(f"{DIRECTORY}/{file}.npz")
        name = re.sub(r'.npz','',file)
        problem_data = [problem[key] for key in problem]
        n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
        n_I = int(n_I)
        n_R = int(n_R)
        n_y = int(n_y)
        m_u = int(m_u)
        m_l = int(m_l)
        problem_data = [n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b]
        print(f"Trying to solve {name} with {ALGORITHM} and submode {SUBPROBLEM_MODE}")
        start = timeit.default_timer()
        try:
            solution,obj,runtime,times_in_sub,num_of_subs, status,gap = solve(problem_data,1e-5,np.infty,TIME_LIMIT,SUBPROBLEM_MODE,ALGORITHM,BIG_M,True)
        except Exception:
            with open(EXCEPTION_REPORT,"a") as out:
                out.write(f"exception occured in name {name} submode {SUBPROBLEM_MODE} algorithm {ALGORITHM}\n")
                out.write(traceback.format_exc())
                out.write("\n")
            continue
        with open(OUTPUT_FILE,'a') as out:
            out.write(f'name {name} n_I {n_I} n_R {n_R} n_y {n_y} m_u {m_u} m_l {m_l} submode {SUBPROBLEM_MODE} algorithm {ALGORITHM} status {status} obj {obj} time {runtime} subtime {times_in_sub} subnum {num_of_subs} gap {gap}\n')
        #with open(SOLVED_FILE,"a") as out:
        #    out.write(f"{name} {algorithm}\n")
        print(f"Time : {timeit.default_timer() - start}")

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

def create_reduced_data_set():
    df = create_dataframe("dimensions.txt",["name","n_I","n_R","n_y","m_u","m_l"],[str,int,int,int,int,int])
    for name in df.loc[:,"name"]:
        create_reduced_data(name)

def create_thirty_dataset():
    df = create_dataframe("Problem_Data.nosync/dimensions.txt",["name","n_I","n_R","n_y","m_u","m_l"],[str,int,int,int,int,int]).sort_values("n_I")
    created = os.listdir("Problem_Data.nosync/Sub_1000_Vars")
    for i,name in enumerate(df.loc[:,"name"]):
        if i > 29 or f"{name}.npz" in created:
            continue
        n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = getProblemData("/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv",name)
        save_to = f"Problem_Data.nosync/Sub_1000_Vars/{name}.npz"
        np.savez(save_to,n_I=n_I,n_R=n_R,n_y=n_y,m_u=m_u,m_l=m_l,H=H,G_u=G_u,G_l=G_l,c_u=c_u,d_u=d_u,d_l=d_l,A=A,B=B,a=a,int_lb=int_lb,int_ub=int_ub,C=C,D=D,b=b)

def testing(DESCRIPTION,ALGORITHMS,SUBPROBLEMS,PROBLEMS):
    TEST_SET = create_dataframe("MIPLIB_RESULTS/Test_Set_Results",["name","obj"],[str,float])[0]
    DIRECTORY = f"MIPLIB_RESULTS/Testing/{DESCRIPTION}"
    TOL = 1e-5
    for algo in ALGORITHMS:
        for subproblem in SUBPROBLEMS:
            Test_Run_npz(DIRECTORY,PROBLEMS,algo,subproblem)
    RESULTS = create_dataframe(f"{DIRECTORY}_results.txt",["name","submode","algorithm","obj"],[str,str,str,float])

    for res in RESULTS:
        res = reduced_df(res)
        TEST_SET = TEST_SET.merge(res,left_on="name",right_on="name")

    VIOLATION = {"name":[],"method":[],"deviation":[]}
    for _,row in TEST_SET.iterrows():
        benchmark = row["obj"]
        for algo, sub in [(a,s) for a in ALGORITHMS for s in SUBMODE]:
            obj = row[f"{algo} {sub}"]
            if  abs(obj - benchmark) > TOL:
                VIOLATION["name"].append(row["name"])
                VIOLATION["method"].append(f"{algo} {sub}")
                VIOLATION["deviation"].append(abs(obj - benchmark))

    print()
    print("Test Results")
    print()
    print(TEST_SET)
    print()
    if VIOLATION["name"] == []:
        print("All algorithms passed")
    else:
        num_vio = len(VIOLATION["name"])
        max_vio = max(VIOLATION["deviation"])
        max_problem = VIOLATION["name"][np.argmax(VIOLATION["deviation"])]
        max_method = VIOLATION["method"][np.argmax(VIOLATION["deviation"])]
        print(f"{num_vio} failed with maximal deviation of {max_vio} while solving {max_problem} with {max_method}")
        print()
        print(pd.DataFrame(VIOLATION))
    return TEST_SET

def reduced_df(df):
    algo = df.loc[0,"algorithm"]
    sub = df.loc[0,"submode"]
    label = f"{algo} {sub}"
    df = df.rename(columns={"obj":label})
    df = df.drop(["submode","algorithm"],axis=1)
    return df    


if __name__ == '__main__':
    DESCRIPTION = "MT_second_measure"
    ALGORITHMS = ["MT"]
    SUBMODE = ["remark_2"]
    SMALL_SET = ["lseu-0.100000","stein45-0.900000","stein27-0.900000","stein45-0.500000"]
    BIG_SET = ["lseu-0.100000","enigma-0.900000","enigma-0.500000","stein45-0.100000","p0282-0.500000","stein27-0.100000","p0201-0.900000","p0033-0.100000","lseu-0.900000","stein45-0.500000","enigma-0.100000","p0033-0.500000","stein27-0.500000","stein27-0.900000","p0033-0.900000","lseu-0.500000","stein45-0.900000"]
    FAST = ["lseu-0.100000","stein45-0.900000","stein27-0.900000"]
    HARD_PROBLEMS = []
    TIMELIMIT_PROBLEM = ["p0201-0.900000"]
    ####

    #REMEMBER TO CHOOSE AN EMPTY LOG FILE!

    ######
    start = timeit.default_timer()
    df = testing(DESCRIPTION,ALGORITHMS,SUBMODE,HARD_PROBLEMS)
    print(f"Time : {timeit.default_timer()-start}")

   


    

    