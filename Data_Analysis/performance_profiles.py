from re import S
from numpy import NAN, NaN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getData(filepath):
    name = []
    algo = []
    status = []
    time = []
    with open(filepath,"r") as file:
        for line in file:
            line = line.split()
            if len(line)>0:
                if line[0] == "name":
                    name.append(line[1])
                    algo.append(line[line.index("algorithm")+1])
                    status.append(int(line[line.index("status")+1]))
                    time.append(float(line[line.index("time")+1]))
    return pd.DataFrame(data={"problem":name,"algorithm":algo,"status":status,"runtime":time}).sort_values(["problem","runtime"])

def create_dataframe(filepath,colnames,dtypes):
    """
    Render pandas dataframe from data in txt file where each line represents one data point
    of the form property_1 {property_1} property_2 {property_2} ...

    Input

    filepath: path of txt file

    colnames: list of column names of the df, needs to coincide with markers in txt file

    dtypes: list of data types of the columns
    """
    dicts = []
    
    with open(filepath,"r") as file:
        for line in file:
            line = line.split()
            if len(line)==0:
                continue
            if line[0] == "Run":
                d = {}
                for name in colnames:
                    d[name] = []
                dicts.append(d)
            if line[0] != "name":
                continue
            for cn,dt in zip(colnames,dtypes):
                ind = line.index(cn)+1
                entry = dt(line[ind])
                dicts[-1][cn].append(entry)

    return list(map(lambda x: pd.DataFrame(x),dicts))

def discardUnsolved(df):
    time_limit = 300
    df["status"] = df.apply(lambda row: row.status if row.runtime <= time_limit else 9, axis=1)

    new = df.groupby("problem")["status"].apply(list).to_frame()

    unsolved_by_all = []
    for i,row in new.iterrows():
        status = row["status"]
        not_solved = np.array(status) != np.array([2]*len(status))
        if all(not_solved):
            unsolved_by_all.append(i)
    for i in unsolved_by_all:
        df = df.drop(df[df.problem == i].index)

    return(df)

def ratios(df):
    solved = df[df["status"]==2]
    best = solved.groupby("problem")["runtime"].min().to_frame().rename({"runtime":"min_t_ps"},axis=1)
    df = pd.merge(df,best,on="problem")
    df["r_ps"] = df.apply(lambda row: row.runtime / row.min_t_ps if row.status == 2 else -1, axis=1)
    r_m = df["r_ps"].max()
    df["r_ps"] = df.apply(lambda row: r_m if row.r_ps == -1 else row.r_ps,axis=1)
    return df

def performance_profile(df):
    df = ratios(df)
    df = df.sort_values(["algorithm","r_ps"])
    df = df.groupby("algorithm")["r_ps"].apply(list)
    n = len(df[0])
    df = df.to_frame()
    y = np.arange(1,n+1)/n
    for index, row in df.iterrows():
        plt.step(row.r_ps,y,label=index)
    #plt.xscale("log")
    plt.rcParams['text.usetex'] = True
    plt.xlabel(r"Factor $\tau$")
    plt.legend()
    plt.show()

def create_performance_profiles():
    ST_FILES = ["MIPLIB_RESULTS/Testing/ST_kelley_corrected_new_results.txt","MIPLIB_RESULTS/Testing/ST_measurement_results.txt"]
    MT_FILES = ["MIPLIB_RESULTS/Testing/MT_second_measure_results.txt","MIPLIB_RESULTS/Testing/MT-K_measurement_results.txt","MIPLIB_RESULTS/Testing/MT-K-F_measurement_results.txt","MIPLIB_RESULTS/Testing/MT-K-F-W_measurement_results.txt"]
    MIXED = ["MIPLIB_RESULTS/Testing/ST_measurement_results.txt","MIPLIB_RESULTS/Testing/MT-K-F-W_measurement_results.txt"]
    BENCHMARKS = ["MIPLIB_RESULTS/Benchmarks/benchmark_results.txt"]
    BM_ST_MT = ["MIPLIB_RESULTS/Benchmarks/st_mt_comparison.txt"]
    data = []
    COLNAMES = ["name","submode","algorithm","status","runtime"]
    DTYPES = [str,str,str,int,float]
    for file in MT_FILES:
        data.append(getData(file))
    df = pd.concat(data)
    df = discardUnsolved(df)
    performance_profile(df)
    

def latex_table_times_obj(ALG):
    ST_FILES = ["MIPLIB_RESULTS/Testing/ST_kelley_corrected_new_results.txt","MIPLIB_RESULTS/Testing/ST_measurement_results.txt"]
    MT_FILES = ["MIPLIB_RESULTS/Testing/MT_second_measure_results.txt","MIPLIB_RESULTS/Testing/MT-K_measurement_results.txt","MIPLIB_RESULTS/Testing/MT-K-F_measurement_results.txt","MIPLIB_RESULTS/Testing/MT-K-F-W_measurement_results.txt"]
    MIXED = ["MIPLIB_RESULTS/Testing/ST_measurement_results.txt","MIPLIB_RESULTS/Testing/MT-K-F-W_measurement_results.txt"]
    BENCHMARKS = ["MIPLIB_RESULTS/Benchmarks/benchmark_results.txt"]
    BM_ST_MT = ["MIPLIB_RESULTS/Benchmarks/st_mt_comparison.txt"]
    data = []
    COLNAMES = ["name","algorithm","status","time","obj"]
    DTYPES = [str,str,int,float,float]
    for file in BENCHMARKS:
        frames = create_dataframe(file,COLNAMES,DTYPES)
        for frame in frames:
            data.append(frame)
    df = pd.concat(data)
    print(df)
    df = df[df.algorithm == ALG]
    df = df.drop(columns="algorithm")
    df["status"] = df["status"].map(status_to_string)
    print(df)
    df = df.rename({"name":"Name","status":"Status","time":"Time","obj":"Objective"},axis=1)
    table = df.to_latex(index=False)
    with open("MIPLIB_RESULTS/Latex/Tables.txt","a") as out:
        out.write(f"{ALG}\n")
        out.write(table)
        out.write("\n")

def status_to_string(x):
    d = {2:"Optimal",9:"Time Limit",6 : "Cutoff",3:"Infeasible"}
    return d[x]

def performance_from_dict(d):
    """
    Dict has form {solver1 : [time for p1, time for p2 ...], solver2:[time for p1, time for p2]} and time is inf if s did not solve p
    """
    times = [d[key] for key in d]
    minimal_times = [min(*a) for a in zip(*times)]
    ratios = {}
    max_ratio = 1
    for solver in d:
        ratios[solver] = [a/b if b != np.infty else np.infty for a,b in zip(d[solver],minimal_times)]
        ratios[solver] = list(map(lambda x:-1 if x == np.infty else x,ratios[solver]))
        if max(ratios[solver]) > max_ratio and max(ratios[solver]) != np.infty:
            max_ratio = max(ratios[solver])
    r_m = 2*max_ratio
    for solver in ratios:
        ratios[solver] = list(map(lambda x: r_m if x == -1 else x,ratios[solver]))
    
    result = {}
    for solver in ratios:
        tau = np.arange(start=1,stop=r_m,step=0.01)
        rho = np.array(list(map(lambda t:rho_of_tau(t,ratios[solver]),tau)))
        result[solver] = {}
        result[solver]['tau'] = tau
        result[solver]['rho'] = rho

    return result

def get_tau_rho(l):
    """
    Input: List of ratios of a solver
    """
    l = sorted(l)
    tau = sorted(list(set(l)))
    rho = [sum(r <= x for r in l) for x in tau]
    return tau,rho

def rho_of_tau(tau,ratios):
    n = len(ratios)
    return sum(a <= tau for a in ratios) / n

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.scale import LogScale

    d = {
        'simplex':[1,1,1],
        'interior':[2,4,8],
        'kkt':[4,8,16]
        }
    linestyles = {
        'simplex' : ':',
        'interior' : '--',
        'kkt' : '-.'
    }
    profiles = performance_from_dict(d)
    for solver in profiles:
        plt.step(profiles[solver]['tau'],profiles[solver]['rho'],where='post',label=solver,ls=linestyles[solver])

    plt.xscale(LogScale(0,base=2))
    plt.ylim([-0.01,1.01])
    plt.xlim(left=1)
    plt.legend()
    plt.show()