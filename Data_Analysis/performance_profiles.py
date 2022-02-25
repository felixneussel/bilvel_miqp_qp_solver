from re import S
from numpy import NAN, NaN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def performance_profile(d):
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

def get_test_data(file):
    d = {}
    algo = []
    submode = []
    name = []
    status = []
    time = []
    obj = []
    gap = []
    subtime = []
    subnum = []
    with open(file,'r') as f:
        for l in f:
            l = l.split()
            algo.append(l[l.index('algorithm')+1])
            submode.append(l[l.index('submode')+1])
            name.append(l[l.index('name')+1])
            status.append(int(l[l.index('status')+1]))
            time.append(float(l[l.index('time')+1]))
            obj.append(float(l[l.index('obj')+1]))
            gap.append(float(l[l.index('gap')+1]))
            subtime.append(float(l[l.index('subtime')+1]))
            subnum.append(int(l[l.index('subnum')+1]))

    d = {
        'status':status,
        'time':time,
        'obj':obj,
        'gap':gap,
        'subtime':subtime,
        'subnum':subnum
    }
    tuples = list(zip(algo,submode,name))
    index = pd.MultiIndex.from_tuples(tuples, names=["algorithm", "submode","problem"])
    return pd.DataFrame(d,index=index)


def rho_of_tau(tau,ratios):
    n = len(ratios)
    return sum(a <= tau for a in ratios) / n

def get_run_times(df,algos,submodes,option):
    """
    ## Input:

    - df : pd Dataframe with test data
    - algos : Algorithms of which times should be retrieved
    - submodes: Submodes of which times should be retrieved
    - status : selection criteria: 'none', 'one' : only instances that at least one solver could solve, 'all' : only instances that all solver could solve
    """
    d = {}
    for a in algos:
        for s in submodes:
            d[f'{a} {s}'] = {}
            d[f'{a} {s}']['times'] = df.loc[(a,s),'time'].tolist()
            d[f'{a} {s}']['status'] = df.loc[(a,s),'status'].tolist()
    return select_problems(d,option)


def select_problems(d,option):
    times = []
    for solver in d:
        d[solver]['times'] = list(map(lambda t,s : t if s == 2 else np.infty, d[solver]['times'],d[solver]['status']))
        times.append(d[solver]['times'])
    times = zip(*times)
    mask = list(map(lambda x : select(x,option),times))
    result = {}
    for key in d:
        result[key] = [t for t,i in zip(d[key]['times'],mask) if i ==True]

    return result




def select(s,option):
    if option == 'one':
            return  s != tuple([np.infty]*len(s))
    elif option == 'all':
        return np.infty not in s
    elif option == 'none':
        return True
    else:
        raise ValueError(f"'{option}' is not a valid option.")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.scale import LogScale

    pd.set_option('display.max_rows', 500)
    df = get_test_data('/Users/felixneussel/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Vertiefung/Bachelorarbeit/Implementierung/MIQP_QP_Solver/results.txt')
    st = get_run_times(df,['KKT-MIQP','SD-MIQCQP'],['-'],'one')
    profiles = performance_profile(st)

    
    for solver in profiles:
        plt.step(profiles[solver]['tau'],profiles[solver]['rho'],where='post',label=solver)

    plt.xscale(LogScale(0,base=2))
    plt.ylim([-0.01,1.01])
    plt.xlim(left=1)
    plt.legend()
    plt.show()

    
