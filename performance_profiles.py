#
#This file contains functions related to the analysis of numerical results. Mean and Median running times can be computed.
#Performance profiles can be plotted.
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import LogScale



def plot_performance_profile(df,solvers,select_option):
    """
    Plots performance profiles.

    # Parameters 

    - df : Data Frame containing the test results
    - solvers : List of tuples where each tuple represents an algorithm-submode combination that should be plotted.
    - select_option : selection criteria: 'none', 'one' : only instances that at least one solver could solve, 'all' : only instances that all solvers could solve.
    """
    st = get_run_times(df,solvers,select_option)
    profiles = performance_profile(st[0])

    max_tau = 1
    
    for solver in profiles:
        plt.step(profiles[solver]['tau'],profiles[solver]['rho'],where='post',label=solver,linestyle='--',alpha=0.9,linewidth=2)
        if max(profiles[solver]['tau']) > max_tau:
            max_tau = max(profiles[solver]['tau'])

    plt.xscale(LogScale(0,base=2))
    plt.ylim([-0,1])
    plt.xlim(left=1,right = max_tau)
    plt.rcParams['text.usetex'] = True
    plt.xlabel(r"Factor $\tau$")
    plt.legend()
    plt.show()


def performance_profile(d):
    """
    Creates performance profile data.

    # Parameters

    - d : dictionary of form {solver1 : [time for p1, time for p2 ...], solver2:[time for p1, time for p2]} and time is inf if s did not solve p.

    # Returns

    - result, a dictionary of the form {solver1 : {tau : [tau1, tau2, ...], rho : [rho1, rho2, ...]}, ...}, for each solver,
    the plot of the performance profile can be created with plt.step(result[solver][tau], result[solver][rho], where = 'post')
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
    """
    Creates a pandas data frame of test results from a .txt file containing the the test results.
    """
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
    """
    Calculates the percentage of instances (y-axis) for which the performance ratio of
    approach s is within a factor tau of the best possible ratio.
    """
    n = len(ratios)
    return sum(a <= tau for a in ratios) / n

def get_run_times(df,algos_submodes,option):
    """
    Returns runtimes, numbers of subproblems and times spent in subproblems of solution approaches for all instances or
    the ones that could be solved by at least one solution approach or by all solution approaches from a Data Frame.

    # Parameters:

    - df : pd Dataframe with test data.
    - algos_submodes : List of tuples where each tuple represents an algorithm-submode combination that should be tested.
    - option : selection criteria: 'none', 'one' : only instances that at least one solver could solve, 'all' : only instances that all solvers could solve.

    # Returns

    Tuple containing
    - times_dict : Dictionary containing the runtimes of each solution approach in the form {solver1 : [time for p1, time for p2 ...], solver2:[time for p1, time for p2]} and time is inf if s did not solve p
    - subnum_dict : Dictionary containing the numbers of solved subproblems of each solution approach in the form {solver1 : [number of subproblems for p1, number of subproblems for p2 ...], solver2:[number of subproblems for p1, number of subproblems for p2]}
    - times_dict : Dictionary containing the times spent for subproblems of each solution approach in the form {solver1 : [time in subproblems for p1, time in subproblems for p2 ...], solver2:[time in subproblems for p1, time in subproblems for p2]}
    """
    d = {}
    for a_s in algos_submodes:
        a,s = a_s
        d[f'{a} {s}'] = {}
        d[f'{a} {s}']['times'] = df.loc[(a,s),'time'].tolist()
        d[f'{a} {s}']['status'] = df.loc[(a,s),'status'].tolist()
        d[f'{a} {s}']['subnum'] = df.loc[(a,s),'subnum'].tolist()
        d[f'{a} {s}']['subtime'] = df.loc[(a,s),'subtime'].tolist()
    return select_problems(d,option)


def select_problems(d,option):
    """
    Returns runtimes, numbers of subproblems and times spent in subproblems of solution approaches for all instances or
    the ones that could be solved by at least one solution approach or by all solution approaches from a dictionary.

    # Parameters:

    - d : dictionary of the form {solver1 : {times : [t for p1, t for p2, ...], status : [s for p1, ...], subnum : [number of subproblems for p1, ...], subtime : [time in subproblems for p1, ...]}, solver2 : {...},... }
    - option : selection criteria: 'none', 'one' : only instances that at least one solver could solve, 'all' : only instances that all solvers could solve.

    # Returns

    Tuple containing
    - times_dict : Dictionary containing the runtimes of each solution approach in the form {solver1 : [time for p1, time for p2 ...], solver2:[time for p1, time for p2]} and time is inf if s did not solve p
    - subnum_dict : Dictionary containing the numbers of solved subproblems of each solution approach in the form {solver1 : [number of subproblems for p1, number of subproblems for p2 ...], solver2:[number of subproblems for p1, number of subproblems for p2]}
    - times_dict : Dictionary containing the times spent for subproblems of each solution approach in the form {solver1 : [time in subproblems for p1, time in subproblems for p2 ...], solver2:[time in subproblems for p1, time in subproblems for p2]}
    """
    times = []
    for solver in d:
        d[solver]['times'] = list(map(lambda t,s : t if s == 2 else np.infty, d[solver]['times'],d[solver]['status']))
        times.append(d[solver]['times'])
    times = zip(*times)
    mask = list(map(lambda x : select(x,option),times))
    times_dict = {}
    for key in d:
        times_dict[key] = [t for t,i in zip(d[key]['times'],mask) if i ==True]
    subnum_dict = {}
    for key in d:
        subnum_dict[key] = [t for t,i in zip(d[key]['subnum'],mask) if i ==True]
    subtime_dict = {}
    for key in d:
        subtime_dict[key] = [t for t,i in zip(d[key]['subtime'],mask) if i ==True]

    return times_dict,subnum_dict,subtime_dict

def select(s,option):
    """
    Checks if one problem instance was solved by at least one or all solution approaches.
    """
    if option == 'one':
            return  s != tuple([np.infty]*len(s))
    elif option == 'all':
        return np.infty not in s
    elif option == 'none':
        return True
    else:
        raise ValueError(f"'{option}' is not a valid option.")

def mean_median_df(df,algo_submodes):
    """
    Creates a pandas dataframe that displays mean and median values for the runtimes, times spent for solving subproblems
    and numbers of solved subproblems for the algorithm-subproblem-mode tuples specified.
    """
    times_dict,subnum_dict,subtime_dict = get_run_times(df,algo_submodes,'all')
    d = {
        ('Running Time','Mean'):[],
        ('Running Time','Median'):[],
        ('Solved subproblems','Mean'):[],
        ('Solved subproblems','Median'):[],
        ('Time in subproblems','Mean'):[],
        ('Time in subproblems','Median'):[]
        }
    for key in times_dict:
        m=2
        d[('Running Time','Mean')].append(round(np.mean(np.array(times_dict[key])),m))
        d[('Running Time','Median')].append(round(np.median(times_dict[key]),m))
        d[('Solved subproblems','Mean')].append(round(np.mean(subnum_dict[key]),m))
        d[('Solved subproblems','Median')].append(int(np.median(subnum_dict[key])))
        d[('Time in subproblems','Mean')].append(round(np.mean(subtime_dict[key]),m))
        d[('Time in subproblems','Median')].append(round(np.median(subtime_dict[key]),m))
    return pd.DataFrame(d,index=algo_submodes)


def opt_bin_exp_plot():
    """
    Plots means, medians and midspreads from the numerical study of the optimized binary expansion.
    """
    d = {'opt_time':[],'std_time':[]}
    index = []
    with open('Results/bin_opt_res_2.txt','r') as f:
        for line in f:
            line = line.split()
            opt = line[line.index('opt_bin_exp')+1]
            if opt == 'True':
                d['opt_time'].append(float(line[line.index('time')+1]))
                index.append(int(line[line.index('shift')+1]))
            else:
                d['std_time'].append(float(line[line.index('time')+1]))    

    df = pd.DataFrame(d,index=index)
    df.index.name = 'shift'
    df = df.sort_index()
    
    dist = 0.25
    std_label = 'BE-STD'
    opt_label = 'BE-OPT'
    
    means = df.groupby(['shift']).mean()
    medians = df.groupby(['shift']).median()
    vars = df.groupby(['shift']).var()
    l_quantile = df.groupby(['shift']).quantile(q=0.5 - dist)
    u_quantile = df.groupby(['shift']).quantile(q=0.5 + dist)



    _, ax = plt.subplots(1)
    ax.plot(means.index,means['opt_time'],label=f'{opt_label} mean',color='orange')
    ax.plot(medians.index,medians['opt_time'],label=f'{opt_label} median',color='orange',linestyle=':')
    ax.fill_between(vars.index, l_quantile['opt_time'], u_quantile['opt_time'], facecolor='orange', alpha=0.4,
                    label=f'{opt_label} midspread')

    ax.plot(means.index,means['std_time'],label=f'{std_label} mean',color='blue')
    ax.plot(medians.index,medians['std_time'],label=f'{std_label} median',color='blue',linestyle=':')
    ax.fill_between(vars.index, l_quantile['std_time'], u_quantile['std_time'], facecolor='blue', alpha=0.4,
                    label=f'{std_label} midspread')
    ax.legend(loc='upper left')

    ax.set_xlabel('Lower integer bound')
    ax.set_ylabel('Running time')
    ax.set_xscale(LogScale(axis=0,base=2))
 
    
    plt.show()

def bin_exp_perf_profile(shift):
    """
    Plots performance profiles of a solver with and without the optimized binary expansion.
    """
    d = {'opt_time':[],'std_time':[]}
    index = []
    with open('Results/bin_opt_res_2.txt','r') as f:
        for line in f:
            line = line.split()
            opt = line[line.index('opt_bin_exp')+1]
            if opt == 'True':
                d['opt_time'].append(float(line[line.index('time')+1]))
                index.append(int(line[line.index('shift')+1]))
            else:
                d['std_time'].append(float(line[line.index('time')+1]))

    df = pd.DataFrame(d,index=index)
    df.index.name = 'shift'
    df = df.sort_index()

    df = df.loc[[shift]]

    d = {'BE-OPT':df['opt_time'],'BE-STD':df['std_time']}
    profiles = performance_profile(d)
    
    
    plt.step(profiles['BE-OPT']['tau'],profiles['BE-OPT']['rho'],where='post',label='BE-OPT',linestyle='-',alpha=0.9,linewidth=2)
    plt.step(profiles['BE-STD']['tau'],profiles['BE-STD']['rho'],where='post',label='BE-STD',linestyle='-',alpha=0.9,linewidth=2)
  

    plt.xscale(LogScale(0,base=2))
    plt.ylim([-0,1])
    plt.xlim(left=1,right = 16)
    plt.rcParams['text.usetex'] = True
    plt.xlabel(r"Factor $\tau$")
    plt.legend(loc='lower right')
    plt.show()

def get_att(line,name):
    return line[line.index(name)+1]

if __name__ == '__main__':
    df = get_test_data('Results/results.txt')
    print(mean_median_df(df,[('ST','remark_2'),('MT-K-F-W','remark_2'),('SD-MIQCQP','-')]))
    plot_performance_profile(df, [('ST','remark_2'),('MT-K-F-W','remark_2'),('SD-MIQCQP','-')],'one')
