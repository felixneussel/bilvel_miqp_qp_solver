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
    print(df)
    new = df.groupby("problem")["status"].apply(list).to_frame()
    print(new)
    unsolved_by_all = []
    for i,row in new.iterrows():
        status = row["status"]
        not_solved = np.array(status) != np.array([2]*len(status))
        if all(not_solved):
            unsolved_by_all.append(i)
    for i in unsolved_by_all:
        df = df.drop(df[df.problem == i].index)
    
    print(df)
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
    plt.legend()
    plt.show()

def create_performance_profiles():
    ST_FILES = ["MIPLIB_RESULTS/Testing/ST_kelley_corrected_new_results.txt","MIPLIB_RESULTS/Testing/ST_measurement_results.txt"]
    MT_FILES = ["MIPLIB_RESULTS/Testing/MT_second_measure_results.txt","MIPLIB_RESULTS/Testing/MT-K_measurement_results.txt","MIPLIB_RESULTS/Testing/MT-K-F_measurement_results.txt","MIPLIB_RESULTS/Testing/MT-K-F-W_measurement_results.txt"]
    MIXED = ["MIPLIB_RESULTS/Testing/ST_measurement_results.txt","MIPLIB_RESULTS/Testing/MT-K-F-W_measurement_results.txt"]
    data = []
    COLNAMES = ["name","submode","algorithm","status","runtime"]
    DTYPES = [str,str,str,int,float]
    for file in ST_FILES:
        data.append(getData(file))
    df = pd.concat(data)
    df = discardUnsolved(df)
    performance_profile(df)
    




if __name__ == "__main__":
    pd.set_option("display.max_rows", 100)
    create_performance_profiles()

