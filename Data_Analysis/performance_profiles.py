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
    plt.xscale("log")
    plt.legend()
    plt.show()
    




if __name__ == "__main__":
    pd.set_option("display.max_rows", 64)
    FILE = "MIPLIB_RESULTS/remark_2_results_15_min.txt"
    data = getData(FILE)
    performance_profile(data)

