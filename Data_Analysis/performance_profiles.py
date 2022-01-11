import pandas as pd

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
                    status.append(line[line.index("status")+1])
                    time.append(line[line.index("time")+1])
    return pd.DataFrame(data={"problem":name,"algorithm":algo,"status":status,"runtime":time})

def performance_profile(df):
    solved = df[df["status"==2]]
    best = solved.groupby("problem")["runtime"].min()

if __name__ == "__main__":
    FILE = "MIPLIB_RESULTS/remark_2_results_15_min.txt"
    data = getData(FILE)
    print(data)
    performance_profile(data)

