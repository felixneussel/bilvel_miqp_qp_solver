

def readData(filepath):
    data = []
    name = ''
    with open(filepath,'r') as file:
        for line in file:
            line = line.split()
            if line[0] == 'newproblem':
                data.append([])
                name = line[1]
            elif line[0] == 'method':
                obj_ind = line.index('obj')
                time_ind = line.index('time')
                d = {'name':name,'method':line[1],'sub_feas_mode':line[3],'status':'solved','obj':line[obj_ind+1],'runtime':line[time_ind+1]}
                data[-1].append(d)
            elif line[1] == 'timeout':
                problems = data[-1]
                solved = {(d['method'],d['sub_feas_mode']) for d in problems}
                all = {(m,sf) for m in ['MT','ST'] for sf in ['remark_1','fixed_master','new']}
                unsolved = all - solved
                for m,sf in unsolved:
                    d = {'name':name,'method':m,'sub_feas_mode':sf,'status':'unsolved'}
    return data

def evaluateData(data):
    for d in data:
        for pr in d:
            print(pr['obj'],end =' ')
        print('\n')

def run_test():
    pa ='/Users/felixneussel/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Vertiefung/Bachelorarbeit/Implementierung/MIQP_QP_Solver/Results/test_run4.txt'
    print(evaluateData(readData(pa)))