


from collections import defaultdict

def readDataOld(filepath):
    data = []
    name = ''
    with open(filepath,'r') as file:
        for line in file:
            line = line.split()
            if line[0] == 'newproblem':
                name = line[1]
            elif line[1] == 'timeout':
                continue
            elif line[0] == 'method' and line[2] != 'infeasible':
                obj_ind = line.index('obj')
                time_ind = line.index('time')
                d = {'name':name,'method':line[1],'sub_feas_mode':line[3],'status':'solved','obj':line[obj_ind+1],'runtime':line[time_ind+1]}
                data.append(d)
            elif line[2] == 'infeasible':
                continue
                """ problems = data[-1]
                solved = {(d['method'],d['sub_feas_mode']) for d in problems}
                all = {(m,sf) for m in ['MT','ST'] for sf in ['remark_1','fixed_master','new']}
                unsolved = all - solved
                for m,sf in unsolved:
                    d = {'name':name,'method':m,'sub_feas_mode':sf,'status':'unsolved'}
                    data[-1].append(d) """
    return data

def evaluateData(oop,data):
    res = {}
    for fun in oop:
        for d in fun:
            name = d['name']
            key = f"OOP-{d['method']}-{d['sub_feas_mode']}"
            try:
                res[name][key] = d['obj']
            except KeyError:
                res[name] = {}
                res[name][key] = d['obj']
    for fun in data:
        for d in fun:
            name = d['name']
            key = f"Func-{d['method']}"
            try:
                res[name][key] = d['obj']
            except KeyError:
                res[name] = {}
    return res
        
    

def run_test():
    pa ='/Users/felixneussel/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Vertiefung/Bachelorarbeit/Implementierung/MIQP_QP_Solver/Results/test_run8.txt'
    pa2 = '/Users/felixneussel/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Vertiefung/Bachelorarbeit/Implementierung/MIQP_QP_Solver/Results/test_run5.txt'
    pa3 = '/Users/felixneussel/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Vertiefung/Bachelorarbeit/Implementierung/MIQP_QP_Solver/Results/test_run9.txt'
    pa4 = '/Users/felixneussel/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Vertiefung/Bachelorarbeit/Implementierung/MIQP_QP_Solver/Results/test_run10.txt'
    pa5 = '/Users/felixneussel/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Vertiefung/Bachelorarbeit/Implementierung/MIQP_QP_Solver/Results/test_run11.txt'
    pa6 = '/Users/felixneussel/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Vertiefung/Bachelorarbeit/Implementierung/MIQP_QP_Solver/Results/test_run14.txt'
    pa7 = '/Users/felixneussel/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Vertiefung/Bachelorarbeit/Implementierung/MIQP_QP_Solver/Results/test_run15.txt'
    data = evaluateData([readDataOld(pa6),readDataOld(pa7)],[])
    for d in data:
        print(d)
        for m in data[d]:
            print(m.ljust(22,' '),data[d][m])
        print()

def readData(filepath):
    data = []
    name = ''
    with open(filepath,'r') as file:
        for line in file:
            line = line.split()
            if line[0] == 'newproblem':
                name = line[1]
            elif line[0] == 'method' and line[2] == 'solution':
                obj_ind = line.index('obj')
                time_ind = line.index('time')
                d = {'name':name,'method':line[1],'status':'solved','obj':line[obj_ind+1],'runtime':line[time_ind+1]}
                data.append(d)
            elif line[0] == 'method' and line[2] == 'infeasible':
                d = {'name':name,'method':line[1],'status':'unsolved'}
                data.append(d)
    return data