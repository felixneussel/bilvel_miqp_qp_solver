from Parsers.file_reader import mps_aux_reader 
from Solver_OOP.miqp_qp_solver import MIQP_QP
import re
import numpy as np
import os
import concurrent.futures as futures

def stop_process_pool(executor):
    for pid, process in executor._processes.items():
        process.terminate()
    executor.shutdown()

def run(mps_file):
    aux_file = re.sub(r'mps','aux',mps_file)
    name = re.sub(r'mps','',mps_file)
    print(f'Trying to solve {name}')
    mps_pa = '/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/'+mps_file
    aux_pa = '/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/'+aux_file
    
    n_I,n_R,n_y,m_u,m_l,c_u,d_u,A,B,a,int_lb,int_ub,d_l,C,D,b = mps_aux_reader(mps_pa,aux_pa)
    #Input data
    np.random.seed(3)
    H = np.random.normal(loc = 1,size=(n_I+n_R,n_I+n_R))
    H = H.T@H
    G_u = np.random.normal(loc = 1,size=(n_y,n_y))
    G_u = G_u.T@G_u
    G_l = np.random.normal(loc = 1,size=(n_y,n_y))
    G_l = G_l.T@G_l  

    m = MIQP_QP(n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c_u,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b)
    with open('Results/test_run4.txt','a') as out:
        out.write(f'newproblem {name} n_I {n_I} n_R {n_R} n_y {n_y} m_u {m_u} m_l {m_l}\n')
    for f in ['MT','ST']:
        for mode in ['remark_1','fixed_master','new']:
            if f == 'MT':
                m.solve(mode)
            else:
                m.solve_ST(mode)
            with open('Results/test_run4.txt','a') as out:
                out.write(f'method {f} sub_feas_creation_mode {mode} solution ')
                for key in m.bilevel_solution:
                    out.write(f'{key} {m.solution[key]} ')
                out.write(f'obj {m.UB} time {m.runtime} iterations {m.iteration_counter}\n') 
            print(f'{name} solved with algo {f} and sub/feas-creation mode {mode}!')

if __name__ == '__main__':
    #get all test instances that are solvalble under 10secs
    data = []
    with open('/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Implementierung/MIQP_QP_Solver/Results/test_run2.txt','r') as d:
        for line in d:
            line = line.split()
            if len(line) > 1:
                if line[1] != 'timeout':
                    data.append(line[0]+'mps')

   
    for mps_file in data[4:]:
        if re.match(r'.*\.mps$', mps_file) is not None:  
            with futures.ProcessPoolExecutor() as e:
                f = e.submit(run,mps_file)
                try:
                    a = f.result(timeout=180)
                except futures._base.TimeoutError:
                    stop_process_pool(e)
                    print(f'problem {mps_file} exceeded time limit')
                    with open('Results/test_run4.txt','a') as out:
                        out.write(f'{mps_file} timeout\n')