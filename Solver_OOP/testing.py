from Problems import loadProblem, randomProblem
from miqp_qp_solver import MIQP_QP
import re
import numpy as np


for i in range(500):
    p = randomProblem(i)
    m = MIQP_QP(*p)
    m.solve_ST()

    if m.UB != np.infty:
        print(i)
        break
    
print('Test completed')    


""" 

#name = 'ClarkWesterberg1990a'
#name = 'GumusFloudas2001Ex4'
name = 'random3'
p = loadProblem(name)
#print(p)
print('\n\n\n')
m = MIQP_QP(*p)
m.solve_ST()
print()
print()
print(f'Results for {name}')
print()
print()
print('All variables')
print()
for key in m.solution:
    print(key,'=', m.solution[key])
print()
print('Variables of the Bilevel problem')
print()
for key in m.bilevel_solution:
    if re.match(r'^x',key):
        print(key,'=', m.solution[key])
for key in m.bilevel_solution:
    if re.match(r'^y',key):
        print(key,'=', m.solution[key])
print()

print('Objective Function : ', m.UB)
print()
print('Runtime : ',m.runtime, 's')
print()
print('Iterations : ', m.iteration_counter) """