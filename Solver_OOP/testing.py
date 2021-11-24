from Problems import loadProblem
from miqp_qp_solver import MIQP_QP
import re

name = 'ClarkWesterberg1990a'
#name = 'GumusFloudas2001Ex4'
p = loadProblem(name)
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
print('Iterations : ', m.iteration_counter)