from pysmps import smps_loader as smps
import numpy as np

name,objective_name,row_names,col_names,var_types,constr_types,c,A_in,rhs_names,rhs,bnd_names,bnd = smps.load_mps('/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/stein27-0.900000.mps')

print(name)
print()
print(objective_name)
print()
print(row_names)
print()
print(col_names)
print()
print(var_types)
print()
print(constr_types)
print()
print(c)
print()
print(A_in)
print()
print(rhs_names)
print()
print(rhs)
print()
print(bnd_names)
print()
print(bnd)
print()

N = -1
M = -1
LC = []
LR = []
LO = []
OS = 1
with open('/Users/felixneussel/Documents/Uni/Vertiefung/Bachelorarbeit/Problemdata/data_for_MPB_paper/miplib3conv/stein27-0.900000.aux','r') as aux:
    for line in aux:
        name, value = line.split()
        if name == 'N':
            N = int(value)
        elif name == 'M':
            M = int(value)
        elif name == 'LC':
            LC.append(int(value))
        elif name == 'LR':
            LR.append(int(value))
        elif name == 'LO':
            LO.append(float(value))
        elif name == 'OS':
            OS = int(value)
        else:
            raise ValueError(f'Auxilary file contains unexpected keyword: {name}')

print(N)
print(M)
print(LC)
print(LR)
print(LO)
print(OS)

#We need c_u, d_u, A, B, a, x-, x+, d_l, C, D, b

upper_constr = []
lower_constr = []

for i,row in enumerate(A_in):
    if i in LR:
        lower_constr.append(row)
    else:
        upper_constr.append(row)

upper_constr = np.array(upper_constr)
lower_constr = np.array(lower_constr)

print(f'upper : {upper_constr.shape}')
print(f'lower : {lower_constr.shape}')

A = []
B = []
C = []
D = []

for j,col in enumerate(upper_constr.T):
    if j in LC:
        B.append(col)
    else:
        A.append(col)

for j,col in enumerate(lower_constr.T):
    if j in LC:
        D.append(col)
    else:
        C.append(col)


A = np.array(A).T
B = np.array(B).T
C = np.array(C).T
D = np.array(D).T

print()
for matrix in [A,B,C,D]:
    print(matrix.shape)
