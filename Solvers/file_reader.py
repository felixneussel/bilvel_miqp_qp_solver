#
#This file contains the function that reads bilevel problems from a .mps and a .aux file.
#
from pysmps import smps_loader as smps
import numpy as np
import re
import os
import threading as th
import multiprocessing as mp
import time
from concurrent import futures

def mps_aux_reader(mps_path,aux_path):
    """
    Reads an MILP-LP from a .mps and a .aux file.

    # Parameters

    - mps_path:  Path of the .mps file.
    - aux_path : Path of the .aux file.

    # Returns

    Tuple containing
    - n_I : The number of upper-level integer variables.
    - n_R : The number of upper-level continuous variables.
    - n_y : The number of lower-level variables.
    - m_u : The number of upper-level constraints.
    - m_l : The number of lower-level constraints.
    - c_u : Linear coefficient vector for upper-level variables in the upper-level objective function.
    - d_u : Linear coefficient vector for lower-level variables in the upper-level objective function.
    - A : Upper-level-variable's upper-level constraint matrix.
    - B : Lower-level variable's upper-level constraint matrix.
    - a : Upper-level right-hand-side constraint vector.
    - int_lb : Lower-Bound vector of upper-level integer variables.
    - int_ub : Upper-Bound vector of upper-level integer variables.
    - d_l : Linear coefficient vector for lower-level variables in the lower-level objective function.
    - C : Upper-level-variable's lower-level constraint matrix.
    - D : Lower-level variable's lower-level constraint matrix.
    - b : Lower-level right-hand-side constraint vector.
    """
    name,_,_,_,_,constr_types,c,A_in,rhs_names,rhs,bnd_names,bnd = smps.load_mps(mps_path)

    N = -1
    M = -1
    LC = []
    LR = []
    LO = []
    OS = 1
    with open(aux_path,'r') as aux:
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

    #Change less than constraints to greater than constraints
    if len(rhs_names) != 1:
        raise ValueError('More than one RHS - vector')
    rhs = rhs[rhs_names[0]]

    for i,sign in enumerate(constr_types):
        if sign == 'L':
            A_in[i,:] = - A_in[i,:]
            rhs[i] = -rhs[i]
        if sign == 'E':
            A_in = np.concatenate((A_in,np.array([-A_in[i,:]])))
            rhs = np.append(rhs,-rhs[i])
            #If i is index of lower level constraint, the index of the new constraint needs to be added to LR
            if i in LR:
                LR.append(rhs.shape[0]-1)

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

    if B == []:
        B = None
    else:
        B = np.array(B).T
    A = np.array(A).T

    C = np.array(C).T
    D = np.array(D).T

 



    c_u = []
    d_u = []
    for i,entry in enumerate(c):
        if i in LC:
            d_u.append(entry)
        else:
            c_u.append(entry)

    c_u = np.array(c_u)
    d_u = np.array(d_u)

    if OS == 1:
        d_l = np.array(LO)
    else:
        d_l = -np.array(LO)


    
    a = []
    b = []
    for i,entry in enumerate(rhs):
        if i in LR:
            b.append(entry)
        else:
            a.append(entry)

    a = np.array(a)
    b = np.array(b)

    int_lb = []
    int_ub = []

    y_lb = []
    y_ub = []

    if len(bnd_names) != 1:
        raise ValueError('More than one bound name')

    lb = bnd[bnd_names[0]]['LO']
    ub = bnd[bnd_names[0]]['UP']

    for i,entry in enumerate(lb):
        if i in LC:
            y_lb.append(entry)
        else:
            int_lb.append(entry)

    for i, entry in enumerate(ub):
        if i in LC:
            y_ub.append(entry)
        else:
            int_ub.append(entry)

    int_lb = np.array(int_lb)
    int_ub = np.array(int_ub)
    y_lb = np.array(y_lb)
    y_ub = np.array(y_ub)


    #Model from paper does not have lower level bounds
    #Thus, they need to be translated to linear constraints

    b = np.concatenate((b,y_lb,-y_ub))

    lb_matrix = np.diag(np.ones(N))
    ub_matrix = np.diag(-np.ones(N))



    D = np.concatenate((D,lb_matrix,ub_matrix),axis=0)



    #If an upper level variable does not appear in lower level constraints,
    #we want it to be a continuous variable

    zero_cols = []
    for j,col in enumerate(C.T):
        if all(col == np.zeros_like(col)):
            zero_cols.append(True)
        else:
            zero_cols.append(False)


    x_int_ub = []
    x_int_lb = []
    x_const_lb = []
    x_const_ub = []

    for i,entry in enumerate(zero_cols):
        if not entry:
            x_int_lb.append(int_lb[i])
            x_int_ub.append(int_ub[i])
        else:
            x_const_lb.append(int_lb[i])
            x_const_ub.append(int_ub[i])

    int_lb = np.array(x_int_lb)
    int_ub = np.array(x_int_ub)



    A_int = []
    A_cont = []

    for i,col in enumerate(A.T):
        if zero_cols[i]:
            A_cont.append(col)
        else:
            A_int.append(col)

    if A_int == [] and A_cont != []:
        A = np.concatenate((np.zeros((np.array(A_cont).T.shape[0],int_lb.shape[0])),np.array(A_cont).T),axis=1)
    elif A_int != [] and A_cont == []:
        A = np.concatenate((np.array(A_int).T,np.zeros((np.array(A_int).T.shape[0],len(x_const_lb)))),axis=1)
    elif A_int != [] and A_cont != []:
        A = np.concatenate((np.array(A_int).T,np.array(A_cont)).T,axis=1)
    else:
        A = None




    new_C = []
    
    for i,entry in enumerate(zero_cols):
        if not entry:
            new_C.append(C[:,i])

    C = np.array(new_C).T
    C = np.concatenate((C,np.zeros((lb_matrix.shape[0],C.shape[1])),np.zeros((lb_matrix.shape[0],C.shape[1]))))

    #Bounds for continous upper level variables need to be modeled
    #as linear constraints


    bound_matrix = np.concatenate((np.diag(np.ones(len(x_const_lb))),np.diag(-np.ones(len(x_const_ub)))))




    if A is None:
        A_left_bottom = np.zeros((bound_matrix.shape[0],len(x_int_lb)))
        bottom = np.concatenate((A_left_bottom,bound_matrix),axis=1)
        A = bottom
    else:
        A_left_bottom = np.zeros((bound_matrix.shape[0],len(x_int_lb)))
        bottom = np.concatenate((A_left_bottom,bound_matrix),axis=1)
        A = np.concatenate((A,bottom))

    if B is None:
        B = np.zeros((bound_matrix.shape[0],N))
    else:
        B = np.concatenate((B,np.zeros((bound_matrix.shape[0],B.shape[1]))))

    a = np.concatenate((a,np.array(x_const_lb),-np.array(x_const_ub)))


    #Number of Integer upper-level variables
    n_I = int_lb.shape[0]
    #Number of Continuous upper-level variables
    n_R = len(x_const_ub)
    #Number of lower-level variables
    n_y = d_u.shape[0]
    #Number of upper level constraints
    m_u = A.shape[0]
    #Number of lower level constaints
    m_l = C.shape[0]

    return n_I,n_R,n_y,m_u,m_l,c_u,d_u,A,B,a,int_lb,int_ub,d_l,C,D,b




                    
                

                

                  