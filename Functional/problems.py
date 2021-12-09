from gurobipy import Model,GRB, tuplelist
from numpy import ones,log2,floor, concatenate, array
from scipy.linalg import block_diag
from re import match

def setup_meta_data(problem_data):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    r_bar = (floor(log2(int_ub)) + ones(int_ub.shape)).astype(int)
    jr = tuplelist([(a,b) for a in range(0,n_I) for b in range(0, r_bar[a])])#Caution, r_bar[a] was changed from r_bar[a-1]
    I = tuplelist([a for a in range(0,n_I)])
    R = tuplelist([a for a in range(0,n_R)])
    J = tuplelist([a for a in range(0,n_y)])
    ll_constr = tuplelist([a for a in range(0,m_l)])
    bin_coeff_dict = getBinaryCoeffsDict(jr)
    bin_coeff_arr = getBinaryCoeffsArray(jr)
    return jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr

def setup_master(problem_data,meta_data):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    model = Model('Masterproblem')
    model.Params.LogToConsole = 0
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data

    x_R = model.addVars(R, vtype=GRB.CONTINUOUS,name='x_R')
    y = model.addVars(J, vtype=GRB.CONTINUOUS,name='y')
    dual = model.addVars(ll_constr,vtype=GRB.CONTINUOUS, lb=0,name='lambda')
    w = model.addVars(jr,vtype=GRB.CONTINUOUS, name="w")
    x_I = model.addVars(I, vtype=GRB.INTEGER,lb=int_lb, ub=int_ub,name='x_I')
    s = model.addVars(jr,vtype= GRB.BINARY,name='s')
    #setObjective(model)
    HG = block_diag(H,G_u)
    cd = concatenate((c,d_u))
    primalvars = x_I.select() + x_R.select() + y.select()
    model.setMObjective(Q=HG/2,c=cd,constant=0.0,xQ_L=primalvars,xQ_R=primalvars,xc=primalvars,sense=GRB.MINIMIZE)
    #setPConstraint(model)
    AB = concatenate((A,B),1)
    #primalvars = x_I.select() + x_R.select() + y.select()
    model.addMConstr(A=AB,x=primalvars,sense='>=',b=a)

    CD = concatenate((C,D),1)
    lower_level_vars = x_I.select() + y.select()
    model.addMConstr(A=CD,x=lower_level_vars,sense='>=',b=b)
    #setDualFeasiblityConstraint(model)
    GD = concatenate((D.T,-G_l),1)
    y_lambda = dual.select() + y.select()
    model.addMConstr(A=GD,x=y_lambda,sense='=',b=d_l)
    #setStrongDualityLinearizationConstraint(model)
    bin_coeff = getBinaryCoeffsDict(jr)
    model.addConstrs((s.prod(bin_coeff,j,'*') == x_I[j] for j,r in jr),'binary expansion')
    ub = 1e5
    lb = 1e5
    model.addConstrs((w[j,r] <= ub*s[j,r] for j,r in jr),'13a')
    model.addConstrs((w[j,r] <= sum([C[i,j]*dual[i] for i in ll_constr]) + lb*(s[j,r] - 1) for j,r in jr),'13b')
    #Possible refactor: replace lam_coeff with C and get rid of lam_coeff
    model.addConstrs((w[j,r] >= lb*s[j,r] for j,r in jr),'13c')
    model.addConstrs((w[j,r] >= sum([C[(i,j)]*dual[i] for i in ll_constr]) + ub*(s[j,r] - 1) for j,r in jr),'13d')
    return model,y.select(),dual.select(),w.select()

def setup_sub(problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
    model = master.fixed()
    removeMasterUnnecessaryMasterConstraints(model,cut_counter)
    #Add non linear Strong Duality Constraint
    linear_vector = concatenate((d_l, - b, bin_coeff_arr))
    y_lam_w = y_var + dual_var + w_var
    model.addMQConstr(Q = G_l, c = linear_vector, sense="<", rhs=0, xQ_L=y_var, xQ_R=y_var, xc=y_lam_w, name="Strong Duality Constraint" )
    return model

def setup_feas(problem_data,master,meta_data,y_var,dual_var,w_var,cut_counter):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
    model = master.fixed()
    #set new Objective
    linear_vector = concatenate((d_l, - b, bin_coeff_arr))
    y_lam_w = y_var + dual_var + w_var
    model.setMObjective(Q=G_l,c=linear_vector,constant=0,xQ_L=y_var,xQ_R=y_var,xc=y_lam_w,sense=GRB.MINIMIZE)
    removeMasterUnnecessaryMasterConstraints(model,cut_counter)
    return model

def check_dimensions(problem_data):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    if H.shape != (n_I+n_R,n_I+n_R):
        raise ValueError('Dimension of H is not n_I+n_R x n_I+n_R.')
    elif G_u.shape != (n_y, n_y):
        raise ValueError('Dimension of G_u is not n_y x n_y.')
    elif G_l.shape != (n_y, n_y):
        raise ValueError('Dimension of G_l is not n_y x n_y.')
    elif c.shape != (n_I+n_R,):
        raise ValueError('Dimension of c is not n_I+n_R')
    elif d_u.shape != (n_y,):
        raise ValueError('Dimension of d_u is not n_y')
    elif d_l.shape != (n_y,):
        raise ValueError('Dimension of d_l is not n_y')
    elif A.shape != (m_u, n_I+n_R):
        raise ValueError('Dimension of A is not m_u x n_I+n_R')
    elif B.shape != (m_u, n_y):
        raise ValueError('Dimension of B is not m_u x n_y')
    elif a.shape != (m_u,):
        raise ValueError('Dimension of a is not m_u')
    elif int_lb.shape != (n_I,):
        raise ValueError('Dimension of int_lb is not n_I')
    elif int_ub.shape != (n_I,):
        raise ValueError('Dimension of int_ub is not n_I')
    elif C.shape != (m_l, n_I,):
        raise ValueError('Dimension of C is not m_l x n_I')
    elif D.shape != (m_l, n_y):
        raise ValueError('Dimension of D is not m_l x n_y')
    elif b.shape != (m_l,):
        raise ValueError('Dimension of b is not m_l')
    else:
        pass

def optimize(model):
    model.optimize()
    return model.status, model.getVars(),model.ObjVal

def getBinaryCoeffsArray(index_set):
    bi_c_arr = []
    for (j,r) in index_set:
        bi_c_arr.append(2**r)
    return array(bi_c_arr)

def getBinaryCoeffsDict(index_set):
    bin_coeff = {}
    for (j,r) in index_set:
        bin_coeff[(j,r)] = 2**r
    return bin_coeff

def removeMasterUnnecessaryMasterConstraints(model,cut_counter):
        constr = model.getConstrs()
        for i in range(cut_counter):
            model.remove(constr.pop())
        filtered_cons = list(filter(lambda c: match(r'^binary expansion',c.ConstrName) is not None,constr))
        for con in filtered_cons:
            model.remove(con)

def add_cut(problem_data,model,meta_data,y_var,dual_var,w_var,p):
        """
        Takes a point \bar{y}, linearizes strong duality constraint at this point and adds the constraint to the model.
        """
        n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
        jr,I,R,J,ll_constr,bin_coeff_dict,bin_coeff_arr = meta_data
        cut_point = p
        #twoyTG = 2*cut_point.T @ G_l
        yTGy = cut_point.T @ G_l @ cut_point
        term1 = 2*cut_point.T @ G_l @ y_var #sum([twoyTG[i]*y_var(i)[0] for i in J])
        term2 = d_l.T @ y_var #sum([d_l[i]*y_var(i)[0] for i in J])
        term3 = -b.T @ dual_var #-sum([b[j]*dual_var(j)[0] for j in ll_constr])
        term4 = bin_coeff_arr@w_var #w_var.prod(bin_coeff_dict)
        
        model.addConstr((term1+term2+term3+term4-yTGy <= 0),'Strong duality linearization')


