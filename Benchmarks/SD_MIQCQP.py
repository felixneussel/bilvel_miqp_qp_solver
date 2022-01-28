from gurobipy import GRB ,tuplelist
import gurobipy as gp
from scipy.linalg import block_diag
from numpy import ones,log2,floor,ceil, concatenate, array, infty, zeros_like
from ..Functional.problems import setup_meta_data, setup_master

def setup_sd_miqcpcp(problem_data,big_M,optimized_binary_expansion):
    _,_,_,_,_,_,_,G_l,_,_,d_l,_,_,_,_,_,_,_,b = problem_data
    meta_data = setup_meta_data(problem_data,optimized_binary_expansion)
    _,_,_,_,_,_,bin_coeff_arr, _ = meta_data
    model,y,dual,w = setup_master(problem_data,meta_data,big_M,optimized_binary_expansion)
    #setup strong duality constraint
    linear_vector = concatenate((d_l, - b, bin_coeff_arr))
    y_lam_w = model.y.select() + dual.select() + w.select()
    model.addMQConstr(Q = G_l, c = linear_vector, sense="<", rhs=0, xQ_L=y.select(), xQ_R=y.select(), xc=y_lam_w, name="Strong Duality Constraint" )
    return model