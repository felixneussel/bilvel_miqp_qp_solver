from gurobipy import GRB 
import gurobipy as gp

def setup_sd_miqcpcp(problem_data):
    n_I,n_R,n_y,m_u,m_l,H,G_u,G_l,c,d_u,d_l,A,B,a,int_lb,int_ub,C,D,b = problem_data
    model = gp.Model("SD-MIQCQP")
    