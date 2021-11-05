"""
This is an implementation of the multi tree algorithm to solve bilevel optimization problems with
an mixed-integer quadratic problem on the upper level and a quadratic problem on the lower level
that was proposed by Thomas Kleinert, Veronika Grimm and Martin Schmidt.
"""

import gurobipy as gp
from gurobipy import Model
from gurobipy import GRB
import numpy as np


#Input data, names according to paper
#Objective function
H = np.array([[1,0],[0,1]])
c = np.array([1,1])
G = np.array([[1,2],[3,4]])
d = np.array([2,0])