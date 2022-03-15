#
#This file contains a functions that are required by the solver.
#

from timeit import default_timer

def time_remaining(start,time_limit):
    return max(time_limit - (default_timer()-start),0)