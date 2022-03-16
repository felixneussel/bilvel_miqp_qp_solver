#
#This file was used to perform tests for the development of the data analysis tools.
#
import unittest
from performance_profiles import rho_of_tau,select_problems, select
import numpy as np

class TestPerformance(unittest.TestCase):


    def test_rho_of_tau(self):
        self.assertEqual(rho_of_tau(4,[1,2,3,4,5]),4/5)
        self.assertEqual(rho_of_tau(1,[1,5,8,4,22,1]),2/6)

    def test_select_problems(self):
        d = {
            's1':{'times':[1,2,3,4,5], 'status': [2,2,9,2,6],'subnum':[12,13,14,15,16],'subtime':[10,20,30,40,50]},
            's2':{'times':[10,9,8,7,6], 'status':[2,9,2,2,8],'subnum':[22,23,24,25,26],'subtime':[33,34,35,36,37]}
        }
        d_times_none = {
            's1':[1,2,np.infty,4,np.infty],
            's2':[10,np.infty,8,7,np.infty]
        }
        d_times_one = {
            's1':[1,2,np.infty,4],
            's2':[10,np.infty,8,7]
        }
        d_times_all = {
            's1':[1,4],
            's2':[10,7]
        }
        d_subnum_none = {
            's1':[12,13,14,15,16],
            's2':[22,23,24,25,26]
        }
        d_subnum_one = {
            's1':[12,13,14,15],
            's2':[22,23,24,25]
        }
        d_subnum_all = {
            's1':[12,15],
            's2':[22,25]
        }
        d_subtimes_none = {
            's1':[10,20,30,40,50],
            's2':[33,34,35,36,37]
        }
        d_subtimes_one = {
            's1':[10,20,30,40],
            's2':[33,34,35,36]
        }
        d_subtimes_all = {
            's1':[10,40],
            's2':[33,36]
        }
        self.assertEqual(select_problems(d,'none'),(d_times_none,d_subnum_none,d_subtimes_none))
        self.assertEqual(select_problems(d,'one'),(d_times_one,d_subnum_one,d_subtimes_one))
        self.assertEqual(select_problems(d,'all'),(d_times_all,d_subnum_all,d_subtimes_all))

    def test_select(self):
        self.assertEqual(select((2,np.infty,2),'none'),True)
        self.assertEqual(select((2,np.infty,2),'one'),True)
        self.assertEqual(select((2,np.infty,2),'all'),False)
        self.assertEqual(select((2,2,2,2),'none'),True)
        self.assertEqual(select((2,2,2,2),'one'),True)
        self.assertEqual(select((2,2,2,2),'all'),True)
        self.assertEqual(select((np.infty,np.infty,np.infty,np.infty),'none'),True)
        self.assertEqual(select((np.infty,np.infty,np.infty,np.infty),'one'),False)
        self.assertEqual(select((np.infty,np.infty,np.infty,np.infty),'all'),False)
        self.assertEqual(select((np.infty,np.infty),'one'),False)
        


if __name__ == '__main__':
    
    unittest.main()