import unittest
from Data_Analysis.performance_profiles import performance_profile,rho_of_tau,select_problems, select

class TestPerformance(unittest.TestCase):


    def test_rho_of_tau(self):
        self.assertEqual(rho_of_tau(4,[1,2,3,4,5]),4/5)
        self.assertEqual(rho_of_tau(1,[1,5,8,4,22,1]),2/6)

    def test_select_problems(self):
        d = {
            's1':{'times':[1,2,3,4,5], 'status': [2,2,9,2,6]},
            's2':{'times':[10,9,8,7,6], 'status':[2,9,2,2,8]}
        }
        d_none = {
            's1':[1,2,3,4,5],
            's2':[10,9,8,7,6]
        }
        d_one = {
            's1':[1,2,3,4],
            's2':[10,9,8,7]
        }
        d_all = {
            's1':[1,4],
            's2':[10,7]
        }
        self.assertEqual(select_problems(d,'none'),d_none)
        self.assertEqual(select_problems(d,'one'),d_one)
        self.assertEqual(select_problems(d,'all'),d_all)

    def test_select(self):
        self.assertEqual(select((2,9,2),'none'),True)
        self.assertEqual(select((2,9,2),'one'),True)
        self.assertEqual(select((2,9,2),'all'),False)
        self.assertEqual(select((2,2,2,2),'none'),True)
        self.assertEqual(select((2,2,2,2),'one'),True)
        self.assertEqual(select((2,2,2,2),'all'),True)
        self.assertEqual(select((9,9,5,1),'none'),True)
        self.assertEqual(select((9,9,5,1),'one'),False)
        self.assertEqual(select((9,9,5,1),'all'),False)


if __name__ == '__main__':
    
    unittest.main()