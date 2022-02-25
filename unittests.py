import unittest
from Data_Analysis.performance_profiles import performance_from_dict,get_tau_rho,rho_of_tau

class TestPerformance(unittest.TestCase):


    def test_get_tau_rho(self):
        self.assertEqual(get_tau_rho([4,8,2,2]),([2,4,8],[2,3,4]))
        self.assertEqual(get_tau_rho([1,4,4,4,3,3,9]),([1,3,4,9],[1,3,6,7]))
        self.assertEqual(get_tau_rho([5,4,3,2,1]),([1,2,3,4,5],[1,2,3,4,5]))

    def test_rho_of_tau(self):
        self.assertEqual(rho_of_tau(4,[1,2,3,4,5]),4/5)
        self.assertEqual(rho_of_tau(1,[1,5,8,4,22,1]),2/6)


if __name__ == '__main__':
    
    unittest.main()