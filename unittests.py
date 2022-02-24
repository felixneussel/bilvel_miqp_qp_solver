import unittest
from Data_Analysis.performance_profiles import performance_from_dict,get_tau_rho

class TestStringMethods(unittest.TestCase):

    def test_performance_from_dict(self):
        d = {
        'simplex':[1,1],
        'interior':[5,10]
        }
        result = performance_from_dict(d)
        test_result = {
            'simplex':{'tau':[1,10],'rho':[1,1]},
            'interior':{'tau':[1,5,10],'rho':[0,0.5,1]}
        }
        self.assertEqual(result,test_result)

        d = {
        's1':[2,3,20],
        's2':[6,9,5]
        }
        result = performance_from_dict(d)
        test_result = {
            's1':{'tau':[1,4],'rho':[2/3,1]},
            's2':{'tau':[1,3,4],'rho':[1/3,1,1]}
        }
        self.assertEqual(result,test_result)


    def test_get_tau_rho(self):
        self.assertEqual(get_tau_rho([4,8,2,2]),([2,4,8],[2,3,4]))
        self.assertEqual(get_tau_rho([1,4,4,4,3,3,9]),([1,3,4,9],[1,3,6,7]))
        self.assertEqual(get_tau_rho([5,4,3,2,1]),([1,2,3,4,5],[1,2,3,4,5]))


if __name__ == '__main__':
    
    unittest.main()