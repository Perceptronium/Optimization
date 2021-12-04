import numpy as np
import sys
sys.path.append("..")
sys.path.append("../../..")


# import the test classes

import unittest


from optimization_algorithms.interface.nlp_solver import  NLPSolver
from optimization_algorithms.interface.mathematical_program_traced import  MathematicalProgramTraced

#from optimization_algorithms.mathematical_programs.quadratic_identity_2 import QuadraticIdentity2
from my_functions.f_sq import f_sq
from my_functions.f_hole import f_hole
from optimization_algorithms.mathematical_programs.hole import Hole
from my_functions.f_rastrigin import f_rastrigin
from plots.plots2d import plotFunc


from solution import GENERAL_SOLVER

class testSolver0(unittest.TestCase):
    """
    test on problem A
    """
    Solver = GENERAL_SOLVER
    #problem = MathematicalProgramTraced(f_sq(c = 10, size = 2))
    problem = MathematicalProgramTraced(f_hole(a = 0.1, c = 10, size = 2))
    #problem = MathematicalProgramTraced(Hole(10,2))
    #problem = MathematicalProgramTraced(f_rastrigin(a = 5, c = 1, size = 2))

    def testConstructor(self):
        """
        check the constructor
        """
        solver = self.Solver()



    def testConvergence(self):
        """
        check that student solver converges
        """
        solver = self.Solver(algorithm = 'newton_method')
        solver.setProblem((self.problem))
        output =  solver.solve()
        last_trace = self.problem.trace_x[-1]
        self.assertTrue( np.linalg.norm( np.zeros(2) - last_trace  ) < .9)



    def testPlotting(self):

        def f(x):
            return self.problem.evaluate(x)[0][0]
            #return MathematicalProgramTraced(f_hole(a = 0.1, c = 10, size = 2)).evaluate(x)[0][0]
            
        trace_x = np.array(self.problem.trace_x)
        trace_phi = np.array(self.problem.trace_phi)
        plotFunc(f, bounds_lo=[-2,-2], bounds_up=[2,2], trace_xy = trace_x, trace_z = trace_phi)

    


if __name__ == "__main__":
   unittest.main()



