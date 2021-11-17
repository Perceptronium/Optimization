import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.nlp_solver import  NLPSolver

class Solver0(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        
        # in case you want to initialize some class members or so...


    def solve(self) :
        """

        See Also:
        ----
        NLPSolver.solve

        """
        
        # write your code here

        # use the following to get an initialization:
        x = self.problem.getInitializationSample()

        # use the following to query the problem:
        phi, J = self.problem.evaluate(x)
        # phi is a vector (1D np.array); use phi[0] to access the cost value (a float number). J is a Jacobian matrix (2D np.array). Use J[0] to access the gradient (1D np.array) of the cost value.

        # now code some loop that iteratively queries the problem and updates x til convergence....
        delta_x = x
        iter_ctr = 0
        alpha = 0.1 # Needs 0.01 for f_hole convergence
        while(np.linalg.norm(delta_x) >= 0.0001):
            old_x = x
            phi, J = self.problem.evaluate(x)
            print(f'Iter {iter_ctr} cost : {phi[0]}')
            x = x - alpha * J[0]
            delta_x = old_x - x
            iter_ctr += 1
        
        phi, J = self.problem.evaluate(x)
        print(f'Found optimum : {x}')
        print(f'Total nb of iterations : {iter_ctr}')
        print(f'Final cost : {phi[0]}')


        # finally:
        return x 

    def solve_line_search(self) : 
        x = self.problem.getInitializationSample()
        phi, J = self.problem.evaluate(x)

        iter_ctr = 0
        rho_plus = 1.2
        rho_minus = 0.5
        delta_max = 2
        rho_ls = 0.01
        alpha = 1
        delta = -J[0]/np.linalg.norm(J[0])

        while(np.linalg.norm(alpha*delta) >= 0.0001):

            phi, J = self.problem.evaluate(x)
            delta = -J[0]/np.linalg.norm(J[0])
            print(f'Iter {iter_ctr} cost : {phi[0]}')

            phi_ad, J_ad = self.problem.evaluate(x + alpha*delta)
            while phi_ad[0] > phi[0] + rho_ls * J[0].T @ (alpha*delta):
                alpha = rho_minus*alpha
                phi_ad, J_ad = self.problem.evaluate(x + alpha*delta)

            x = x + alpha*delta
            alpha = np.minimum(rho_plus*alpha, delta_max)

            iter_ctr += 1

        phi, J = self.problem.evaluate(x)
        print(f'Found optimum : {x}')
        print(f'Total nb of iterations : {iter_ctr}')
        print(f'Final cost : {phi[0]}')

        # finally:
        return x
