import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.nlp_solver import  NLPSolver

class GENERAL_SOLVER(NLPSolver):

    def __init__(self, algorithm = 'gradient_descent'):
        """
        Available algorithms :
        - gradient_descent
        - gradient_descent_bls
        - newton_method

        See also:
        ----
        NLPSolver.__init__
        """
        self.algorithm = algorithm

    
    def init_value(self):
        x = self.problem.getInitializationSample()
        return x
    

    def evaluate(self, x):
        phi, J = self.problem.evaluate(x)
        return phi[0], J[0]


    def solve(self) :
        """
        See Also:
        ----
        NLPSolver.solve

        """
        x = self.init_value()
        phi, J = self.evaluate(x)
        
        if self.algorithm == 'gradient_descent':
            delta_x = x
            iter_ctr = 0
            alpha = 0.1 # Needs 0.01 for f_hole convergence
            while(np.linalg.norm(delta_x) >= 0.0001):
                old_x = x
                phi, J = self.evaluate(x)
                print(f'Iter {iter_ctr} cost : {phi}')
                x = x - alpha * J
                delta_x = old_x - x
                iter_ctr += 1
            
            phi, J = self.evaluate(x)
            print(f'Found optimum : {x}')
            print(f'Total number of iterations : {iter_ctr}')
            print(f'Final cost : {phi}')

        
        
        elif self.algorithm == 'gradient_descent_bls':
            iter_ctr = 0
            rho_plus = 1.2
            rho_minus = 0.5
            delta_max = 2
            rho_ls = 0.01
            alpha = 1
            delta = -J/np.linalg.norm(J)

            while(np.linalg.norm(alpha*delta) >= 0.0001):

                phi, J = self.evaluate(x)
                delta = -J/np.linalg.norm(J)
                print(f'Iter {iter_ctr} cost : {phi}')

                phi_ad, J_ad = self.evaluate(x + alpha*delta)
                while phi_ad > phi + rho_ls * J.T @ (alpha*delta):
                    alpha = rho_minus*alpha
                    phi_ad, J_ad = self.evaluate(x + alpha*delta)

                x = x + alpha*delta
                alpha = np.minimum(rho_plus*alpha, delta_max)

                iter_ctr += 1

            phi, J = self.evaluate(x)
            print(f'Found optimum : {x}')
            print(f'Total nb of iterations : {iter_ctr}')
            print(f'Final cost : {phi}')  


        elif self.algorithm == 'newton_method':
            H = self.problem.getFHessian(x)
            eigs = np.linalg.eigvals(H)
            
            if np.all(eigs > 0):
                lambd = 0
            else:
                lambd = np.amin(eigs) + 0.1 # Minimal eigval will be 0.1

            iter_ctr = 0
            rho_plus = 1.2
            rho_minus = 0.5
            rho_ls = 0.01
            alpha = 1
            phi, J = self.evaluate(x)
            
            try :
                delta = np.linalg.solve(H + lambd * np.eye(2), -J)
            except :    
                delta = -J/np.linalg.norm(J) # Ill-defined
            
            #if J.T @ delta > 0 : # Non-descent
                #delta = -J/np.linalg.norm(J)

            while(np.linalg.norm(alpha*delta) >= 0.01):

                phi, J = self.evaluate(x)
                #H = self.problem.getFHessian(x)
                delta = np.linalg.solve(H + lambd * np.eye(2), -J)
                print(f'Iter {iter_ctr} cost : {phi}')

                phi_ad, _ = self.evaluate(x + alpha*delta)
                while phi_ad > phi + rho_ls * J.T @ (alpha*delta):
                    alpha = rho_minus*alpha
                    phi_ad, _ = self.evaluate(x + alpha*delta)

                x = x + alpha*delta
                alpha = np.minimum(rho_plus*alpha, 1)

                iter_ctr += 1
            
            phi, J = self.evaluate(x)
            print(f'Found optimum : {x}')
            print(f'Total nb of iterations : {iter_ctr}')
            print(f'Final cost : {phi}')
            

            
            

        
        else:
            print(f'Unknown algorithm : {self.algorithm}.')


        
        return x
