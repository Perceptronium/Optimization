import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverUnconstrained(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """

        """
            Food for thought :
            The cost function is a linear combination of non-linear function that are not guaranteed convex.
            Trying to optimize using plain gradient descent thus wouldn't necessarily make plenty of sense here.
            Here are the conditions for our algorithm :
                On a purely quadratic cost, it should require less than 10 iterations to converge.
                    => Sign for Newton-ish optimization
                On a linear LS problem, it should also require less than 10 iterations to converge.
                    => Sign for Gauss-Newton approximation
                It should converge on non-convex cost functions :
                    => Sign for a robust Newton-ish optimization
                The precision of the returned solution should be Delta <= 0.001, unique optimum
                    => Not really sure what to say here
                It should minimize the number of queries :
                    => Sign for a Newton-ish optimization
                It should be "fairly independant" of the dimensionality :
                    => Sign for a Newton-ish approximation
                Maximum of 1000 queries:
                    => Fix a watchdog

            Based on all those ""constraints"" (see what I did there?) and considering 
            the algorithms seen in lecture, I think it should make sense to go for a Newton variation.
            Options :
                Classic Newton : Not very robust
                Levenberg-Marquardt : Works on F but not SOS
                Gauss-Newton : Works on SOS but not F
                BFGS / L-BFGS : Only needed for big dimensions
                Conjugate gradient : Wouldn't scale well
        """

    
    
    def init_value(self):
        x = self.problem.getInitializationSample()
        return x

   
    def evaluate(self, x, index_f, index_r):
        phi, J = self.problem.evaluate(x)
        c = 0
        jacob = 0
        if len(index_f) > 0:
            c += phi[index_f][0]
            jacob += J[index_f][0]
        if len(index_r) > 0:
            c += phi[index_r].T @ phi[index_r]
            jacob += 2 * J[index_r].T @ phi[index_r] # Jacobian of F if there's a SOS
        
        return c, jacob

    
    def compute_hessian(self, x, J, index_f, index_r):
        H = 0
        _, J = self.problem.evaluate(x) # using Jacobian of phi
        if len(index_f) > 0:
            H += self.problem.getFHessian(x)
        if len(index_r) > 0:
           H += 2 * J[index_r].T @ J[index_r]
        
        return H
    

    def compute_lambda(self, H):
        eigs = np.linalg.eigvals(H)       # Initial eigenvalues of Hessian       
        if np.all(eigs > 0):              # No need for damping
            lambd = 0
        else:
            lambd = -np.amin(eigs) + 0.1   # Damping by at least 0.01 and minimal eigval
        return lambd
    

    def compute_delta(self, H, lambd, J):
        try:
            delta = np.linalg.solve(H + lambd * np.eye(self.problem.getDimension()),-J)
        except np.linalg.LinAlgError:
            delta = - J / np.linalg.norm(J) 

        if (J.T @ delta) > 0: # Non_descent
            delta = - J / np.linalg.norm(J)

        return delta


    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """
        
        #=========================================
        # LEARNING ABOUT THE PROBLEM
        #=========================================
        print("========================")
        types = self.problem.getFeatureTypes()
        index_f = [i for i, x in enumerate(types) if x == OT.f] # Get all features of type f
        assert( len(index_f) <= 1 ) # At most, only one term of type OT.f
        index_r = [i for i, x in enumerate(types) if x == OT.sos] # Get all sum-of-square features
        print(f'Index F : {index_f}')
        print(f'Index R : {index_r}')

        
        #=========================================
        # INITS
        #=========================================
        watchdog = 1000                   # Maximum allowed number of queries
        iter_ctr = 0                      # Iteration counter
        rho_plus = 1.2                    # Stepsize increaser
        rho_minus = 0.5                   # Stepsize decreaser
        rho_ls = 0.01                     # Stepsize conditioner
        alpha = 1                         # Stepsize
        theta = 0.001                     # Tolerance

        x = self.problem.getInitializationSample()       # Initalize starting point
        print(f'Starting point : {x}')
        phi, J = self.evaluate(x, index_f, index_r)      # Initial values for cost and Jacobian
           
        H = self.compute_hessian(x, J, index_f, index_r) # Initial value for Hessian
        lambd = self.compute_lambda(H)                   # Computing initial lambda

        x_old = np.inf

        #==================================================
        # CORE : CLASSIC NEWTON WITH GAUSS-NEWTON FALLBACK
        #==================================================
        while(np.linalg.norm(x_old - x) >= theta):
            x_old = x  
            phi, J = self.evaluate(x, index_f, index_r) # Updating current cost and Jacobian values
             
            H = self.compute_hessian(x, J, index_f, index_r) # Updating current Hessian values
            #print(f'Iter {iter_ctr} cost : {phi}') # Uncomment this for more verbosity

            lambd = self.compute_lambda(H) # Updating lambda
            delta = self.compute_delta(H, lambd, J) # Updating delta

            phi_ad, _ = self.evaluate(x + alpha*delta, index_f, index_r) # Computing f(x + alpha*delta)
            while phi_ad > phi + rho_ls * J.T @ (alpha*delta): # Doing linesearch
                alpha = rho_minus*alpha # Decreasing stepsize
                phi_ad, _ = self.evaluate(x + alpha*delta, index_f, index_r) # Updating f(x + alpha*delta)

            x = x + alpha*delta # Accepting step
            #alpha = np.minimum(rho_plus*alpha, 1) # Incresaing stepsize
            alpha = 1 # Resetting alpha works better on the test functions
            
            
            iter_ctr += 1 # Updating iteration counter
            if iter_ctr == watchdog: # If watchdog is awaken, abort program and return -1
                print("Suspected divergence. Aborting program.")
                return -1
            #print(np.linalg.norm(x_old - x)) # Uncomment this for more verbosity
        

        
        #=========================================
        # RESULTS
        #=========================================           
        phi, J = self.evaluate(x, index_f, index_r)
        print(f'Found optimum : {x}')
        print(f'Total nb of iterations : {iter_ctr}')
        print(f'Final cost : {phi}')

        # finally:
        return x
