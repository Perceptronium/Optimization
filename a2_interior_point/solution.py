import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverInteriorPoint(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """

        """
        Food for thought :
        The interior point LogBarrier method comprises two main loops : 
        An outer loop, which draws the central path
        And an inner loop which walks down each new step of the central path
        Overall, in LogBarrier methods, we are thus approaching the optimum "from the inside"
        Following the recommendation, we will be using the optimizer from a1_unconstrained_solver
        CLASSIC NEWTON WITH GAUSS-NEWTON FALLBACK for the inner loop
        
        Relaxations : 
        Starting point guaranteed to be feasible
        Hessian of constraint assumed to be 0

        Requirements :
        x is feasible at all times (interior point method)
        delta < 0.001
        minimize the number of queries
        maximum of 10000 queries
        limit compute time
        """

    #=========================================
    # COPYING AND ADAPTING HELPER FUNCTIONS FROM A1
    #=========================================
    def init_value(self):
        x = self.problem.getInitializationSample()
        return x

    def evaluate(self, x, index_f, index_r, index_ineq, mu):
        phi, J = self.problem.evaluate(x)
        c = 0
        jacob = 0
        if len(index_f) > 0:
            c += phi[index_f][0]
            jacob += J[index_f][0]

        if len(index_r) > 0:
            c += phi[index_r].T @ phi[index_r]
            jacob += 2 * J[index_r].T @ phi[index_r] # Jacobian of F if there's a SOS

        if len(index_ineq) > 0:
            c += -mu * np.sum(np.log(-phi[index_ineq])) # Adding barrier to the cost function
            if(np.isnan(c)):
                c = np.inf
            jacob += -mu * np.sum( J[index_ineq] / phi[index_ineq][:,None], axis = 0 ) # Jacobian of F if there's an ineq constraint
        return c, jacob

    def compute_hessian(self, x, J, index_f, index_r, index_ineq, mu):
        H = 0
        phi, J = self.problem.evaluate(x) # using Jacobian of phi
        if len(index_f) > 0:
            H += self.problem.getFHessian(x)

        if len(index_r) > 0:
           H += 2 * J[index_r].T @ J[index_r]
           
        # HESSIAN OF CONSTRAINTS APPROXIMATED AS 0
        if len(index_ineq) > 0:
            tmp_constraints = np.array([np.outer(J[index_ineq][i], J[index_ineq][i]) for i in range(J[index_ineq].shape[0])])
            H += np.sum( tmp_constraints , axis = 0 )
        
        return H

    def compute_lambda(self, H):
        eigs = np.linalg.eigvals(H)       # Initial eigenvalues of Hessian       
        if np.all(eigs > 0):              # No need for damping
            lambd = 0
        else:
            lambd = -np.amin(eigs) + 0.1   # Damping by at least 0.1 and minimal eigval
        return lambd

    def compute_delta(self, H, lambd, J):
        try:
            delta = np.linalg.solve(H + lambd * np.eye(self.problem.getDimension()),-J)
        except np.linalg.LinAlgError:
            delta = - J / np.linalg.norm(J) 

        if (J.T @ delta) > 0: # Non_descent
            delta = - J / np.linalg.norm(J)

        return delta

    
    def centering(self, x, mu, index_f, index_r, index_ineq):
        """
        Solving unconstrained problem argmin f(x) - mu * sum[ log( -g(x) ) ]
        """

        #=========================================
        # INITS
        #=========================================
        #print("====INNER LOOP====")
        watchdog = 1000                   # Maximum allowed number of queries
        iter_ctr = 0                      # Iteration counter
        rho_plus = 1.2                    # Stepsize increaser
        rho_minus = 0.5                   # Stepsize decreaser
        rho_ls = 0.01                     # Stepsize conditioner
        alpha = 1                         # Stepsize
        theta = 0.0001                     # Tolerance

        #x = self.problem.getInitializationSample()       # Initalize starting point
        #print(f'Inner loop starting point : {x}')
        #phi, J = self.evaluate(x, index_f, index_r, index_ineq, mu) # Initial values for cost and Jacobian
           
        #H = self.compute_hessian(x, J, index_f, index_r, index_ineq, mu) # Initial value for Hessian
        #lambd = self.compute_lambda(H)                   # Computing initial lambda

        x_old = np.inf

        #==================================================
        # CORE : CLASSIC NEWTON WITH GAUSS-NEWTON FALLBACK
        #==================================================
        while(np.linalg.norm(x_old - x) >= theta):
            x_old = x  
            phi, J = self.evaluate(x, index_f, index_r, index_ineq, mu) # Updating current cost and Jacobian values
             
            H = self.compute_hessian(x, J, index_f, index_r, index_ineq, mu) # Updating current Hessian values
            #print(f'Iter {iter_ctr} cost : {phi}') # Uncomment this for more verbosity

            lambd = self.compute_lambda(H) # Updating lambda
            delta = self.compute_delta(H, lambd, J) # Updating delta

            phi_ad, _ = self.evaluate(x + alpha*delta, index_f, index_r, index_ineq, mu) # Computing f(x + alpha*delta)
            while phi_ad > phi + rho_ls * J.T @ (alpha*delta): # Doing linesearch
                alpha = rho_minus*alpha # Decreasing stepsize
                phi_ad, _ = self.evaluate(x + alpha*delta, index_f, index_r, index_ineq, mu) # Updating f(x + alpha*delta)

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
        #phi, J = self.evaluate(x, index_f, index_r, index_ineq, mu)
        #print(f'Inner optimum : {x}')
        #print(f'Inner nb of iterations : {iter_ctr}')
        #print(f'Inner cost : {phi}')

        # finally:
        return x
  
    

    
    
    #=========================================
    # IMPLEMENTING LOGBARRIER
    #=========================================
    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """
        #=========================================
        # LEARNING ABOUT THE PROBLEM
        #=========================================
        print(f"====New Problem====")
        types = self.problem.getFeatureTypes()
        index_f = [i for i, x in enumerate(types) if x == OT.f] # Get all features of type f
        assert( len(index_f) <= 1 ) # At most, only one term of type OT.f
        index_r = [i for i, x in enumerate(types) if x == OT.sos] # Get all sum-of-square features
        index_ineq = [i for i,x in enumerate(types) if x == OT.ineq] # Get all inequality features
        #print(f'Index F : {index_f}')
        #print(f'Index SOS : {index_r}')
        #print(f'Index INEQ : {index_ineq}')

        #=========================================
        # INITS
        #=========================================
        watchdog_outer = 10000                  # Maximum allowed number of queries
        iter_ctr_outer = 0                      # Iteration counter
        mu_zero = 1                       # Initial mu
        rho_mu = 0.5                      # mu decreaser
        theta_outer = 0.001                     # Tolerance

        #=========================================
        # CORE : LOG BARRIER METHOD
        #=========================================
        x = self.problem.getInitializationSample() # Assumed feasible
        print(f'Outer loop starting point : {x}')

        x_old_outer = np.inf
        mu = mu_zero
        while(np.linalg.norm(x_old_outer - x) >= theta_outer):
            #print(f'Outer delta : {np.linalg.norm(x_old_outer - x)}')
            x_old_outer = x
            x = self.centering(x, mu, index_f, index_r, index_ineq)
            mu *= rho_mu
            #print(f'mu : {mu}')

            iter_ctr_outer += 1 # Updating iteration counter
            if iter_ctr_outer == watchdog_outer: # If watchdog is awaken, abort program and return -1
                print("Suspected divergence. Aborting program.")
                return -1

            
            # Check if x is still in feasible region
            phi_sanity, _ = self.problem.evaluate(x)
            if not np.all(phi_sanity[index_ineq] <= 0):
                print("x trespassed to unfeasible region. Aborting program.")
                return -1

        
        #print(f'Outer delta : {np.linalg.norm(x_old_outer - x)}')
        phi, J = self.evaluate(x, index_f, index_r, index_ineq, mu)
        print(f'Final optimum : {x}')
        print(f'Outer nb of iterations : {iter_ctr_outer}')
        print(f'Final cost : {phi}')
        # finally:
        return x


