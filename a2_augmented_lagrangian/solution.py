import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverAugmentedLagrangian(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """

        """
        Assumption : 
        No assumptions on starting point
        Hessian of constraints assumed to be 0

        Requirements :
        Returned solution is at most 0.001 infeasible
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

    def evaluate(self, x, index_f, index_r, index_ineq, mu, index_eq, lam, nu, kappa):
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
            OR = np.array([(phi[index_ineq][i] >= 0) or (lam[i] > 0) for i in range(phi[index_ineq].shape[0])])
            c += mu * np.sum (OR * phi[index_ineq]**2, axis = 0) + np.sum(lam * phi[index_ineq], axis = 0)
            jacob +=  2 * mu * (OR * phi[index_ineq]) @ J[index_ineq] + lam.T @ J[index_ineq]


        if len(index_eq) > 0:
            c += nu * np.sum(phi[index_eq]**2, axis = 0) + np.sum(kappa * phi[index_eq], axis = 0)
            jacob += nu * np.sum(2 * phi[index_eq] @ J[index_eq], axis=0) + np.sum(kappa * J[index_eq], axis = 0)

        return c, jacob



    def compute_hessian(self, x, J, index_f, index_r, index_ineq, mu, index_eq, lam, nu, kappa):
        H = 0
        phi, J = self.problem.evaluate(x) # using Jacobian of phi
        if len(index_f) > 0:
            H += self.problem.getFHessian(x)
        
        if len(index_r) > 0:
           H += 2 * J[index_r].T @ J[index_r]
        
        # HESSIAN OF CONSTRAINTS APPROXIMATED AS 0
        if len(index_ineq) > 0:
            H += 2 * mu * np.sum([np.outer(J[index_ineq][i], J[index_ineq][i]) for i in range(J[index_ineq].shape[0])], axis=0)
        
        if len(index_eq) > 0:
            H += 2 * nu * np.sum(J[index_eq] @ J[index_eq].T, axis = 0)

        #print(H)
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
            delta = np.linalg.solve(H + lambd * np.eye(self.problem.getDimension()), -J)
        except np.linalg.LinAlgError:
            delta = - J / np.linalg.norm(J) 

        if (J.T @ delta) > 0: # Non_descent
            delta = - J / np.linalg.norm(J)
        
        #print(f'delta : {delta}')
        return delta

    
    def centering(self, x, mu, index_f, index_r, index_ineq, index_eq, lam, nu, kappa):
        """
        Solving unconstrained problem argmin f(x) - mu * sum[ log( -g(x) ) ]
        """

        #=========================================
        # INITS
        #=========================================
        #print("====INNER LOOP====")
        watchdog = 1000                  # Maximum allowed number of queries
        iter_ctr = 0                      # Iteration counter
        rho_plus = 1.2                    # Stepsize increaser
        rho_minus = 0.5                   # Stepsize decreaser
        rho_ls = 0.01                     # Stepsize conditioner
        alpha = 1                         # Stepsize
        theta = 0.001                     # Tolerance

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
            phi, J = self.evaluate(x, index_f, index_r, index_ineq, mu, index_eq, lam, nu, kappa) # Updating current cost and Jacobian values
            H = self.compute_hessian(x, J, index_f, index_r, index_ineq, mu, index_eq, lam, nu, kappa) # Updating current Hessian values
            #print(f'Iter {iter_ctr} cost : {phi}') # Uncomment this for more verbosity

            lambd = self.compute_lambda(H) # Updating lambda
            delta = self.compute_delta(H, lambd, J) # Updating delta

            phi_ad, _ = self.evaluate(x + alpha*delta, index_f, index_r, index_ineq, mu, index_eq, lam, nu, kappa) # Computing f(x + alpha*delta)
            while phi_ad > (phi + rho_ls * J.T @ (alpha*delta)): # Doing linesearch
                alpha = rho_minus*alpha # Decreasing stepsize
                phi_ad, _ = self.evaluate(x + alpha*delta, index_f, index_r, index_ineq, mu, index_eq, lam, nu, kappa) # Updating f(x + alpha*delta)

            #print(alpha)
            x = x + alpha*delta # Accepting step
            #alpha = np.minimum(rho_plus*alpha, 1) # Incresaing stepsize
            alpha = 1 # Resetting alpha works better on the test functions
            
            
            iter_ctr += 1 # Updating iteration counter
            if iter_ctr == watchdog: # If watchdog is awaken, abort program and return -1
                print("Suspected divergence inner. Aborting program.")
                break
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
    # IMPLEMENTING AUGMENTED LAG
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
        index_eq = [i for i, x in enumerate(types) if x == OT.eq] # Get all equality features
        #print(f'Index F : {index_f}')
        #print(f'Index SOS : {index_r}')
        #print(f'Index INEQ : {index_ineq}')
        #print(f'Index EQ : {index_eq}')

        #=========================================
        # INITS
        #=========================================
        watchdog_outer = 50                  # Maximum allowed number of queries
        iter_ctr_outer = 0                      # Iteration counter
        mu_zero = 1                             # Initial mu
        nu_zero = 1                             # Initial mu
        rho_mu = 1.2                            # mu increaser
        rho_nu = 1.2                            # nu increaser
        theta_outer = 0.001                     # Tolerance
        epsilon = 0.001                         # Infeasibility tolerance
        lam = np.zeros_like(index_ineq)         # Lagrange lambda
        kappa = np.zeros_like(index_eq)         # Lagrange kappa

        #=========================================
        # CORE : AUGMENTED LAG METHOD
        #=========================================
        x = self.problem.getInitializationSample()
        print(f'Outer loop starting point : {x}')

        x_old_outer = np.inf
        mu = mu_zero
        nu = nu_zero

        phi, J = self.problem.evaluate(x)
        
        while(1):

            #print(f'Outer delta : {np.linalg.norm(x_old_outer - x)}')
            x_old_outer = x
            x = self.centering(x, mu, index_f, index_r, index_ineq, index_eq, lam, nu, kappa)

            #print(x)
            phi, J = self.problem.evaluate(x)
            lam = np.maximum(lam + 2 * mu * phi[index_ineq], np.zeros_like(lam))
            kappa = kappa + 2 * nu * phi[index_eq]
            mu = rho_mu * mu
            nu = rho_nu * nu



            iter_ctr_outer += 1 # Updating iteration counter
            if iter_ctr_outer == watchdog_outer: # If watchdog is awaken, abort program and return -1
                print("Suspected divergence outer. Aborting program.")
                break

            if np.linalg.norm(x_old_outer - x) <= theta_outer\
             and np.all(phi[index_ineq] < epsilon)\
                  and np.linalg.norm(phi[index_eq]) < epsilon:
                  break
            

        
        print(f'Outer delta : {np.linalg.norm(x_old_outer - x)}')
        phi, J = self.evaluate(x, index_f, index_r, index_ineq, mu, index_eq, lam, nu, kappa)
        print(f'Final optimum : {x}')
        print(f'Outer nb of iterations : {iter_ctr_outer}')
        print(f'Final cost : {phi}')
        # finally:
        return x