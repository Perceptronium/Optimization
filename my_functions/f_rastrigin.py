import numpy as np
import sys
sys.path.append("..")


from optimization_algorithms.interface.mathematical_program import  MathematicalProgram

class f_rastrigin ( MathematicalProgram ):
    """
    testing stuff
    f_sq = x.T @ C @ x
    grad_f_sq = 2*C @ x
    """

    def __init__(self, a, c, size):
        self.c = c
        self.size = size
        self.a = a

    def evaluate(self, x) :
        """
        This returns the current value of the cost function and the gradient
        """

        phi = np.array([ np.sin(self.a * x[0]), np.sin(self.a*self.c*x[1]), 2*x[0], 2*self.c*x[1] ])
        f = phi.T @ phi
        deriv_phi = np.array([ np.cos(self.a * x[0]) * self.a, np.cos(self.a*self.c*x[1]) * self.a*self.c, 2, 2*self.c ])
        grad_f = (2 * phi.T @ deriv_phi).T

        # print(f'phi.shape : {phi.shape}')
        # print(f'deriv_phi.shape : {deriv_phi.shape}')
        # print(f'f.shape : {f.shape}')
        # print(f'grad_f.shape : {grad_f.shape}')

        return  np.array([f]), grad_f.reshape(1,-1)

    def getDimension(self) : 
        """
        ?
        """
        return self.size

    def getFHessian(self, x) : 
        """
        hessian_f_sq = 2*C
        """
        pass

    def getInitializationSample(self) : 
        """
        """
        x = np.array([-1, 1])
        return x

    def report(self , verbose ): 
        """
        """
        strOut = "f_rastrigin function"
        return  strOut


