import numpy as np
import sys
# sys.path.append("..")


from ..interface.mathematical_program import  MathematicalProgram

class f_hole ( MathematicalProgram ):
    """
    testing stuff
    f_hole = (x.T @ C @ x) / (a**2 + x.T @ C @ x)
    grad_f_sq = 2*C @ x
    """

    def evaluate(self, x) :
        """
        This returns the current value of the cost function and the gradient
        """
        f = np.array((x.T @ self.C @ x) / (self.a**2 + x.T @ self.C @ x))
        grad_f = ((2*self.a**2) / (self.a**2 + x.T @ self.C @ x)**2) * self.C @ x
        return  f, grad_f

    def getDimension(self) : 
        """
        ?
        """
        raise NotImplementedError("No need for now")

    def getFHessian(self, x) : 
        """
        hessian_f_hole
        """
        raise NotImplementedError("No need for now")

    def getInitializationSample(self) : 
        """
        """
        x = np.array([1, 1])
        c = 10
        diag = np.zeros(x.shape[0])
  
        for i in range(x.shape[0]):
            diag[i] = c**((i-1)/(x.shape[0]-1))
  
        self.C = np.diag(diag)
        self.a = 0.1

        return x

    def report(self , verbose ): 
        """
        """
        strOut = "f_hole function"
        return  strOut


