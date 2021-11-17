import numpy as np
import sys
# sys.path.append("..")


from ..interface.mathematical_program import  MathematicalProgram

class f_sq ( MathematicalProgram ):
    """
    testing stuff
    f_sq = x.T @ C @ x
    grad_f_sq = 2*C @ x
    """

    def evaluate(self, x) :
        """
        This returns the current value of the cost function and the gradient
        """
        return  np.array(x.T @ self.C @ x), 2*self.C @ x

    def getDimension(self) : 
        """
        ?
        """
        return 2

    def getFHessian(self, x) : 
        """
        hessian_f_sq = 2*C
        """
        return 2*self.C

    def getInitializationSample(self) : 
        """
        """
        x = np.array([1, 1])
        c = 10
        diag = np.zeros(x.shape[0])
  
        for i in range(x.shape[0]):
            diag[i] = c**((i-1)/(x.shape[0]-1))
  
        self.C = np.diag(diag)
        return x

    def report(self , verbose ): 
        """
        """
        strOut = "f_sq function"
        return  strOut


