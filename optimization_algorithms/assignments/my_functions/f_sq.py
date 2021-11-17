import numpy as np
import sys
sys.path.append("..")


from optimization_algorithms.interface.mathematical_program import  MathematicalProgram

class f_sq ( MathematicalProgram ):
    """
    testing stuff
    f_sq = x.T @ C @ x
    grad_f_sq = 2*C @ x
    """

    def __init__(self, c, size):
        self.c = c
        self.size = size
        #for i in range(size):
            #diag[i] = c**((i-1)/(size-1))
        diag = [self.c**((i-1)/(size-1)) for i in range(size)]
        self.C = np.diag(diag)

    def evaluate(self, x) :
        """
        This returns the current value of the cost function and the gradient
        """
        return  np.array([x.T @ self.C @ x]), (2*self.C @ x).reshape(1,-1)

    def getDimension(self) : 
        """
        ?
        """
        return self.size

    def getFHessian(self, x) : 
        """
        hessian_f_sq = 2*C
        """
        return 2*self.C

    def getInitializationSample(self) : 
        """
        """
        x = np.array([1, 1])
        return x

    def report(self , verbose ): 
        """
        """
        strOut = "f_sq function"
        return  strOut


