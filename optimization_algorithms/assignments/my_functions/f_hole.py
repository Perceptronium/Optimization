import numpy as np
import sys
sys.path.append("..")


from optimization_algorithms.interface.mathematical_program import  MathematicalProgram

class f_hole ( MathematicalProgram ):
    """
    testing stuff
    f_hole = (x.T @ C @ x) / (a**2 + x.T @ C @ x)
    grad_f_sq = 2*C @ x
    """

    def __init__(self,a,c,size):
        self.a = a
        self.c = c
        self.size = size

        diag = [self.c**((i-1)/(size-1)) for i in range(size)]
        self.C = np.diag(diag)

    def evaluate(self, x) :
        """
        This returns the current value of the cost function and the gradient
        """
        f = np.array((x.T @ self.C @ x) / (self.a**2 + x.T @ self.C @ x))
        grad_f = ((2*self.a**2) / (self.a**2 + x.T @ self.C @ x)**2) * self.C @ x
        return  np.array([f]), grad_f.reshape(1,-1)

    def getDimension(self) : 
        """
        ?
        """
        raise NotImplementedError("No need for now")

    def getFHessian(self, x) : 
        """
        hessian_f_hole
        """
        # Imported solution from MatrixCalculus.org
        t_0 = (self.a ** 2)
        t_1 = (self.C) @ (x)
        t_2 = (t_0 + (x) @ (t_1))
        t_3 = ((2 * t_0) / (t_2 ** 2))
        t_4 = ((4 * t_0) / (t_2 ** 3))
        return ((t_3 * self.C) - ((t_4 * np.multiply.outer(t_1, t_1)) + (t_4 * np.multiply.outer(t_1, (x).dot(self.C)))))

        

    def getInitializationSample(self) : 
        """
        """
        x = np.array([1, 1])
        return x

    def report(self , verbose ): 
        """
        """
        strOut = "f_hole function"
        return  strOut


