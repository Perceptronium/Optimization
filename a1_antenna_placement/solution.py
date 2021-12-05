import math
import sys
from typing_extensions import ParamSpec
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class AntennaPlacement(MathematicalProgram):
    """
    """

    def __init__(self, P, w):
        """
        Arguments
        ----
        P: list of 1-D np.arrays
        w: 1-D np.array
        """
        # in case you want to initialize some class members or so...
        self.P = np.array(P) # Matrix 
        self.w = w # Vector
        #print(f'w shape : {self.w.shape}')
        #print(f'P shape : {self.P.shape}')



    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """
        y = 0
        J = 0
        # Could be improved with vectorizing but I found this more readable
        for i, pop in enumerate(self.w):
            sub = x - self.P[i]
            inside_exp = - np.inner(sub,sub)
            potential = np.exp(inside_exp)
            y -= pop * potential
            J += 2 * pop * potential * sub

        return np.array([y]) , J.reshape(1, -1)

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return 2

    def getFHessian(self, x):
        """
        See Also
        ------
        MathematicalProgram.getFHessian
        """
        # Could be improved with vectorizing but I found this more readable
        H = 0
        for i, pop in enumerate(self.w):
            sub = x - self.P[i]
            inside_exp = - np.inner(sub, sub)
            potential = np.exp(inside_exp)
            H += ( - 4 * pop * potential * np.outer(sub,sub) ) + ( 2 * pop * potential * np.eye(len(self.w)))

        return H.T # Hessian is transpose of grad-of-grad by convention

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        x0 = np.mean(self.P, axis = 0) # Geometric mean
        return x0

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types

        """
        return [OT.f]
