import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class RobotTool(MathematicalProgram):
    """
    """

    def __init__(self, q0, pr, l):
        """
        Arguments
        ----
        q0: 1-D np.array
        pr: 1-D np.array
        l: float
        """
        
        """
            Food for thought : 
                We are asked for the implementation to only contain terms of type SOS
                The cost function J(q) = (p-p*).T@(p-p*) + lambda*(q-q0).T@(q-q0) can be
                seen as a SOS with features phi_1 = p-p* and phi_2 = lambda**0.5 * (q-q0),
                each containing their own features as well. q is in R3 and p is in R2 for a
                total feature amount of 5.
        """
        self.q0 = q0 # Initial position
        self.pr = pr # Desired position
        self.l = l # Regularization lambda

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        #####################################
        # COMPUTING FEATURES
        #####################################

        q1 = x[0]
        q2 = x[1]
        q3 = x[2]
        p1 = np.cos(q1) + 0.5 * np.cos(q1 + q2) + (1/3) * np.cos(q1 + q2 + q3)
        p2 = np.sin(q1) + 0.5 * np.sin(q1 + q2) + (1/3) * np.sin(q1 + q2 + q3)
        p = np.array([p1,p2])
        phi_1 = p - self.pr # 2 features
        phi_2 = self.l**0.5 * (x - self.q0) # 3 features

        y = np.concatenate([phi_1, phi_2]) # All features

        #######################################
        # COMPUTING FEATURE JACOBIANS
        #######################################

        dp11 = - p2
        dp12 = - (p2 - np.sin(q1)) 
        dp13 = - (p2 - np.sin(q1) - 0.5*np.sin(q1 + q2))
        dp21 = p1 
        dp22 = p1 - np.cos(q1)
        dp23 = p1 - np.cos(q1) - 0.5*np.cos(q1 + q2)
        Jp1 = np.array([dp11, dp12, dp13]) # 1x3 vector
        Jp2 = np.array([dp21, dp22, dp23]) # 1x3 vector
        Jphi_1 = np.vstack((Jp1, Jp2))     # 2x3 matrix

        Jphi_2 = self.l**0.5 * np.eye(x.shape[0]) # 3x3 matrix
        
        J = np.vstack((Jphi_1, Jphi_2))

        # y is a 1-D np.array of dimension m
        # J is a 2-D np.array of dimensions (m,n)
        # where m is the number of features and n is dimension of x
        return  y, J

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return self.q0.shape[0]

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return self.q0

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.sos] * 5
