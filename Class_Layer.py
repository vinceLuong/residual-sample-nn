# Standard imports
import numpy as np
from Class_Bias import *

# Class Layer
class Layer():
    
    def __init__(self, n_nodes, prior_distribution = None, act='logistic'):
        '''
            lyr = Layer(n_nodes, act='logistic')
            
            Creates a layer object.
            
            Inputs:
             n_nodes  the number of nodes in the layer
             act      specifies the activation function
                      Use 'logistic' or 'identity'
        '''
        self.N = n_nodes  # number of nodes in this layer
        self.h = []      # node activities
        self.z = []
        self.b = []
        self.bias_vector = Bias(self.N, prior_distribution)   # biases
        
        # Activation functions
        self.sigma = self.Logistic
        self.sigma_p = (lambda : self.Logistic_p())
        if act=='identity':
            self.sigma = self.Identity
            self.sigma_p = (lambda : self.Identity_p())
       
    def Logistic(self):
        return 1. / (1. + np.exp(-self.z))
    def Logistic_p(self):
        return self.h * (1.-self.h)
    def Identity(self):
        return self.z
    def Identity_p(self):
        return np.ones_like(self.h)