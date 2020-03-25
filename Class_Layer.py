# Standard imports.
import numpy as np
from Class_Bias import *

# Class Layer.
class Layer():
    
    def __init__(self, n_nodes, prior_distribution, act='logistic'):
        '''
            lyr = Layer(n_nodes, act='logistic')
            
            Creates a layer object.
            
            Inputs:
             n_nodes: The number of nodes in the layer.
             prior_distribution: the distribution bias follows.
             act: Specifies the activation function. Use 'logistic' or 'identity'
        '''
        self.N = n_nodes  # Number of nodes in this layer.
        self.h = []      # Node activities.
        self.z = []
        self.b = []      # Stores sampled bias from bias_vector.
        self.bias_vector = Bias(self.N, prior_distribution)   # Biases.
        
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