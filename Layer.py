# Standard imports.
import numpy as np
from Bias import *

# Class Layer.
class Layer():
    
    def __init__(self, n_nodes, prior_distribution, act='logistic'):
        '''
        Creates a layer object.
            
        @param n_nodes: The number of nodes in the layer.
        @param prior_distribution: the distribution bias follows.
        @param act: Specifies the activation function. Use 'logistic' or 'identity'
        '''
        prior_distribution = str(prior_distribution)
        self.N = abs(int(n_nodes))  # Number of nodes in this layer.
        self.h = []      # Node activities.
        self.z = []
        self.SampledBias = []      # Stores sampled bias from bias_vector.
        self.bias_vector = Bias(self.N, prior_distribution)   # Biases.
        
        # Activation functions
        if act == 'identity':
            self.sigma = self.Identity
            self.sigma_p = self.Identity_p
        elif act == 'logistic':
            self.sigma = self.Logistic
            self.sigma_p = self.Logistic_p
        elif act == 'softmax':
            self.sigma = self.Softmax
            self.sigma_p = None
        else:
            print('Error: Activation function ' + act + ' not implemented')
       
    def Logistic(self):
        '''
        Calculates the logistic output using self.z

        @returns: (logistic output), the logistic output given input self.z.
        '''
        return 1. / (1. + np.exp(-self.z))

    def Logistic_p(self):
        '''
        Calculates derivaitive of logistic function given input.

        @returns: (output of derivaitive of logistic function), given input.
        '''
        return self.h * (1.-self.h)
    
    def Identity(self):
        '''
        Calculates identity output.

        @returns: (self.z), the output for identity function given input.
        '''
        return self.z
    
    def Identity_p(self):
        '''
        Calculates derivative for identity function.

        @returns: (output of derivaitive of identity function), given input.
        '''
        return np.ones_like(self.h)
    
    def Softmax(self):
        '''
        Calculates softmax output.

        @returns: (output for softmax function), given input.
        '''
        v = np.exp(self.z)
        s = np.sum(v, axis=1)
        return v / np.tile(s[:, np.newaxis], [1, np.shape(v)[1]])
