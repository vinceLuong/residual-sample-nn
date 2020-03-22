# Standard imports
import numpy as np

# Class bias
class Bias():

    def __init__(self, n_nodes, distribution='gaussian'):
        '''
            bias_vector = Bias(n_nodes, type='gaussian')

            Creates a Bias class and saves it in the variable 'b'.

            Inputs:
              n_nodes is the number of nodes in the hidden layer
              distribution is the kind of distribution, the bias follows
        '''
        self.n = n_nodes    # Number of nodes in the hidden layer
        self.dis = distribution # Distribution type
        
        # When prior weight distribution is normal
        if self.dis == 'gaussian':
            self.b = np.random.normal(size=[1,self.n])
            self.mu = np.zeros((1, self.n))
            self.sigma = np.zeros((1, self.n)) + 1
    
    def Sample(self):
        if self.dis == 'gaussian':
            temp = np.random.normal(self.mu, self.sigma, size=[1,self.n])
        return temp

    def Update(self, lst, times):
        self.mu = np.sum(lst, 0) / times
        # self.sigma = np.sqrt(np.sum(np.power(lst, 2)) / times - np.power(self.mu, 2))