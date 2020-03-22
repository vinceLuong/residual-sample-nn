# Standard imports
import numpy as np

# Class Weight
class Weight():

    def __init__(self, num_row, num_col, distribution='gaussian'):
        '''
            weight_matrix = Weight(num_row, num_col, type='gaussian')

            Creates a Weight class and saves it in the variable 'weight_matrix'.

            Inputs:
              num_row is the number of rows there will be in the weight matrix
              num_col is the number of columns there will be in the weight matrix
              type can be either 'classifier' or 'regression', and
                  sets the activation function on the output layer,
                  as well as the loss function.
                  'classifier': logistic, cross entropy
                  'regression': linear, mean squared error
        '''
        self.m = num_row    # Number of rows
        self.n = num_col    # Number of columns
        self.dis = distribution # Distribution type
        
        # When prior weight distribution is normal
        if self.dis == 'gaussian':
            self.mu = np.zeros((self.m, self.n))
            self.sigma = np.zeros((self.m, self.n)) + 1
    
    def Sample(self):
        if self.dis == 'gaussian':
            temp = np.random.normal(self.mu, self.sigma, size=[self.m,self.n])
        return temp

    def Update(self, lst, times):
        self.mu = np.sum(lst, 0) / times
        # self.sigma = np.sqrt(np.sum(np.power(lst, 2)) / times - np.power(self.mu, 2))
