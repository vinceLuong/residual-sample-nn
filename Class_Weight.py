# Standard imports
import numpy as np

# Class Weight
class Weight():

    def __init__(self, num_row, num_col, distribution='gaussian'):
        '''
            weight_matrix = Weight(num_row, num_col, type='gaussian')

            Creates a Weight class and saves it in the variable "weight_matrix" in class Network.

            Inputs:
              num_row: The number of rows there will be in the weight matrix.
              num_col: The number of columns there will be in the weight matrix.
              distribution: The distribution class Bias follows. Default is Gaussian.
        '''
        self.m = num_row    # Number of rows
        self.n = num_col    # Number of columns
        self.dis = distribution # Distribution type
        # When prior weight distribution is normal
        if self.dis == 'gaussian':
            #self.mu = np.zeros((self.m, self.n))
            self.sigma = np.random.uniform(low=0, high=1, size=(self.m, self.n))
            self.mu = np.random.uniform(low=-1, high=1, size=(self.m, self.n))
    

    def Initialize_Bootstrap(self, times):
        '''
            weight_matrix[index].Initialize_Bootstrap(times)

            Initialize the 3D matrix within the class we sample from.

            Inputs:
              times: The number of times we sample the weight.
        '''
        self.bootstrap_matrix = []  # Initialze the matrix.
        if self.dis == 'gaussian':
            for _ in range(times):
                # Sample from the distribution and append to matrix.
                weight = np.random.normal(self.mu, self.sigma, size=[self.m,self.n])
                self.bootstrap_matrix.append(weight)

    def Sample(self, bootstrap):
        '''
            weight = Sample(bootstrap)

            Sample the data from the distribution class Weight follows.

            Inputs:
              bootstrap: Boolean. If true, using bootstrap to sample.
                  Otherwise, sample using distribution parameters.
            
            Outputs:
              bias: The weight used in Feedforward.
        '''
        if self.dis == 'gaussian':
            # When bootstrap == False, sample from distribution directly.
            if not bootstrap:
                weight = np.random.normal(self.mu, self.sigma, size = [self.m,self.n])
            else:
                # Otherwise using bootstrap.
                max_len = len(self.bootstrap_matrix)
                idx_matrix = np.random.randint(max_len, size = (self.m, self.n))
                weight = self.idx2element(idx_matrix, self.bootstrap_matrix)
        return weight

    def idx2element(self, index_matrix, data):
        '''
            bootstrap_data = idx2element(index_vector, data)

            Using index matrix to get element matrix with same dimension.

            Inputs:
              index_vector: Vector of size (1 x self.n). Elements are randomly selected indices.
              data: Matrix of size (times x 1 x self.n). Each element is "weight" sampled from distribution.
            
            Outputs:
              bootstrap_data: Element vector corresponding to index_vector from data
        '''
        bootstrap_data = index_matrix   # Same size.
        for i in range(len(index_matrix)):
            for j in range(len(index_matrix[0])):
            # Using the index to select element.
                bootstrap_data[i][j] = data[index_matrix[i][j]][i][j]
        return bootstrap_data

    def Update(self, lst, times, bootstrap):
        '''
            weight_matrix[index].Update(lst, times)

            Update distribution parameters using many samples.

            Inputs:
              lst: A list of samples we use to update.
              times: Number of times we sampled. 
                  It's also the number of elements in lst.
              bootstrap: Boolean, Whether on or not to bootstrap
        '''
        self.bootstrap_matrix = lst # Update bootstrap_matrix.

        if bootstrap:
            # Update mu using sample mean.
            self.mu = np.sum(lst, 0) / times
        # When normal, the distribution parameters are mu and sigma.
        elif self.dis == 'gaussian':
            # Update mu using sample mean.
            self.mu = np.sum(lst, 0) / times
            # Update sigma using sample standard deviation.
            sigma = np.zeros(lst[0].shape)
            for i in range(len(lst)):
                difference = lst[i] - self.mu
                sigma = sigma + np.power(difference, 2) / len(lst)
            self.sigma = sigma
