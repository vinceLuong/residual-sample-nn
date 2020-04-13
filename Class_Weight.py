# Standard imports
import numpy as np

# Class Weight
class Weight():

    def __init__(self, num_row, num_col, distribution='gaussian'):
        '''
        Creates a Weight class and saves it in the variable "weight_matrix" in class Network.

        @param num_row: The number of rows there will be in the weight matrix.
        @param num_col: The number of columns there will be in the weight matrix.
        @param distribution: The distribution class Bias follows. Default is Gaussian.
        @returns: None.
        '''
        self.m = num_row    # Number of rows
        self.n = num_col    # Number of columns
        self.dis = distribution # Distribution type
        # When prior weight distribution is normal
        if self.dis == 'gaussian':
            #self.mu = np.zeros((self.m, self.n))
            self.mu = np.zeros((self.m, self.n))
            self.sigma = np.ones((self.m, self.n))
    
    def Initialize_Bootstrap(self, times):
        '''
        Initialize the 3D matrix within the class we sample from.

        @param times: The number of times we sample the weight from prior distribution.
        @returns: None.
        '''
        self.bootstrap_matrix = []  # Initialze the matrix.
        if self.dis == 'gaussian':
            for _ in range(times):
                # Sample from the distribution and append to matrix.
                weight = np.random.normal(self.mu, self.sigma, size=[self.m,self.n])
                self.bootstrap_matrix.append(weight)

    def Sample(self, bootstrap):    
        '''
        Sample the data from the distribution class Weight follows.

        @param bootstrap: Boolean. If true, using bootstrap to sample.
            Otherwise, sample using distribution parameters.
        @returns: (bias), the weight used in Feedforward.
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
        Using index matrix to get element matrix with same dimension.

        @param index_vector: Vector of size (1 x self.n). Elements are randomly selected indices.
        @param data: Matrix of size (times x 1 x self.n). Each element is "weight" sampled from distribution.
        @returns: (bootstrap_data), element vector corresponding to index_vector from data.
        '''
        # bootstrap_data = index_matrix   # Same size.
        bootstrap_data = np.zeros(np.shape(index_matrix))
        for i in range(len(index_matrix)):
            for j in range(len(index_matrix[0])):
            # Using the index to select element.
                bootstrap_data[i][j] = data[index_matrix[i][j]][i][j]
        return bootstrap_data

    def Update(self, lst, times, bootstrap, coefficient):
        '''
        Update distribution parameters using many samples.

        @param lst: A list of samples we use to update.
        @param times: Number of times we sampled. 
            It's also the number of elements in lst.
        @param bootstrap: Boolean, Whether on or not to bootstrap
        @param coefficient: Sigma = coefficient * mean.
            Note: Usually we use sample variance to update sigma. 
            However, it converges to zero. Therefore we use coefficient.
        @returns: None.
        '''
        self.bootstrap_matrix = lst # Update bootstrap_matrix.

        if bootstrap:
            # Update mu using sample mean.
            self.mu = np.sum(lst, 0) / times
        # When normal, the distribution parameters are mu and sigma.
        elif self.dis == 'gaussian':
            # Update mu using sample mean.
            self.mu = np.sum(lst, 0) / times
            # Update sigma using coefficient * self.mu.
            self.sigma = abs(self.mu) * coefficient
