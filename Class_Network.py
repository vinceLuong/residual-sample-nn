# Standard imports
import numpy as np
import pandas as pd
from Class_Layer import *
from Class_Weight import *

# Supplied functions

def NSamples(x):
    '''
        n = NSamples(x)
        
        Returns the number of samples in a batch of inputs.
        
        Input:
         x   is a 2D array
        
        Output:
         n   is an integer
    '''
    return len(x)

def OneHot(z):
    '''
        y = OneHot(z)

        Applies the one-hot function to the vectors in z.
        Example:
          OneHot([[0.9, 0.1], [-0.5, 0.1]])
          returns np.array([[1,0],[0,1]])

        Input:
         z    is a 2D array of samples

        Output:
         y    is an array the same shape as z
    '''
    y = []
    # Locate the max of each row
    for zz in z:
        idx = np.argmax(zz)
        b = np.zeros_like(zz)
        b[idx] = 1.
        y.append(b)
    y = np.array(y)
    return y

def CrossEntropy(y, t):
    '''
        E = CrossEntropy(y, t)

        Evaluates the mean cross entropy loss between outputs y and targets t.

        Inputs:
          y is an array holding the network outputs
          t is an array holding the corresponding targets

        Outputs:
          E is the mean CE
    '''
    E = -np.sum(t*np.log(y) + (1.-t)*np.log(1.-y))
    return E / len(t)

def gradCrossEntropy(y, t):
    '''
        E = gradCrossEntropy(y, t)

        Given targets t, evaluates the gradient of the mean cross entropy loss
        with respect to the output y.

        Inputs:
          y is the array holding the network's output
          t is an array holding the corresponding targets

        Outputs:
          dEdy is the gradient of CE with respect to output y
    '''
    # Initialize parameters.
    dEdy = ( y - t ) / y / (1.-y)
    return dEdy / len(t)

def MSE(y, t):
    '''
        E = MSE(y, t)

        Evaluates the mean squared error loss between outputs y and targets t.

        Inputs:
          y is the array holding the network's output
          t is an array holding the corresponding targets

        Outputs:
          E is the MSE
    '''
    # Initialize parameters.
    N = NSamples(y)
    E = 1/2 * np.sum(np.power(y - t, 2))
    # Return the mean squared error loss
    return E / N

def gradMSE(y, t):
    '''
        E = gradMSE(y, t)

        Given targets t, evaluates the gradient of the mean squared error loss
        with respect to the output y.

        Inputs:
          y is the array holding the network's output
          t is an array holding the corresponding targets

        Outputs:
          dEdy is the gradient of MSE with respect to output y
    '''
    # Initialize parameters.
    N = NSamples(y)
    dEdy = (y - t)
    # Return the gradient of the mean squared error loss
    return dEdy / N


class Network():

    
    def FeedForward(self, x):
        '''
            y = net.FeedForward(x)

            Runs the network forward, starting with x as input.
            Returns the activity of the output layer.

            All node use Logistic
            Note: The activation function used for the output layer
            depends on what self.Loss is set to.
        '''
        # initialize variables.
        x = np.array(x)  # Convert input to array, in case it's not
        self.lyr[0].h = x
        self.W = []
        # Loop through all layers.
        for i in range(self.n_layers-1):
            # Construct Weight and Bias matrix
            current_weight = self.weight_matrix[i].Sample()
            self.W.append(current_weight)
            self.lyr[i+1].b = self.lyr[i+1].bias_vector.Sample()
            # Update next layer's income currents and activities.
            self.lyr[i+1].z = np.matmul(self.lyr[i].h, current_weight) + self.lyr[i+1].b
            self.lyr[i+1].h = self.lyr[i+1].sigma()
        return self.lyr[-1].h

    
    def BackProp(self, t, lrate=0.05):
        '''
            net.BackProp(targets, weight, bias, lrate=0.05, times)
            
            Given the current network state and targets t, updates the connection
            weights and biases using the backpropagation algorithm.
            
            Inputs:
             t      an array of targets (number of samples must match the
                    network's output)
             lrate  learning rate
        '''
        t = np.array(t)  # convert t to an array, in case it's not
        # Initialize variables.
        dEdh = self.gradLoss(self.lyr[-1].h, t)
        dhdz = self.lyr[-1].sigma_p()
        dEdz = dhdz * dEdh
        for ind in range(self.n_layers-2 ,-1, -1):
            weights = self.W[ind]
            dense_dEdb = np.array([sum(x) for x in zip(*dEdz)])

            matrix_dEdW = (dEdz.T @ self.lyr[ind].h).T
            dhdz = self.lyr[ind].sigma_p()
            dEdz = np.multiply(dhdz, np.matmul(dEdz, weights.T))
            self.lyr[ind+1].b -= lrate * dense_dEdb 
            self.W[ind] -= lrate * matrix_dEdW
            
    def Learn(self, inputs, targets, lrate=0.05, epochs=1, times = 100, progress=True):
        '''
            Network.Learn(data, lrate=0.05, epochs=1, progress=True)

            Run through the dataset 'epochs' number of times, incrementing the
            network weights after each epoch. For each epoch, it
            shuffles the order of the samples.

            Inputs:
              data is a list of 2 arrays, one for inputs, and one for targets
              lrate is the learning rate (try 0.001 to 0.5)
              epochs is the number of times to go through the training data
              progress (Boolean) indicates whether to show cost
        '''
        weight = pd.DataFrame(index = range(times),columns=range(self.n_layers-1))
        bias = pd.DataFrame(index = range(times),columns=range(self.n_layers-1))
        for _ in range(epochs):
            for j in range(times):
                y = self.FeedForward(inputs)
                self.BackProp(targets, lrate)
                for i in range(self.n_layers-1):
                    weight[i][j] = self.W[i]
                    bias[i][j] = self.lyr[i+1].b
            for idx in range(self.n_layers-1):
                self.weight_matrix[idx].Update(weight[idx].tolist(), times)
                self.lyr[idx+1].bias_vector.Update(bias[idx].tolist(), times)
            if progress:
                self.cost_history.append(self.Loss(self.lyr[-1].h, targets))
            
        if progress:
            self.cost_history.append(self.Evaluate(inputs, targets))
    
    
    def __init__(self, sizes, type='classifier', prior_dist_weight=None, prior_dist_bias=None):
        '''
            net = Network(sizes, type='classifier')

            Creates a Network and saves it in the variable 'net'.

            Inputs:
              sizes is a list of integers specifying the number
                  of nodes in each layer
                  eg. [5, 20, 3] will create a 3-layer network
                      with 5 input, 20 hidden, and 3 output nodes
              type can be either 'classifier' or 'regression', and
                  sets the activation function on the output layer,
                  as well as the loss function.
                  'classifier': logistic, cross entropy
                  'regression': linear, mean squared error
              prior_dist is the prior distribution weights follow,
                  default is None, which is the normal
                  neural network setup.
        '''
        self.n_layers = len(sizes)
        self.lyr = []    # a list of Layers
        self.W = []
        self.weight_matrix = []      # Weight matrices, indexed by the layer below it
        self.cost_history = []  # keeps track of the cost as learning progresses
        
        # Two common types of networks
        # The member variable self.Loss refers to one of the implemented
        # loss functions: MSE, or CrossEntropy.
        # Call it using self.Loss(t)
        if type=='classifier':
            self.classifier = True
            self.Loss = CrossEntropy
            self.gradLoss = gradCrossEntropy
            activation = 'logistic'
        else:
            self.classifier = False
            self.Loss = MSE
            self.gradLoss = gradMSE
            activation = 'identity'

        # Create and add Layers (using logistic for hidden layers)
        for i, e in enumerate(sizes[:-1]):
            self.lyr.append( Layer(e, prior_dist_bias[i]))
   
        # For the top layer, we use the appropriate activtaion function
        self.lyr.append( Layer(sizes[-1], prior_dist_bias[-1], act=activation))
    
        # Randomly initialize weight matrices
        for idx in range(self.n_layers-1):
            m = self.lyr[idx].N
            n = self.lyr[idx+1].N
            self.weight_matrix.append(Weight(m,n,prior_dist_weight[idx]))

    def Evaluate(self, inputs, targets):
        '''
            E = net.Evaluate(data)

            Computes the average loss over the supplied dataset.

            Inputs
             inputs  is an array of inputs
             targets is a list of corresponding targets

            Outputs
             E is a scalar, the average loss
        '''
        y = self.FeedForward(inputs)
        return self.Loss(y, targets)

    def ClassificationAccuracy(self, inputs, targets):
        '''
            a = net.ClassificationAccuracy(data)
            
            Returns the fraction (between 0 and 1) of correct one-hot classifications
            in the dataset.
        '''
        y = self.FeedForward(inputs)
        yb = OneHot(y)
        n_incorrect = np.sum(yb!=targets) / 2.
        return 1. - float(n_incorrect) / NSamples(inputs)