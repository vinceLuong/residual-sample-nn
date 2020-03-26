# Standard imports.
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
         x: A 2D array. a (N by X) matrix.
        
        Output:
         n: An integer. The number of samples in the input.
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
         z: A 2D array of samples.

        Output:
         y: An array the same shape as z.
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
          y: An array holding the network outputs.
          t: An array holding the corresponding targets.

        Outputs:
          E: The mean CrossEntropy.
    '''
    eps = 1e-5
    y_temp = np.where(y == 1, 1-eps, y)
    y_temp = np.where(y_temp == 0, eps, y_temp)
    E = -np.sum(t*np.log(y_temp) + (1.-t)*np.log(1.-y_temp))
    return E / len(t)

def gradCrossEntropy(y, t):
    '''
        E = gradCrossEntropy(y, t)

        Given targets t, evaluates the gradient of the mean cross entropy loss
        with respect to the output y.

        Inputs:
          y: The array holding the network's output.
          t: An array holding the corresponding targets.

        Outputs:
          dEdy: The gradient of CE with respect to output y.
    '''
    # Initialize parameters.
    dEdy = ( y - t ) / y / (1.-y)
    return dEdy / len(t)

def MSE(y, t):
    '''
        E = MSE(y, t)

        Evaluates the mean squared error loss between outputs y and targets t.

        Inputs:
          y: The array holding the network's output.
          t: An array holding the corresponding targets.

        Outputs:
          E: The mean squared error.
    '''
    # Initialize parameters.
    N = NSamples(y)
    E = 1/2 * np.sum(np.power(y - t, 2))
    # Return the mean squared error loss.
    return E / N

def gradMSE(y, t):
    '''
        E = gradMSE(y, t)

        Given targets t, evaluates the gradient of the mean squared error loss
        with respect to the output y.

        Inputs:
          y: The array holding the network's output.
          t: An array holding the corresponding targets.

        Outputs:
          dEdy: The gradient of MSE with respect to output y.
    '''
    # Initialize parameters.
    N = NSamples(y)
    dEdy = (y - t)
    # Return the gradient of the mean squared error loss.
    return dEdy / N


class Network():

    def FeedForward(self, x, bootstrap):
        '''
            y = net.FeedForward(x)

            Runs the network forward, starting with x as input.
            Returns the activity of the output layer.

            All node use Logistic
            Note: The activation function used for the output layer
            depends on what self.Loss is set to.

            Inputs:
              x: The inputs. a (N by X) matrix.
              bootstrap: Boolean. If true, using bootstrap to sample.
                  Otherwise, sample using distribution parameters.
        '''
        # initialize variables.
        x = np.array(x)  # Convert input to array, in case it's not
        self.lyr[0].h = x
        self.W = [] # Initialize the weights, so that we can append sampled weights.
        # Loop through all layers.
        for i in range(self.n_layers-1):
            # Sample weights and biases
            current_weight = self.weight_matrix[i].Sample(bootstrap)
            self.lyr[i+1].b = self.lyr[i+1].bias_vector.Sample(bootstrap)
            # Stored the sampled results to W
            self.W.append(current_weight)
            # Update next layer's income currents and activities.
            self.lyr[i+1].z = np.matmul(self.lyr[i].h, current_weight) + self.lyr[i+1].b
            self.lyr[i+1].h = self.lyr[i+1].sigma()
        return self.lyr[-1].h

    def Predict(self, x):
        '''
            y = net.FeedForward(x)

            Runs the network forward, starting with x as input.
            Returns the activity of the output layer.

            All node use Logistic
            Note: The activation function used for the output layer
            depends on what self.Loss is set to.

            Inputs:
              x: The inputs. a (N by X) matrix.
              bootstrap: Boolean. If true, using bootstrap to sample.
                  Otherwise, sample using distribution parameters.
        '''
        # initialize variables.
        x = np.array(x)  # Convert input to array, in case it's not
        self.lyr[0].h = x
        self.W = []  # Initialize the weights, so that we can append sampled weights.
        # Loop through all layers.
        for i in range(self.n_layers - 1):
            # Sample weights and biases
            current_weight = self.weight_matrix[i].mu
            self.lyr[i + 1].b = self.lyr[i + 1].bias_vector.mu
            # Stored the sampled results to W
            self.W.append(current_weight)
            # Update next layer's income currents and activities.
            self.lyr[i + 1].z = np.matmul(self.lyr[i].h, current_weight) + self.lyr[i + 1].b
            self.lyr[i + 1].h = self.lyr[i + 1].sigma()
        return self.lyr[-1].h

    def BackProp(self, t, lrate=0.05):
        '''
            net.BackProp(targets, weight, bias, lrate=0.05, times)
            
            Given the current network state and targets t, updates the connection
            weights and biases using the backpropagation algorithm.
            
            Inputs:
              t: An array of targets (number of samples must match the
                  network's output)
             lrate: learning rate
        '''
        t = np.array(t)  # Convert t to an array, in case it's not.
        # Initialize variables.
        ###Code to set threshold
        res_prep = self.lyr[-1].h - t
        eps = 1e-5
        cond = abs(res_prep) < self.th
        y = np.where(cond, 0.5, self.lyr[-1].h)
        y = np.where(y == 1, 1-eps, y)
        y = np.where(y == 0, eps, y)
        targ = np.where(cond, 0.5, t)
        ###
        dEdh = self.gradLoss(y, targ)
        dhdz = self.lyr[-1].sigma_p()
        dEdz = dhdz * dEdh
        for ind in range(self.n_layers-2 ,-1, -1):
            # Use sample mean to update, ignore the variance.
            weights = self.weight_matrix[ind].mu ##################
            dense_dEdb = np.array([sum(x) for x in zip(*dEdz)])
            matrix_dEdW = (dEdz.T @ self.lyr[ind].h).T
            dhdz = self.lyr[ind].sigma_p()
            dEdz = np.multiply(dhdz, np.matmul(dEdz, weights.T))
            self.lyr[ind+1].b = self.lyr[ind+1].bias_vector.mu - lrate * dense_dEdb
            self.W[ind] = self.weight_matrix[ind].mu - lrate * matrix_dEdW
            
    def Learn(self, inputs, targets, lrate=0.05, epochs=1, times = 100, threshold = 0.5, bootstrap = False, progress=True):
        '''
            Network.Learn(inputs, targets, lrate=0.05, epochs=1, times = 100, bootstrap = False, progress=True)

            Run through the dataset 'epochs' number of times, each time we sample the distribution
            weight and bias follows 'times' times to update the distribution parameters.

            Inputs:
              data: A list of 2 arrays, one for inputs, and one for targets.
              lrate: The learning rate (try 0.001 to 0.5).
              epochs: The number of times to go through the training data.
              times: The number of times we sample weights and biases.
              bootstrap: Boolean. If true, using bootstrap to sample.
                  Otherwise, sample using distribution parameters. Default is False.
              progress (Boolean) indicates whether to show cost. Default is True.
              threshold used to create residual for sampling
        '''
        # Initialize a dataframe to store 4D weight and bias results.
        #weight = pd.DataFrame(index = range(times), columns=range(self.n_layers-1))
        #bias = pd.DataFrame(index = range(times), columns=range(self.n_layers-1))
        self.th = threshold
        weight = []
        bias = []
        for k in range((self.n_layers-1)):
            weight.append([0]*(times))
            bias.append([0]*(times))
        # If bootstrap == True, initialize the weights and bias samples.
        if bootstrap:
            for i in range(self.n_layers-1):
                self.weight_matrix[i].Initialize_Bootstrap(times)
                self.lyr[i+1].bias_vector.Initialize_Bootstrap(times)
        # Loop through all epochs.
        for _ in range(epochs):
            # In each epoch, running FeedForward and BackProp "times" times.
            for j in range(times):
                y = self.FeedForward(inputs, bootstrap)
                self.BackProp(targets, lrate)
                # Store the updated weights and biases in the DataFrame.
                for i in range(self.n_layers-1):
                    weight[i][j] = self.W[i]
                    bias[i][j] = self.lyr[i+1].b
            # Then Update each connection weights and bias vector.
            for idx in range(self.n_layers-1):
                self.weight_matrix[idx].Update(weight[idx], times, bootstrap)
                self.lyr[idx+1].bias_vector.Update(bias[idx], times, bootstrap)
            if progress:
                self.cost_history.append(self.Loss(self.lyr[-1].h, targets))

        ###################################################
        # When bootstrap == True, this need to be changed.#
        ###################################################
        if progress:
            pass
            # self.cost_history.append(self.Evaluate(inputs, targets))
    
    
    def __init__(self, sizes, type='classifier', prior_dist_weight=None, prior_dist_bias=None, threshold=0.5):
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
              prior_dist_weight is the prior distribution weights follow,
                  default is None, which is the normal neural network setup.
              prior_dist_weight is the prior distribution biases follow,
                  default is None, which is the normal neural network setup.
        '''
        self.n_layers = len(sizes)
        self.lyr = []    # a list of Layers.
        self.W = []
        self.weight_matrix = [] # Weight matrices, indexed by the layer below it.
        self.cost_history = []  # keeps track of the cost as learning progresses.
        
        # Two common types of networks.
        # The member variable self.Loss refers to one of the implemented.
        # loss functions: MSE, or CrossEntropy.
        # Call it using self.Loss(t).
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

        # Create and add Layers (using logistic for hidden layers).
        for i, e in enumerate(sizes[:-1]):
            self.lyr.append(Layer(e, prior_dist_bias[i]))
        # For the top layer, we use the appropriate activtaion function.
        self.lyr.append(Layer(sizes[-1], prior_dist_bias[-1], act=activation))
        # Initialize the weight matrices.
        for idx in range(self.n_layers-1):
            m = self.lyr[idx].N
            n = self.lyr[idx+1].N
            self.weight_matrix.append(Weight(m,n,prior_dist_weight[idx]))

    def Evaluate(self, inputs, targets):
        '''
            E = net.Evaluate(inputs, targets)

            Computes the average loss over the supplied dataset.

            Inputs:
             inputs: An array of inputs.
             targets: A list of corresponding targets.

            Outputs
             E : A scalar. The average loss.
        '''
        y = self.Predict(inputs)
        return self.Loss(y, targets)

    def ClassificationAccuracy(self, inputs, targets):
        '''
            a = net.ClassificationAccuracy(data)
            
            Returns the fraction (between 0 and 1) of correct one-hot classifications
            in the dataset.

            Inputs:
              inputs: An array of inputs.
              targets: A list of corresponding targets.
        '''
        y = self.Predict(inputs)
        yb = OneHot(y)
        n_incorrect = np.sum(yb!=targets) / 2.
        return 1. - float(n_incorrect) / NSamples(inputs)



if __name__ == "__main__":
    # 5 Classes in 8-Dimensional Space
    np.random.seed(15)
    noise = 0.1
    InputClasses = np.array([[1, 0, 1, 0, 0, 1, 1, 0],
                             [0, 1, 0, 1, 0, 1, 0, 1],
                             [0, 1, 1, 0, 1, 0, 0, 1],
                             [1, 0, 0, 0, 1, 0, 1, 1],
                             [1, 0, 0, 1, 0, 1, 0, 1]], dtype=float)
    OutputClasses = np.array([[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]], dtype=float)
    n_input = np.shape(InputClasses)[1]
    n_output = np.shape(OutputClasses)[1]
    n_classes = np.shape(InputClasses)[0]

    # Create a training dataset
    n_samples = 100
    training_output = []
    training_input = []
    for idx in range(n_samples):
        k = np.random.randint(n_classes)
        x = InputClasses[k, :] + np.random.normal(size=n_input) * noise
        t = OutputClasses[k, :]
        training_input.append(x)
        training_output.append(t)

    # Create a test dataset
    n_samples = 100
    test_output = []
    test_input = []
    for idx in range(n_samples):
        k = np.random.randint(n_classes)
        x = InputClasses[k, :] + np.random.normal(size=n_input) * noise
        t = OutputClasses[k, :]
        test_input.append(x)
        test_output.append(t)

    train = [np.array(training_input), np.array(training_output)]
    test = [np.array(test_input), np.array(test_output)]
    net = Network([n_input, 6, n_output], type='classifier', prior_dist_weight=['gaussian', 'gaussian'],
                  prior_dist_bias=['gaussian', 'gaussian', 'gaussian'])

    # Evaluate it before training
    CE = net.Evaluate(train[0], train[1])
    accuracy = net.ClassificationAccuracy(train[0], train[1])
    print('Cross Entropy = ' + str(CE))
    print('     Accuracy = ' + str(accuracy * 100.) + '%')

    net.Learn(train[0], train[1], epochs=100, lrate=0.1, times=100, threshold=1, bootstrap=False, progress=True)
    layer = 1
    print(net.lyr[layer].bias_vector.mu)
    print(net.lyr[layer].bias_vector.sigma)
    print(net.weight_matrix[layer - 1].mu)
    print(net.weight_matrix[layer - 1].sigma)
