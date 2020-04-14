# Standard imports.
import numpy as np
import pandas as pd
from Class_Layer import *
from Class_Weight import *
from sklearn.metrics import accuracy_score

# Supplied functions
def NSamples(x):
    '''
    Calculates the number of samples in a batch of inputs.

    @param x: A 2D array of samples. Size of (number of samples, number of variables).
    @returns: (n), int, the number of samples in the input.
    '''
    return len(x)

def OneHot(z):
    '''
    Applies the one-hot function to the vectors in z.
    Example: OneHot([[0.9, 0.1], [-0.5, 0.1]]) returns np.array([[1,0],[0,1]])

    @param z: A 2D array of samples.
    @returns: (y), an array the same shape as z.
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
    Evaluates the mean cross entropy loss between outputs y and targets t.

    @param y: An array holding the network outputs.
    @param t: An array holding the corresponding targets.
    @returns: (E), the mean CrossEntropy.
    '''
    E = -np.sum(t*np.log(y) + (1.-t)*np.log(1.-y))
    return E / len(t)

def gradCrossEntropy(y, t):
    '''
    Given targets t, evaluates the gradient of the mean cross entropy loss
    with respect to the output y.

    @param y: The array holding the network's output.
    @parm t: An array holding the corresponding targets.
    @returns: (dEdy), the gradient of CE with respect to output y.
    '''
    # Initialize parameters.
    dEdy = ( y - t ) / y / (1.-y)
    return dEdy / len(t)

def MSE(y, t):
    '''
    Evaluates the mean squared error loss between outputs y and targets t.

    @param y: The array holding the network's output.
    @param t: An array holding the corresponding targets.
    @returns: (E), the mean squared error.
    '''
    # Initialize parameters.
    N = NSamples(y)
    E = 1/2 * np.sum(np.power(y - t, 2))
    # Return the mean squared error loss.
    return E / N

def gradMSE(y, t):
    '''
    Given targets t, evaluates the gradient of the mean squared error loss
    with respect to the output y.

    @param y: The array holding the network's output.
    @param t: An array holding the corresponding targets.
    @returns: (dEdy), the gradient of MSE with respect to output y.
    '''
    # Initialize parameters.
    N = NSamples(y)
    dEdy = (y - t)
    # Return the gradient of the mean squared error loss.
    return dEdy / N

def CategoricalCE(y, t):
    '''
    Given targets t, evaluates the gradient of the 
    categorical cross entropy loss with respect to the output y.

    @param y: The array holding the network's output.
    @param t: An array holding the corresponding targets.
    @returns: (dEdy), the gradient of MSE with respect to output y.
    '''
    N = NSamples(y)
    dEdy = -np.sum(t * np.log(y))
    return dEdy / N

def Shuffle(inputs, targets):
    '''
    Randomly shuffles the dataset.

    @param inputs: An array of inputs.
    @param targets: An array of corresponding targets.

    @returns: (s_inputs,s_targets), the shuffled array of inputs and corresponding targets.
    '''
    data = list(zip(inputs,targets))
    np.random.shuffle(data)
    s_inputs, s_targets = zip(*data)
    return np.array(s_inputs), np.array(s_targets)

def MakeBatches(data_in, data_out, batch_size=10, shuffle=True):
    '''
    Breaks up the dataset into batches of size batch_size.

    @param data_in： A list of inputs.
    @param data_out: A list of outputs.
    @param batch_size: The number of samples in each batch. Default is 10.
    @param shuffle: Boolean. If true, then shuffle samples. Default is True.
    @returns: (batches), a list containing batches, where each batch is: [in_batch, out_batch].

    Note: The last batch might be incomplete (smaller than batch_size).
    '''
    N = len(data_in) # Number of samples.
    r = range(N) # Indexes.
    if shuffle:
        r = np.random.permutation(N)
    batches = []
    for k in range(0, N, batch_size):
        if k + batch_size <= N:
            din = data_in[r[k:k+batch_size]]
            dout = data_out[r[k:k+batch_size]]
        # Possible Last Batch case.
        else:
            din = data_in[r[k:]]
            dout = data_out[r[k:]]
        if isinstance(din, (list, tuple)):
            batches.append([np.stack(din, axis=0),np.stack(dout, axis=0)])
        else:
            batches.append([din , dout])
    return batches

class Network():

    def FeedForward(self, x):
        '''
        Runs the network forward, starting with x as input.
        Returns the activity of the output layer.

        All inner layer use Logistic sigma.
        Note: The activation function used for the output layer
        depends on what self.Loss is set to.

        @param x: The inputs. Size of (number of samples, number of variables).
        @returns: (self.lyr[-1].h), the predicted target.
        '''
        # initialize variables.
        x = np.array(x)  # Convert input to array, in case it's not
        self.lyr[0].h = x
        self.W = [] # Initialize the weights, so that we can append sampled weights.
        # Loop through all layers.
        for i in range(self.n_layers-1):
            # Sample weights and biases
            current_weight = self.weight_matrix[i].Sample(self.bootstrap)
            self.lyr[i+1].b = self.lyr[i+1].bias_vector.Sample(self.bootstrap)
            # Stored the sampled results to W
            self.W.append(current_weight)
            # Update next layer's income currents and activities.
            self.lyr[i+1].z = np.matmul(self.lyr[i].h, current_weight) + self.lyr[i+1].b
            self.lyr[i+1].h = self.lyr[i+1].sigma()
        return self.lyr[-1].h

    def Predict(self, x):
        '''
        Runs the network forward, starting with x as input.
        Using distribution mean for weights and biases.
        Returns the activity of the output layer.

        All node use Logistic
        Note: The activation function used for the output layer
        depends on what self.Loss is set to.

        @param x: The inputs. Size of (number of samples, number of variables).
        @returns: (self.lyr[-1].h), the predicted target.
        '''
        # Initialize variables.
        x = np.array(x)  # Convert input to array, in case it's not
        self.lyr[0].h = x
        self.W = []  # Initialize the weights, so that we can append sampled weights.
        # Loop through all layers.
        for i in range(self.n_layers - 1):
            # Set weights and biases using distribution mean.
            current_weight = self.weight_matrix[i].mu
            self.lyr[i + 1].b = self.lyr[i + 1].bias_vector.mu
            # Stored the sampled results to W
            self.W.append(current_weight)
            # Update next layer's income currents and activities.
            self.lyr[i + 1].z = np.matmul(self.lyr[i].h, current_weight) + self.lyr[i + 1].b
            self.lyr[i + 1].h = self.lyr[i + 1].sigma()
        return self.lyr[-1].h

    def TopGradient(self, y, t):
        '''
        Computes and returns the gradient of the cost with respect to the input current
        to the output nodes.

        @param targets: A batch of targets corresponding to the last FeedForward run.
        @returns: (dEdz), a batch of gradient vectors corresponding to the output nodes.
        '''
        if self.type=='classifier':
            return ( y - t ) / len(t)
        elif self.type=='regression':
            return ( y - t ) / len(t)
        elif self.type=='bernoulli':
            return ( y - t ) / len(t)
        return self.gradLoss(self.lyr[-1].h, t) * self.lyr[-1].sigma_p() / len(t)

    def BackProp(self, t, lrate=1):
        '''
        Given the current network state and targets t, updates the connection
        weights and biases using the backpropagation algorithm.
            
        @param t: An array of targets (number of samples must match the network's output).
        @param lrate: The learning rate.
        @returns: None.
        '''
        t = np.array(t)  # Convert t to an array, in case it's not.
        # Set threshold: if within threshold, no changes
        # Otherwise, set the value to 0.5.
        res_prep = self.lyr[-1].h - t
        cond = abs(res_prep) < self.th
        y = np.where(cond, 0.5, self.lyr[-1].h)
        targ = np.where(cond, 0.5, t)
        # Initialize top gradient.
        dEdz = self.TopGradient(y, targ)
        for ind in range(self.n_layers-2 ,-1, -1):
            # Use backpropogation to update weights/biases.
            weights = self.W[ind]
            dense_dEdb = np.array([sum(x) for x in zip(*dEdz)])
            matrix_dEdW = (dEdz.T @ self.lyr[ind].h).T
            dhdz = self.lyr[ind].sigma_p()
            dEdz = np.multiply(dhdz, np.matmul(dEdz, weights.T))
            # Store the updated weights/biases in lyr.b and W.
            self.lyr[ind+1].b -= lrate * dense_dEdb
            self.W[ind] -= lrate * matrix_dEdW
            
    def Learn(self, inputs, targets, lrate=1, epochs=1, times = 100, threshold = 0, coefficient = 0.05, bootstrap = False):
        '''
        Run through the dataset 'epochs' number of times, each time we sample the distribution
        weight and bias follows 'times' times to update the distribution parameters.

        @param data: A list of 2 arrays, one for inputs, and one for targets.
        @param lrate: The learning rate (try 0.001 to 0.5).
        @param epochs: The number of times to go through the training data.
        @param times: The number of times we sample weights and biases. Default is 100.
        @param threshold: If y - t < threshold, we will ignore the change. 
            It's Used to create residual for sampling.
        @param coefficient: Sigma = coefficient * mean. Default is 0.05.
            Note: Usually we use sample variance to update sigma. 
            However, it converges to zero. Therefore we use coefficient.
        @param bootstrap: Boolean. If true, using bootstrap to sample.
            Otherwise, sample using distribution parameters. Default is False.
        @returns: (progress), an (epochs)x1 array with cost in the column.
        '''
        # Setting threshold, later used in self.backprop.
        self.th = threshold
        # Setting the boolean variable bootstrap.
        self.bootstrap = bootstrap
        # Initialize a matrices to store 4D weight and bias results.
        weight = []
        bias = []
        for _ in range((self.n_layers-1)):
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
                _ = self.FeedForward(inputs)
                self.BackProp(targets, lrate)
                # Store the updated weights and biases in the Matrices.
                for i in range(self.n_layers-1):
                    weight[i][j] = self.W[i]
                    bias[i][j] = self.lyr[i+1].b        
            self.weight = weight 
            self.bias = bias
            # Then Update each connection weights and bias vector.
            for idx in range(self.n_layers-1):
                self.weight_matrix[idx].Update(weight[idx], times, bootstrap, coefficient)
                self.lyr[idx+1].bias_vector.Update(bias[idx], times, bootstrap, coefficient)
            self.cost_history.append(self.Evaluate(inputs, targets))
        return np.array(self.cost_history)

    def MBGD(self, inputs, targets, lrate=0.05, epochs=1, batch_size=10, times = 100, threshold = 0, coefficient = 0.05, bootstrap = False):
        '''
        Performs Mini-Batch Gradient Descent on the network.
        Run through the dataset in batches 'epochs' number of times, incrementing the
        network weights after each batch. For each epoch, it shuffles the dataset.

        @param inputs: An array of input samples.
        @param targets: A corresponding array of targets.
        @param lrate: The learning rate (try 0.001 to 5). Default is 0.05.
        @param epochs: The number of times to go through the training data. Default is 1.
        @param batch_size: The number of samples for each batch. Default is 10.
        @param Threshold: If y - t < threshold, we will ignore the change. 
            It's Used to create residual for sampling.
        @param coefficient: Sigma = coefficient * mean. Default is 0.05.
            Note: Usually we use sample variance to update sigma. 
            However, it converges to zero. Therefore we use coefficient.
        @param bootstrap: Boolean. If true, using bootstrap to sample.
            Otherwise, sample using distribution parameters. Default is False.
        @returns: (progress), an (epochs)x1 array with cost in the column.
        '''
        # Initialize matrices to store 4D weight and bias results.
        weight = []
        bias = []
        for _ in range((self.n_layers-1)):
            weight.append([0]*(times))
            bias.append([0]*(times))
        # Setting threshold, later used in backprop.
        self.th = threshold
        # Setting the boolean variable bootstrap.
        self.bootstrap = bootstrap
        # If bootstrap == True, initialize the weights and bias samples.
        if bootstrap:
            for i in range(self.n_layers-1):
                self.weight_matrix[i].Initialize_Bootstrap(times)
                self.lyr[i+1].bias_vector.Initialize_Bootstrap(times)
        # For each epoch.
        for _ in range(epochs):
            # Make the batchs.
            batches = MakeBatches(inputs, targets, batch_size=batch_size, shuffle=True)
            # For each batch, run Feedforward/Backprop "times" times
            for mini_batch in batches:
                for j in range(times):
                    _ = self.FeedForward(mini_batch[0])
                    self.BackProp(mini_batch[1], lrate=lrate)
                    # Store the updated weights and biases in the Matrices.
                    for i in range(self.n_layers-1):
                        weight[i][j] = self.W[i]
                        bias[i][j] = self.lyr[i+1].b
                # Then Update each connection weights and bias vector.
                self.weight = weight 
                self.bias = bias
                for idx in range(self.n_layers-1):
                    self.weight_matrix[idx].Update(weight[idx], times, bootstrap, coefficient)
                    self.lyr[idx+1].bias_vector.Update(bias[idx], times, bootstrap, coefficient)
            self.cost_history.append(self.Evaluate(inputs, targets))
        return np.array(self.cost_history)
    
    def __init__(self, sizes, type='classifier', pdw=None, pdb=None):
        '''
        Creates a Network and saves it in the variable 'net'.

        @param sizes: A list of integers specifying the number
            of nodes in each layer.
            eg. [5, 20, 3] will create a 3-layer network
            with 5 input, 20 hidden, and 3 output nodes
        @param type: Can be either 'bernoulli', 'classifier' or 'regression',
            and sets the activation function on the output layer, as well as the loss function.
            'bernoulli':  Logistic, cross entropy.
            'classifier': Softmax, categorical cross entropy.
            'regression': Linear, mean squared error.
        @param pdw: The prior distribution weights follow. Default is 'gaussian'.
        @param pdb: The prior distribution biases follow. Default is 'gaussian'.
        @returns: None.
        '''
        self.n_layers = len(sizes)
        self.lyr = []    # a list of Layers.
        self.W = []
        self.weight_matrix = [] # Weight matrices, indexed by the layer below it.
        self.cost_history = []  # keeps track of the cost as learning progresses.
        self.type = type  # 'bernoulli', 'classifier', 'regression'
        # Two common types of networks.
        # The member variable self.Loss refers to one of the implemented.
        # loss functions: MSE, or CrossEntropy.
        # Call it using self.Loss(t).
        if type == 'bernoulli':
            self.classifier = True
            self.Loss = CrossEntropy
            self.gradLoss = gradCrossEntropy
            activation = 'logistic'
        elif type == 'classifier':
            self.classifier = True
            self.Loss = CategoricalCE
            self.gradLoss = None
            activation = 'softmax'
        elif type == 'regression':
            self.classifier = False
            self.Loss = MSE
            self.gradLoss = gradMSE
            activation = 'identity'
        else:
            print('Error, no top gradient available')

        # Create and add Layers (using logistic for hidden layers).
        for i, e in enumerate(sizes[:-1]):
            self.lyr.append(Layer(e, pdb[i]))
        # For the top layer, we use the appropriate activtaion function.
        self.lyr.append(Layer(sizes[-1], pdb[-1], act=activation))
        # Initialize the weight matrices.
        for idx in range(self.n_layers-1):
            m = self.lyr[idx].N
            n = self.lyr[idx+1].N
            self.weight_matrix.append(Weight(m,n,pdw[idx]))

    def Evaluate(self, inputs, targets, times = 1):
        '''
        Computes the average loss over the supplied dataset.

        Note: Depends on the value of times, we use distribution mean/sampled values
        for weights and biases when evaluating the data.

        @param inputs: An array of inputs.
        @param targets: A list of corresponding targets.
        @param times: Int, the number of samples we evaluate.
        @returns: (E), scalar. The average loss.
        '''
        # If only evaluate once, use the sample mean for weights/biases.
        if times == 1:
            y = self.Predict(inputs)
        # Otherwise, weights/biases are sampled from the distribution.
        else:
            y = np.zeros(np.shape(targets))
            for _ in range(times):
                y += self.FeedForward(inputs)
        return self.Loss(y/times, targets)

    def ClassificationAccuracy(self, inputs, targets, times = 1):
        '''
        Returns the fraction (between 0 and 1) of correct one-hot classifications
        in the dataset.

        @param inputs: An array of inputs.
        @param targets: A list of corresponding targets.
        @returns: (accuracy), the percentage of correctly classified samples.
        '''
        # Handle exceptions.
        if self.type == 'regression':
            return('This function is only used for network of type \'bernoulli\' or \'classifer\'')
        # Different FeedForward depends on "time" value.
        # If times == 1, we use distribution mean for weights and biases.
        if times == 1:
            y = self.Predict(inputs)
        # Else, sample from the distribution.
        else:
            y = np.zeros(np.shape(targets))
            for _ in range(times):
                y += self.FeedForward(inputs)
        # If type = classifier, we use OneHot encoding.
        if self.type == 'classifier':
            # No need to take average since the result 
            # for OneHot encoding Will be the same.
            yb = OneHot(y)
            n_incorrect = np.sum(yb!=targets) / 2.
            accuracy = 1. - float(n_incorrect) / NSamples(inputs)
        # Otherwise type = bernoulli, we set the value to one if > 0.5.
        elif self.type == 'bernoulli':
            # Take the average for y.
            y = y / times
            yb = np.where(y < 0.5, 0, 1)
            accuracy = accuracy_score(yb, targets)
        return accuracy
