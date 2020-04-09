# Network.py
# (C) Jeff Orchard, 2019

import numpy as np
from copy import deepcopy


#============================================================
#
# Untility functions
#
#============================================================
# Supplied functions

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

def Shuffle(inputs, targets):
    '''
        s_inputs, s_targets = Shuffle(inputs, targets)

        Randomly shuffles the dataset.

        Inputs:
         inputs     array of inputs
         targets    array of corresponding targets

        Outputs:
         s_inputs   shuffled array of inputs
         s_targets  corresponding shuffled array of targets
    '''
    data = list(zip(inputs,targets))
    np.random.shuffle(data)
    s_inputs, s_targets = zip(*data)
    return np.array(s_inputs), np.array(s_targets)

def MakeBatches(data_in, data_out, batch_size=10, shuffle=True):
    '''
    batches = MakeBatches(data_in, data_out, batch_size=10)

    Breaks up the dataset into batches of size batch_size.

    Inputs:
      data_in    is a list of inputs
      data_out   is a list of outputs
      batch_size is the number of samples in each batch
      shuffle    shuffle samples first (True)

    Output:
      batches is a list containing batches, where each batch is:
                 [in_batch, out_batch]

    Note: The last batch might be incomplete (smaller than batch_size).
    '''
    N = len(data_in)
    r = range(N)
    if shuffle:
        r = np.random.permutation(N)
    batches = []
    for k in range(0, N, batch_size):
        if k+batch_size<=N:
            din = data_in[r[k:k+batch_size]]
            dout = data_out[r[k:k+batch_size]]
        else:
            din = data_in[r[k:]]
            dout = data_out[r[k:]]
        if isinstance(din, (list, tuple)):
            batches.append( [np.stack(din, dim=0) , np.stack(dout, dim=0)] )
        else:
            batches.append( [din , dout] )

    return batches


# Cost Functions--------------------------
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
    #====== REMOVE ABOVE IF YOU DON'T PLAN TO USE THE SOLUTIONS ======
    # [1] Cross entropy formula
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
    # [1] Compute the gradient of CE w.r.t. output
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
    # [1] MSE formula
    E = np.sum((y-t)**2)/2./len(t)
    return E

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
    # [1] Compute the gradient of MSE w.r.t. output
    return ( y - t ) / len(t)

def CategoricalCE(outputs, t):
    return -np.sum(t * np.log(outputs)) / len(t)




#==================================================
#
# Layer Class
#
#==================================================
class Layer():

    def __init__(self, n_nodes=0, act='logistic'):
        '''
            lyr = Layer(n_nodes, act='logistic')

            Creates a layer object.

            Inputs:
             n_nodes  the number of nodes in the layer
             act      specifies the activation function
                      Use 'logistic' or 'identity'
        '''
        self.N = n_nodes  # number of nodes in this layer
        self.h = []       # node activities
        self.z = []
        self.b = np.zeros(self.N)  # biases
        self.SetActivationFunction(act)


    def SetActivationFunction(self, act):
        if act=='identity':
            self.act_text = 'identity'
            self.sigma = self.Identity
            self.sigma_p = self.Identity_p
        elif act=='softmax':
            self.act_text = 'softmax'
            self.sigma = self.Softmax
            self.sigma_p = None
        elif act=='logistic':
            self.act_text = 'logistic'
            self.sigma = self.Logistic
            self.sigma_p = self.Logistic_p
        else:
            print('Error: Activation function '+act+' not implemented!')
            self.act_text = ''


    def Save(self, fp):
        np.save(fp, self.N)
        np.save(fp, self.act_text)
        np.save(fp, self.b)

    def Load(self, fp):
        self.N = np.asscalar( np.load(fp) )
        act_text = str( np.load(fp) )
        self.b = np.array( np.load(fp) )
        self.SetActivationFunction(act_text)


    def Logistic(self):
        return 1. / (1. + np.exp(-self.z))
    def Logistic_p(self):
        return self.h * (1.-self.h)
    def Identity(self):
        return self.z
    def Identity_p(self):
        return np.ones_like(self.h)
    def Softmax(self):
        v = np.exp(self.z)
        s = np.sum(v, axis=1)
        return v/np.tile(s[:,np.newaxis], [1,np.shape(v)[1]])



#==================================================
#
# Network Class
#
#==================================================
class Network():

    def __init__(self, sizes, type='classifier'):
        '''
            net = Network(sizes, type='classifier')

            Creates a Network and saves it in the variable 'net'.

            Inputs:
              sizes is a list of integers specifying the number
                  of nodes in each layer
                  eg. [5, 20, 3] will create a 3-layer network
                      with 5 input, 20 hidden, and 3 output nodes
              type can be either 'Bernoulli', 'classifier' or 'regression',
                   and sets the activation function on the output layer,
                   as well as the loss function.
                   'Bernoulli':  logistic, cross entropy
                   'classifier': softmax, categorical cross entropy
                   'regression': linear, mean squared error
        '''
        self.n_layers = 0 #len(sizes)
        self.lyr = []    # a list of Layers
        self.W = []      # Weight matrices, indexed by the layer below it

        self.type = type # 'Bernoulli', 'classifier', 'regression'
        self.output_activation = None
        self.SetCostFunction()

        self.cost_history = []  # keeps track of the cost as learning progresses


        # Create and add Layers (using logistic for hidden layers)
        for n in sizes[:-1]:
            self.AddLayer( Layer(n) )
        # For the top layer, we use the appropriate activtaion function
        self.AddLayer( Layer(sizes[-1], act=self.output_activation) )


    def AddLayer(self, layer):
        '''
            net.AddLayer(layer)

            Adds the layer object to the network and connects it to the preceding layer.

            Inputs:
              layer is a layer object
        '''
        self.lyr.append(layer)
        self.n_layers += 1
        # If this isn't our first layer, add connection weights
        if self.n_layers>=2:
            m = self.lyr[-1].N
            n = self.lyr[-2].N
            temp = np.random.normal(size=[n,m])/np.sqrt(n)
            self.W.append(temp)


    def SetCostFunction(self):
        if self.type=='Bernoulli':
            self.Loss = CrossEntropy
            self.gradLoss = gradCrossEntropy
            self.output_activation = 'logistic'
        elif self.type=='classifier':
            self.Loss = CategoricalCE
            self.gradLoss = None
            self.output_activation = 'softmax'
        elif self.type=='regression':
            self.Loss = MSE
            self.gradLoss = gradMSE
            self.output_activation = 'identity'
        else:
            self.Loss = None
            self.gradLoss = None
            self.output_activation = 'logistic'
            print('Error: Network type '+self.type+' not implemented!')

    def Save(self, fname):
        '''
            net.Save(fname)

            Saves the Network object to a file.

            Input:
              fname is a string filename. Should probably use the extension ".npy".
        '''
        fp = open(fname, 'wb')
        np.save(fp, self.n_layers)
        np.save(fp, self.type)
        np.save(fp, self.output_activation)
        for l in self.lyr:
            l.Save(fp)
        for w in self.W:
            np.save(fp, w)
        fp.close()

    @classmethod
    def Load(cls, fname):
        '''
            net.Load(fname)

            Load a Network object from a file. The object needs to be created already,
            but Load will alter it. For example,

               >> net = Network.Network()
               >> net.Load('savednet.npy')

            Input:
              fname is a string filename
        '''
        fp = open(fname, 'rb')
        n_layers = np.asscalar( np.load(fp) ) # self.n_layers is incremented as we call AddLayer
        net = cls([1,1])
        net.lyr = []
        net.n_layers = 0
        net.type = str( np.load(fp) )
        net.SetCostFunction()
        net.output_activation = str( np.load(fp) )
        # Load layers, one at a time
        for k in range(n_layers):
            l = Layer()
            l.Load(fp)
            net.AddLayer(l)
        # Load weight matrices, one at a time
        net.W = []
        for k in range(n_layers-1):
            w = np.array( np.load(fp) )
            net.W.append(w)
        fp.close()
        return net


    def FeedForwardFrom(self, idx, h):
        '''
           y = net.FeedForwardFrom(idx, h)

           Sets the state of layer idx to h, and then performs a FeedForward
           pass from that layer to the output layer.

           Inputs:
             idx     index of layer to set
             h       array holding a batch of hidden states

           Output:
             y       array of outputs corresponding to the hidden states
        '''
        self.lyr[idx].h = h[:]
        for pre,post,W in zip(self.lyr[idx:-1], self.lyr[idx+1:], self.W[idx:]):
            # [1] Calc. (and record) input current to next layer
            post.z = pre.h @ W + post.b

            # [1] Use activation function to get activities
            post.h = post.sigma()

        # Return activity of output layer
        return self.lyr[-1].h


    def FeedForward(self, x):
        '''
            y = net.FeedForward(x)

            Runs the network forward, starting with x as input.
            Returns the activity of the output layer.

            All node use
            Note: The activation function used for the output layer
            depends on what self.Loss is set to.
        '''
        x = np.array(x)  # Convert input to array, in case it's not

        self.lyr[0].h = x # [1] Set input layer

        # Loop over connections...
        for pre,post,W in zip(self.lyr[:-1], self.lyr[1:], self.W):

            # [1] Calc. (and record) input current to next layer
            post.z = pre.h @ W + post.b

            # [1] Use activation function to get activities
            post.h = post.sigma()

        # Return activity of output layer
        return self.lyr[-1].h

    def TopGradient(self, t):
        '''
            dEdz = net.TopGradient(targets)

            Computes and returns the gradient of the cost with respect to the input current
            to the output nodes.

            Inputs:
              targets is a batch of targets corresponding to the last FeedForward run

            Outputs:
              dEdz is a batch of gradient vectors corresponding to the output nodes
        '''
        if self.type=='classifier':
            return ( self.lyr[-1].h - t ) / len(t)
        elif self.type=='regression':
            return ( self.lyr[-1].h - t ) / len(t)
        elif self.type=='Bernoulli':
            return ( self.lyr[-1].h - t ) / len(t)
        return self.gradLoss(self.lyr[-1].h, t) * self.lyr[-1].sigma_p() / len(t)

    def BackProp(self, t, lrate=0.05):
        '''
            net.BackProp(targets, lrate=0.05)

            Given the current network state and targets t, updates the connection
            weights and biases using the backpropagation algorithm.

            Inputs:
             t      an array of targets (number of samples must match the
                    network's output)
             lrate  learning rate
        '''
        t = np.array(t)  # convert t to an array, in case it's not

        # Error gradient for top layer
        dEdz = self.TopGradient(t)

        # Loop down through the layers
        # Start second-from-the-top, and go down to layer 0
        for i in range(self.n_layers-2, -1, -1):
            pre = self.lyr[i]

            # Gradient w.r.t. weights
            dEdW = pre.h.T @ dEdz

            # Gradient w.r.t. biases
            dEdb = np.sum(dEdz, axis=0)

            # Use Sigma'
            # Project error gradient down to layer below
            dEdz = ( dEdz @ self.W[i].T ) * pre.sigma_p()

            # Update weights and biases
            self.W[i] -= lrate*dEdW
            self.lyr[i+1].b -= lrate*dEdb

    def Learn(self, inputs, targets, lrate=0.05, epochs=1, progress=True):
        '''
            Network.Learn(data, lrate=0.05, epochs=1, progress=True)

            Run through the dataset 'epochs' number of times, incrementing the
            network weights after each epoch.

            Inputs:
              data is a list of 2 arrays, one for inputs, and one for targets
              lrate is the learning rate (try 0.001 to 0.5)
              epochs is the number of times to go through the training data
              progress (Boolean) indicates whether to show cost
        '''
        try: Learn
        except NameError:

            #========= YOUR IMPLEMENTATION BELOW =========

            # [1] Perform multiple epochs
            for k in range(epochs):

                self.FeedForward(inputs)  # [1] FeedForward pass
                self.BackProp(targets, lrate=lrate)    # [1] BackProp pass

                # [1] Record cost after each epoch if progress=True
                if progress:
                    Error = self.Loss(self.lyr[-1].h, targets)
                    self.cost_history.append(Error)
                    if np.mod(k, 20)==0:
                        print('Epoch '+str(k)+': Cost = '+str(Error))

            #========= YOUR IMPLEMENTATION ABOVE =========

        else:
            Learn(self, inputs, targets, lrate=lrate, epochs=epochs, progress=progress)

    def SGD(self, inputs, targets, lrate=0.05, epochs=1, batch_size=10):
        '''
            progress = net.SGD(inputs, targets, lrate=0.05, epochs=1, batch_size=10)

            Performs Stochastic Gradient Descent on the network.
            Run through the dataset in batches 'epochs' number of times, incrementing the
            network weights after each batch. For each epoch, it shuffles the dataset.

            Inputs:
              inputs  is an array of input samples
              targets is a corresponding array of targets
              lrate   is the learning rate (try 0.001 to 5)
              epochs  is the number of times to go through the training data
              batch_size is the number of samples for each batch

            Outputs:
              progress is an (epochs)x2 array with epoch in the first column, and
                      cost in the second column
        '''
        loss_history = []
        for k in range(epochs):
            batches = MakeBatches(inputs, targets, batch_size=batch_size, shuffle=True)
            for mini_batch in batches:
                self.FeedForward(mini_batch[0])
                self.BackProp(mini_batch[1], lrate=lrate)

            loss_history.append([k, self.Evaluate(inputs, targets)])
            print('Epoch '+str(k)+': cost '+str(loss_history[-1]))

        return np.array(loss_history)

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
        return 1. - float(n_incorrect) / len(inputs)





# end
