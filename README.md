# residual-sample-nn
A residual sample neural network is a neural network where between each layer in the network, the weights connecting the layer are sampled from a distribution. 
This way, the output of the network has a certain distribution.  We hope that this gives the result a neural network more interpretability. 
To train this network, the distribution of each the weights are updated.  To do so, we perform regular feedforward and backprop on a sample of weights, to get a collection of different updates. These different updates are then used to create the new distribution of each weight.

This is the work has been done as part of coursework at the University of Waterloo.     
# Installation
