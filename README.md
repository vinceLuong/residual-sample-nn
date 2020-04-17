# residual-sample-nn
A residual sample neural network is a neural network where between each layer in the network, the weights connecting the layer are sampled from a distribution. 
This way, the output of the network has a certain distribution.  We hope that this gives the result a neural network more interpretability. 

To train this network, the distribution of each the weights are updated.  To do so, we perform regular feedforward and backprop on a sample of weights, to get a collection of different updates. These different updates are then used to create the new distribution of each weight.  We hope to reduce overfitting by training our network with only a subset of the gradients in stochasitc gradient descent.  We choose only the gradients corresponding to input that have failed to meet their target output plus/mine a certain threshold.
  

This is the work has been done as part of coursework at the University of Waterloo.     
# Installation
This is a package that can be installed with the following command:

```pip install -i https://test.pypi.org/simple/ residual-sample-nn```

Furthermore, if you want to download the files on GitHub, that can be done.  The package files are contained under the ```residual-sample-nn``` folder. These are the python files that contain the implementation of the residual sample neural net.


# Testing/Benchmarking
To run the tests on the repo, you will need to download it from the GitHub page. Note that the code was developed on python 3.7.  

The testing/benchmarking files are stored under the ```tests``` folder. To run the tests, follow the steps below:  

1. Download the dependencies of the code base.  To do so, navigate to the root directory of the repository and type the following command into your console: <br/>
```pip install -r requirements.txt```

2. Go into the tests, folder and run any of the network testing files. As an example: <br/>
```python network_benchmark_plots.py```

3. Configure the tests to your desire of hyperparamaters/dataset with each appropriate config file. As an example, to configure ```network_lrate_plots.py```, edit the ```config_lr.ini``` file.  The accepted datasets are "simulated", "iris", and "mnist".***   


***To run the tests with the MNIST dataset, you will need an appropriate mnist.pkl file in the ```tests``` folder. 