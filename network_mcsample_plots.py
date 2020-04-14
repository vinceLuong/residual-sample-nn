import sys
sys.path.insert(0,'C:\\Users\\pmcw9\\Winter 2020\\CS 698\\Project Final Folder\\network files')
import Class_Network as Network
import Class_generate_data as generate_data
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser


def RSNN_mcsample_test(X_train, X_test, y_train, y_test, h_nodes, epochs, lr, times, threshold, coefficient, type):
    """
    Trains a regular 3-layer our NN, and returns the accuracy results

    :param X_train: Numpy array/pandas dataframe, data for the keras model.
    :param X_test: Numpy array/pandas dataframe, data for the keras model.
    :param y_train: Numpy array/pandas dataframe, data for the keras model.
    :param y_test: Numpy array/pandas dataframe, data for the keras model.
    :param h_nodes: Int, number of nodes in the hidden layer.
    :param epochs: Int, number of epochs to train the NN.
    :param lr: Float, learning rate for training.
    :param times: Integer, number of times to MC sample.
    :param threshold: Integer, used as threshold for forming residual to sample from.
    :param coefficient: Float, used as the ratio for updating the variance.
    :param type: string, select type of loss function.
    :return: (training_acc, test_acc), training/testing accuracies
    """
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]

    #Initialize the network
    np.random.seed(10)  # Ensures each network is initialized the same way.
    net = Network.Network([in_dim, h_nodes, out_dim], type = type, pdw = ['gaussian']*2, pdb = ['gaussian']*2)
    net.Learn(X_train, y_train, epochs=epochs, lrate = lr, times = times, threshold = threshold, bootstrap = False, coefficient = coefficient)
    acc_train = net.ClassificationAccuracy(X_train, y_train)
    acc_test = net.ClassificationAccuracy(X_test, y_test)

    return(acc_train, acc_test)

def network_mcsample_plots(X_train, X_test, y_train, y_test, rs_nn_params, mc_samples):
    """
    Trains on a data set.

    Then plots are created and saved to the working directory comparing training/test accuracies during the training
    process for each neural network.

    :param X_train: numpy array/pandas dataframe, input data to train
    :param X_test: numpy array/pandas dataframe, input data to test
    :param y_train: numpy array/pandas dataframe, target data to train
    :param y_test: numpy array/pandas dataframe, target data to test
    :param rs_nn_params: Dict, parameters used to create and train the residual sameple neural net
    :param keras_params: Dict, parameters used to create and train the keras neural net
    :param decay_params: Dict, parameters used to create and train the keras neural net with decay
    :param mc_samples: Dict, used to set start,stop, and step size for various mc samples attempted.
    :return:
    """
    # Train a RS-NN
    mc_trials = np.arange(int(mc_samples['start']),int(mc_samples['stop'])+int(mc_samples['step']),int(mc_samples['step']))
    trials = len(mc_trials)
    train_acc = [0]*trials # For storing training accuracies.
    test_acc = [0]*trials # For storing test accuracies.

    for i, num_mcsample in enumerate(mc_trials):
        (RSNN_train_acc, RSNN_test_acc) = RSNN_mcsample_test(X_train, X_test, y_train, y_test,
                                int(rs_nn_params['h_nodes']), int(rs_nn_params['epochs']),
                                    float(rs_nn_params['lr']), num_mcsample,
                                    float(rs_nn_params['threshold']),
                                    float(rs_nn_params['coefficient']),type= rs_nn_params['type'])
        print('Complete %d'%i,'of %d'%trials)
        train_acc[i] = RSNN_train_acc
        test_acc[i] = RSNN_test_acc

    # Plot the accuracies

    plt.figure(0)
    plt.title("Training Accuracy vs. MC Samples")
    plt.scatter(mc_trials, train_acc)
    plt.xlabel('MC Samples')
    plt.ylabel('Accuracy')
    plt.savefig('train_acc_mcsamples.png')
    plt.show()

    plt.figure(1)
    plt.title("Test Accuracy vs. MC Samples")
    plt.scatter(mc_trials, test_acc)
    plt.xlabel('MC Samples')
    plt.ylabel('Accuracy')
    plt.savefig('test_acc_mcsamples.png')
    plt.show()


def main():
    config = configparser.ConfigParser()
    config.read('config_mcsample.ini')

    # select correct data
    train_size = int(config['DATA']['train_size'])
    test_size = int(config['DATA']['test_size'])

    if config['DATA']['dataset'] == "simulated":
        num_cov = int(config['DATA']['num_cov'])
        mu = float(config['DATA']['mu'])
        std = float(config['DATA']['std'])
        range_cov = float(config['DATA']['range_cov'])
        range_coef = float(config['DATA']['range_coef'])
        range_bias = float(config['DATA']['range_bias'])
        generator = generate_data.generate_data(train_size, num_cov,mu, std,
                                                range_cov, range_coef, range_bias, seed=100)# Maybe add to config file..
        X_train, y_train, yt = generator.generate(seed=15)
        X_test,y_test,yt = generator.generate(seed=16)


    if config['DATA']['dataset'] == "mnist":
        import mnist_loader
        train_full, validate_full, test_full = mnist_loader.load_data_wrapper() # we wont need validate dataset
        X_train = np.array(train_full[0][:train_size])
        y_train = np.array(train_full[1][:train_size])
        X_test = np.array(test_full[0][:test_size])
        y_test = np.array(test_full[1][:test_size])

    # run benchmarking function
    network_mcsample_plots(X_train, X_test, y_train, y_test, config['RS NN PARAMS'], config['MCSAMPLE'])

if __name__ == "__main__":
    main()
