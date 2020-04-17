import sys
sys.path.append("../residual-sample-nn") # for finding the source files
import GenerateData as generate_data
import Network as Network
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser
import mnist_loader

def RSNN_param_test(X_train, X_test, y_train, y_test, h_nodes, epochs, lr, times, threshold, coefficient, type):
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
    net = Network.Network([in_dim, h_nodes, out_dim], type = type, pdw =['gaussian'] * 2, pdb =['gaussian'] * 2)
    net.Learn(X_train, y_train, epochs=epochs, lrate = lr, times = times, threshold = threshold, bootstrap = False, coefficient = coefficient)
    acc_train = net.ClassificationAccuracy(X_train, y_train)
    acc_test = net.ClassificationAccuracy(X_test, y_test)

    return(acc_train, acc_test)

def network_lrate_plots(X_train, X_test, y_train, y_test, rs_nn_params, lr_samples, fold=False):
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
    :param lrate_samples: Dict, used to set start,stop, and number of points for various learning rate attempted.
    :param fold: Boolean, used to select whether folds are to be used (only iris).
    :return:
    """
    lr_trials = np.linspace(float(lr_samples['start']), float(lr_samples['stop']),num = int(lr_samples['points']))
    trials = len(lr_trials)
    train_acc = [0] * trials  # For storing training accuracies.
    test_acc = [0] * trials  # For storing test accuracies.
    if fold:
        # Train a RS-NN
        splits = len(X_train)
        for i, num_sample in enumerate(lr_trials):
            for j in range(0, splits):
                (RSNN_train_acc, RSNN_test_acc) = RSNN_param_test(X_train[j], X_test[j], y_train[j], y_test[j],
                                        int(rs_nn_params['h_nodes']), int(rs_nn_params['epochs']),
                                            num_sample, int(rs_nn_params['times']),
                                            float(rs_nn_params['threshold']),
                                            float(rs_nn_params['coefficient']),type= rs_nn_params['type'])
                train_acc[i] += np.array(RSNN_train_acc) / splits
                test_acc[i] += np.array(RSNN_test_acc) / splits

    else:
        # Train a RS-NN
        for i, num_sample in enumerate(lr_trials):
            (RSNN_train_acc, RSNN_test_acc) = RSNN_param_test(X_train, X_test, y_train, y_test,
                                                              int(rs_nn_params['h_nodes']), int(rs_nn_params['epochs']),
                                                              num_sample, int(rs_nn_params['times']),
                                                              float(rs_nn_params['threshold']),
                                                              float(rs_nn_params['coefficient']),
                                                              type=rs_nn_params['type'])
            train_acc[i] = RSNN_train_acc
            test_acc[i] = RSNN_test_acc

    # Plot the accuracies
    plt.figure(0)
    plt.title("Training Accuracy vs. Learning Rate")
    plt.scatter(lr_trials, train_acc)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.savefig('train_acc_lrsamples.png')

    plt.figure(1)
    plt.title("Test Accuracy vs. Learning Rate")
    plt.scatter(lr_trials, test_acc)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.savefig('test_acc_lrsamples.png')
    plt.show()

def main():
    config = configparser.ConfigParser()
    config.read('config_lr.ini')
    #config.read("config_mcsample.ini")

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
        generator = generate_data.GenerateData(num_cov, mu, std,
                                               range_cov, range_coef, range_bias, seed=100)# Maybe add to config file..
        X_train, y_train, _ = generator.generate(seed=15, sample_size=train_size)
        X_test, y_test, _ = generator.generate(seed=16, sample_size=test_size)
        network_lrate_plots(X_train, X_test, y_train, y_test, config['RS NN PARAMS'], config['LEARNINGRATE'])

    if config['DATA']['dataset'] == "mnist":
        train_full, validate_full, test_full = mnist_loader.load_data_wrapper() # we wont need validate dataset
        X_train = np.array(train_full[0][:train_size])
        y_train = np.array(train_full[1][:train_size])
        X_test = np.array(test_full[0][:test_size])
        y_test = np.array(test_full[1][:test_size])
        network_lrate_plots(X_train, X_test, y_train, y_test, config['RS NN PARAMS'], config['LEARNINGRATE'])

    if config['DATA']['dataset'] == "iris":
        from sklearn import datasets
        from sklearn.model_selection import KFold
        fold = True
        data = datasets.load_iris()
        y_train = pd.get_dummies(data.target).values
        X_train = data.data
        splits = int(config['DATA']['splits'])  # Number of splits selected.
        kf = KFold(splits)
        kf.get_n_splits(X_train)
        kf.split(X_train)
        Xtr = []
        Xtest = []
        ytr = []
        ytest = []
        for train_index, test_index in kf.split(X_train):
            # train
            # print("TRAIN:", train_index, "TEST:", test_index)
            Xtr.append(X_train[train_index])
            ytr.append(y_train[train_index])
            # test
            Xtest.append(X_train[test_index])
            ytest.append(y_train[test_index])
        network_lrate_plots(Xtr, Xtest, ytr, ytest, config['RS NN PARAMS'], config['LEARNINGRATE'], fold=fold)

    # run benchmarking function


if __name__ == "__main__":
    main()
