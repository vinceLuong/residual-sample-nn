import Class_Network as Network
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import regularizers
import configparser


def keras_test(X_train, X_test, y_train, y_test, h_nodes, epochs, batch_size, lr, weight_decay):
    """
    Trains a regular 3-layer neural net in keras, and returns the accuracy results

    :param X_train: numpy array/pandas dataframe, data for the keras model
    :param X_test: numpy array/pandas dataframe, data for the keras model
    :param y_train: numpy array/pandas dataframe, data for the keras model
    :param y_test: numpy array/pandas dataframe, data for the keras model
    :param h_nodes: int, number of nodes in the hidden layer
    :param epochs: int, number of epochs to train the NN
    :param batch_size: int, size of a batch when training
    :param lr: float, learning rate for training
    :param weight_decay: float, l2-norm weight decay constant
    :return: (training_acc, test_acc), lists of training/testing accuracies
    """

    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]

    keras_model = Sequential()
    keras_model.add(Dense(h_nodes, input_dim=in_dim, activation='sigmoid',
                          kernel_regularizer=regularizers.l2(weight_decay)))
    keras_model.add(Dense(out_dim, activation='sigmoid',
                          kernel_regularizer=regularizers.l2(weight_decay)))

    optimizer = optimizers.SGD(lr=lr)
    keras_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = keras_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                              validation_data=(X_test, y_test))

    return (history.history['acc'], history.history['val_acc'])


def network_benchmark_plots(X_train, X_test, y_train, y_test, rs_nn_params, keras_params, decay_params):
    """
    Train a dataset with 3 different models.
    1. A Residual Sample Neural Network
    2. A regular neural net in keras
    3. A regular neural net in keras with weight decay

    Then plots are created and saved to the working directory comparing training/test accuracies during the training
    process for each neural network.

    :param X_train: numpy array/pandas dataframe, input data to train
    :param X_test: numpy array/pandas dataframe, input data to test
    :param y_train: numpy array/pandas dataframe, target data to train
    :param y_test: numpy array/pandas dataframe, target data to test
    :param rs_nn_params: Dict, parameters used to create and train the residual sameple neural net
    :param keras_params: Dict, parameters used to create and train the keras neural net
    :param decay_params: Dict, parameters used to create and train the keras neural net with decay
    :return:
    """
    # Train a RS-NN

    # Train a regular NN
    (keras_train_acc, keras_test_acc) = keras_test(X_train, X_test, y_train, y_test, int(keras_params['h_nodes']),
                                                   int(keras_params['epochs']), int(keras_params['batch_size']),
                                                   float(keras_params['lr']), weight_decay=0)

    # Train a regular NN with weight decay
    (decay_train_acc, decay_test_acc) = keras_test(X_train, X_test, y_train, y_test, int(decay_params['decay']['h_nodes']),
                                                   int(decay_params['epochs']), int(decay_params['batch_size']),
                                                   float(decay_params['lr']), float(decay_params['weight_decay']))

    # plot the accuracies


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    # select correct data
    train_size = int(config['data']['train_size'])
    test_size = int(config['data']['test_size'])

    if config['data']['dataset'] == "simulated":
        X_train, X_test, y_train, y_test = []

    if config['data']['dataset'] == "mnist":
        import mnist_loader
        train_full, validate_full, test_full = mnist_loader.load_data_wrapper() # we wont need validate dataset
        X_train = np.array(train_full[0][:train_size])
        y_train = np.array(train_full[1][:train_size])
        X_test = np.array(test_full[0][:test_size])
        y_test = np.array(test_full[1][:test_size])

    # run benchmarking function
    network_benchmark_plots(X_train, X_test, y_train, y_test, config['RS NN PARAMS'], config['KERAS PARAMS'], config['DECAY PARAMS'])


if __name__ == "__main__":
    main()
