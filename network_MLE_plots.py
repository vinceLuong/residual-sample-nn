import sys
sys.path.insert(0,'C:\\Users\\pmcw9\\Winter 2020\\CS 698\\Project Final Folder\\network files')
import Class_Network as Network
import Class_generate_data as generate_data
import numpy as np
import matplotlib.pyplot as plt
import configparser


def plot_MLE_weight(net, row, col, lyr, n):
    '''
    Calculates and plots the MLE for a selected weigths, help to
    identify if our network's MLE is correct.

    Then plots are created and saved to the working directory.

    :param net: Object (class Network), our neural network object.
    :param row: Integer, selected row of a weight matrix.
    :param col: Integer, selected column of a weight matrix.
    :param lyr: integer, selected weight matrices, there is one between each pair of layers.
    :param n: integer, number of points to evaluate at.
    :return:
    '''
    data = np.take(net.weight[lyr], indices=col, axis=2)[:, row]  # Extract data.
    mu = net.weight_matrix[lyr].mu[row, col]  # Mean as calcualated within our network.
    sigma = net.weight_matrix[lyr].sigma[row, col]  # Sigma as calculated within our network.
    plot_MLE(data, mu, sigma, n)


def plot_MLE_bias(net, ele, lyr, n):
    '''
    Calculates and plots the MLE for a selected bias, help to
    identify if our network's MLE is correct.

    Then plots are created and saved to the working directory.

    :param net: Object (class Network), our neural network object.
    :param ele: Integer, selected element of a bias vector.
    :param lyr: integer, selected weight matrices, there is one for each layer.
    :param n: integer, number of points to evaluate at.
    :return:
    '''
    data = np.take(net.bias[lyr], indices=0, axis=2)[:, ele]  # Extract data.
    mu = net.lyr[lyr + 1].bias_vector.mu[0][ele]  # Mean as calcualated within our network.
    sigma = net.lyr[lyr + 1].bias_vector.sigma[0][ele]  # Sigma as calculated within our network.
    plot_MLE(data, mu, sigma, n)


def plot_MLE(data, mu, sigma, n):
    '''
    Uses the updates fora particualr w to ensure the mu is the correct MLE.

    Then plots are created and saved to the working directory.

    :param data: (1D) numpy array dataframe, data to fit.
    :param mu: numpy array, mean parameter used.
    :param sigma: numpy array, standard deviation used.
    :param n: integer, number of points to evaluate at.
    :return:
    '''
    n = n  # number of values to try for the mean
    lyr = 0  # Select which weight matrix.
    col = 0  # Select column.
    row = 0  # Select row.

    eval_pts_l = np.linspace(start=mu - sigma, stop=mu, num=int(n / 2))  # points to eval at
    eval_pts_r = np.linspace(start=mu, stop=mu + sigma, num=int(n / 2))
    eval_pts = np.append(eval_pts_l, eval_pts_r[1:])
    output = [0] * len(eval_pts)
    for i, point in enumerate(eval_pts):
        output[i] = lnormal(data=data, mu=point, sigma=sigma)

    plt.figure(0)
    plt.title('Likelihood of data vs. mean parameter')
    plt.xlabel('Mean parameter')
    plt.ylabel('Likelihood')
    plt.plot(eval_pts, output)
    plt.scatter(mu, lnormal(data, mu, sigma), c='r')
    plt.legend(['Likelihood of tested values', 'Likelihood of our mean'])
    plt.savefig('likelihood_plot.png')
    plt.show()
    return (output)


def lnormal(data, mu, sigma):
    '''
    Calculates the normal pdf

    :param data: (1D) numpy array dataframe, data to fit.
    :param mu: numpy array, mean parameter used.
    :param sigma: numpy array, standard deviation used.
    :param n: integer, number of points to evaluate at.
    :return: Likelihood of data w.r.t parameters.
    '''
    return (np.prod(normal(data, mu, sigma)))


def normal(data, mu, sigma):
    '''
    Calculates the normal pdf

    :param data: (1D) numpy array dataframe, data to fit.
    :param mu: numpy array, mean parameter used.
    :param sigma: numpy array, standard deviation used.
    :param n: integer, number of points to evaluate at.
    :return: numpy array,
    '''
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(data - mu) ** 2 / (2 * (sigma ** 2)))
    return (pdf)

def main():
    config = configparser.ConfigParser()
    config.read('config_MLE.ini')

    # select correct data
    train_size = int(config['DATA']['train_size'])
    test_size = int(config['DATA']['test_size'])
    params = config['RS NN PARAMS']

    if config['DATA']['dataset'] == "simulated":
        num_cov = int(config['DATA']['num_cov'])
        mu = float(config['DATA']['mu'])
        std = float(config['DATA']['std'])
        range_cov = float(config['DATA']['range_cov'])
        range_coef = float(config['DATA']['range_coef'])
        range_bias = float(config['DATA']['range_bias'])
        generator = generate_data.generate_data(num_cov,mu, std,
                                                range_cov, range_coef, range_bias, seed=100)# Maybe add to config file..
        X_train, y_train, _ = generator.generate(seed=15, sample_size=train_size)

    # Init network
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]
    net = Network.Network([in_dim, int(params['h_nodes']), out_dim], type = params['type'],
                          pdw = ['gaussian']*2, pdb = ['gaussian']*3)

    net.Learn(X_train, y_train, epochs=int(params['epochs']), lrate=float(params['lr']),
              times=int(params['times']), threshold=float(params['threshold']), bootstrap=False,
              coefficient=float(params['coefficient']))
    user_io = config['DATA']
    if user_io['select'] == 'weight':
        plot_MLE_weight(net, row=int(user_io['row']), col=int(user_io['row']), lyr=int(user_io['layer']), n=int(user_io['n']))
    elif user_io['select'] == 'bias':
        plot_MLE_bias(net, ele=int(user_io['ele']), lyr=int(user_io['layer']), n=int(user_io['n']))
    else:
        print('Error: Not a valid selection of select = %s' %user_io['select'])



if __name__ == "__main__":
    main()
