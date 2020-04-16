import numpy as np
import matplotlib.pyplot as plt

class generate_data():

    def __init__(self, p, mu, std, range_cov, range_coef, range_bias,seed=100):
        '''
        generator = generate_data(mu, std, rng)

        Creates a data generator, for all your data generating needs.

        Inputs:
         p: The number of features. (scalar)
         mu: Mean of the noise.
         std: Standard deviation of the noise..
         range_cov: Size of interval for uniform distribution to generate covariates of the data.
         range_coef: Size of interval for uniform distribution to generate coefficients of gen process.
         range_bias: Size of interval for uniform distribution to generate bias of gen process.
        '''
        np.random.seed(seed)
        # Parameters for generating data
        self.mu = mu
        self.std = std
        self.rng = range_cov
        self.rng_coef = range_coef
        self.rng_bias = range_bias
        self.p = p

        # Parameters of generating process.
        self.coef = np.random.uniform(size=p, low=-self.rng_coef,
                                      high=self.rng_coef)  # The coefficients for each co-varitate in the data generating process. (1D np.array of shape (p,))
        self.bias = np.random.uniform(size=1, low=-self.rng_bias,
                                      high=self.rng_bias)  # The RHS threshold in the data generating process. (scalar)

    def generate(self, sample_size, seed=100):
        '''

        x_data, noisy_labels, true_labels = generator.generate(n, p, seed)

        Crates data

        Inputs:
         sample_size: Sample size to generate. (Integer)
         seed: Seed for random generation, can be used to reproduce results. (Integer)
        Outputs:
          Returns 3 items.
          The first item returned are covariates of the data as an np.array with shape (n,p).
          The second item returned is the labels generated with noise with shape (n,1).
          The third item returned is the labels generated without nouse with shape (n,1).

        '''
        # Sets seed for generation of data.
        np.random.seed(seed)
        n = sample_size
        p = self.p  # Store the number of covariates.

        # Generating Process.
        eps = np.random.normal(loc=self.mu, scale=self.std, size=n)  # Creates noise to be used in generating y-labels.
        self.x = np.random.uniform(size=(n, p), low=-self.rng, high=self.rng)  # Creates covariates.
        self.y_true = (np.dot(self.x, self.coef) > self.bias) * np.ones((n))  # Creates y_true, labels without noise.
        self.y = (np.dot(self.x, self.coef) + eps > self.bias) * np.ones((n))  # Creates labels.

        return (self.x, np.reshape((self.y), (n, 1)), np.reshape((self.y_true), (n, 1)))

    def plot(self, truth=False):
        '''

        plot= Weight(n, p, coef, bias, seed, truth)

        Plots the space of the covariates and shows the decision boundary with their true labels.

        Inputs:
          truth: Whether to label using true or noisy labels. (boolean)

        '''

        if (truth == 1):
            x_zero = self.x[np.argwhere(self.y_true == 1)[:, 0].tolist()]  # Points of label 1.
            x_one = self.x[np.argwhere(self.y_true == 0)[:, 0].tolist()]  # Points of Label 0.
            title = 'True Labels'
        else:
            x_zero = self.x[np.argwhere(self.y == 1)[:, 0].tolist()]  # Points of label 1.
            x_one = self.x[np.argwhere(self.y == 0)[:, 0].tolist()]  # Points of Label 0.
            title = 'Noisy Labels'

        if (self.p == 1):  # Plot for when only 1 covariate
            plt.figure(100)
            plt.xlabel('Dimension 1')
            plt.title('Labelled data in covariate space' + '(' + title + ')')
            plt.plot([self.bias / self.coef] * 3, [0.5, 1, 1.5], c='black')
            plt.scatter(x_zero, np.ones(len(x_zero)), c='r')
            plt.scatter(x_one, np.ones(len(x_one)), c='b')
            plt.legend(['Decision Boundary', 'Label 0', 'Label 1'])
            plt.savefig('1D_generated_points.png')
            plt.show()


        elif (self.p == 2):  # Plot for when 2 covariates are available.
            plt.figure(100).savefig('2D_generated_points.png')
            plt.ylabel('Dimension 2')
            plt.xlabel('Dimension 1')
            plt.title('Labelled data in covariate space,' + '(' + title + ')')
            x_temp = np.linspace(start=-self.rng, stop=self.rng, num=10)  # Points to plot decision boundary.
            y_temp = (self.bias - self.coef[0] * x_temp) / self.coef[1]
            plt.plot(x_temp, y_temp, c='black')
            plt.scatter(x_zero[:, 0], x_zero[:, 1], c='r')
            plt.scatter(x_one[:, 0], x_one[:, 1], c='b')
            plt.legend(['Decision Boundary', 'Label 0', 'Label 1'])
            plt.savefig('2D_generated_points.png')

            plt.show()

        else:
            print('Sorry, not complete for more than 2 dimensions.')