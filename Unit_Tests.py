# Standard imports.
import unittest
import numpy as np
import random
# Import functions that need to be tested.
from Class_Network import *

class TestNetwork(unittest.TestCase):
    def test_Learn(self):
        inputs = [[1.53243,0.4354657],[0.468873,1.425436557]]
        label = [[1],[1]]
        BernoulliNet = Network([2,1], type='bernoulli',pdw=None,pdb=None)
        # Check the the loss changes or not after learning one epoch.
        cost_history_before = BernoulliNet.cost_history.copy # Make sure it will not change.
        loss_before = BernoulliNet.Evaluate(inputs,label)
        cost_history_after = BernoulliNet.Learn(inputs,label,epochs=1)
        loss_after = BernoulliNet.Evaluate(inputs,label)
        self.assertNotEqual(loss_before, loss_after)
        # Check the cost history, should be updated
        self.assertNotEqual(cost_history_before, cost_history_after)
        self.assertEqual(len(cost_history_after),1)


class TestNetworkFunctions(unittest.TestCase):
    def test_Initialization(self):
        # Define different types of network.
        BernoulliNet = Network([2,1], type='bernoulli', pdw='gaussian',pdb=None)
        ClassifierNet = Network([2,1], type='classifier',pdw=None,pdb=None)
        RegressionNet = Network([2,1], type='regression', pdw=['gaussian'],pdb = ['gaussian'])
        # Check classifier.
        self.assertTrue(BernoulliNet.classifier)
        self.assertTrue(ClassifierNet.classifier)
        self.assertFalse(RegressionNet.classifier)
        # Check loss function.
        self.assertEqual(BernoulliNet.Loss, CrossEntropy)
        self.assertEqual(ClassifierNet.Loss, CategoricalCE)
        self.assertEqual(RegressionNet.Loss, MSE)
        # Check gradient of loss function.
        self.assertEqual(BernoulliNet.gradLoss, gradCrossEntropy)
        self.assertEqual(ClassifierNet.gradLoss, None)
        self.assertEqual(RegressionNet.gradLoss, gradMSE)
        # Check activation function in first layer, all using logistic.
        self.assertEqual(BernoulliNet.lyr[0].sigma,BernoulliNet.lyr[0].Logistic)
        self.assertEqual(BernoulliNet.lyr[0].sigma_p,BernoulliNet.lyr[0].Logistic_p)
        self.assertEqual(ClassifierNet.lyr[0].sigma,ClassifierNet.lyr[0].Logistic)
        self.assertEqual(ClassifierNet.lyr[0].sigma_p,ClassifierNet.lyr[0].Logistic_p)
        self.assertEqual(RegressionNet.lyr[0].sigma,RegressionNet.lyr[0].Logistic)
        self.assertEqual(RegressionNet.lyr[0].sigma_p,RegressionNet.lyr[0].Logistic_p)
        # Check activation function in second layer, all different.
        self.assertEqual(BernoulliNet.lyr[1].sigma,BernoulliNet.lyr[1].Logistic)
        self.assertEqual(BernoulliNet.lyr[1].sigma_p,BernoulliNet.lyr[1].Logistic_p)
        self.assertEqual(ClassifierNet.lyr[1].sigma,ClassifierNet.lyr[1].Softmax)
        self.assertEqual(ClassifierNet.lyr[1].sigma_p,None)
        self.assertEqual(RegressionNet.lyr[1].sigma,RegressionNet.lyr[1].Identity)
        self.assertEqual(RegressionNet.lyr[1].sigma_p,RegressionNet.lyr[1].Identity_p)
        # Check prior distributions for weights and biases.
        self.assertEqual(BernoulliNet.weight_matrix[0].dis, 'gaussian')
        self.assertEqual(BernoulliNet.lyr[0].bias_vector.dis, 'gaussian')
        self.assertEqual(BernoulliNet.lyr[1].bias_vector.dis, 'gaussian')
        self.assertEqual(ClassifierNet.weight_matrix[0].dis, 'gaussian')
        self.assertEqual(ClassifierNet.lyr[0].bias_vector.dis, 'gaussian')
        self.assertEqual(ClassifierNet.lyr[1].bias_vector.dis, 'gaussian')
        self.assertEqual(RegressionNet.weight_matrix[0].dis, 'gaussian')
        self.assertEqual(RegressionNet.lyr[0].bias_vector.dis, 'gaussian')
        self.assertEqual(RegressionNet.lyr[1].bias_vector.dis, 'gaussian')
        # Check distribution parameters are setup correctly or not.
        self.assertEqual(BernoulliNet.weight_matrix[0].mu.all(), np.zeros((2,1)).all())
        self.assertEqual(BernoulliNet.weight_matrix[0].sigma.all(), np.ones((2,1)).all())
        self.assertEqual(BernoulliNet.lyr[0].bias_vector.mu.all(), np.zeros((1,2)).all())
        self.assertEqual(BernoulliNet.lyr[0].bias_vector.sigma.all(), np.zeros((1,2)).all())
    
    def test_Evaulate(self):
        inputs = [[1.53243,0.4354657],[0.468873,1.425436557]]
        label = [[1],[1]]
        # Define a bernoulli network
        BernoulliNet = Network([2,1], type='bernoulli',pdw=None,pdb=None)
        # Check various input types, should be equal.
        result1 = BernoulliNet.Evaluate(inputs,label,1)
        result2 = BernoulliNet.Evaluate(inputs,label,1.5)
        self.assertEqual(result1, result2)
        # Check the loss it outputs is greater than zero.
        self.assertGreater(result1, 0)
        # Check if we FeedForward twice, the results are different.
        random.seed(1)
        result1 = BernoulliNet.Evaluate(inputs,label,1)
        result2 = BernoulliNet.Evaluate(inputs,label,2)
        result3 = BernoulliNet.Evaluate(inputs,label,2)
        self.assertNotEqual(result1,result2)
        self.assertNotEqual(result1,result3)

    def test_ClassificationAccuracy(self):
        inputs = [[1.53243,0.4354657],[0.468873,1.425436557],[1.64353, 2.65756]]
        label = [[0,1],[1,0],[1,0]]
        # Define a different types of networks.
        BernoulliNet = Network([2,2], type='bernoulli',pdw=None,pdb=None)
        ClassifierNet = Network([2,2], type='classifier',pdw=None,pdb=None)
        RegressionNet = Network([2,2], type='regression',pdw=None,pdb=None)
        # Regression type network don't have classification accuracy.
        Regression_accuracy = RegressionNet.ClassificationAccuracy(inputs,label)
        self.assertEqual(Regression_accuracy, None)
        # For classifier and bernoulli network, the returned classification accuracy should between 0 and 1.
        Classifier_accuracy = ClassifierNet.ClassificationAccuracy(inputs,label)
        Bernoulli_accuracy = BernoulliNet.ClassificationAccuracy(inputs,label)
        self.assertGreaterEqual(Classifier_accuracy, 0)
        self.assertGreaterEqual(Bernoulli_accuracy, 0)
        self.assertLessEqual(Classifier_accuracy,1)
        self.assertLessEqual(Bernoulli_accuracy,1)
        # Also check for FeedForward multiple times.
        Classifier_accuracy_twice = ClassifierNet.ClassificationAccuracy(inputs,label,2)
        Bernoulli_accuracy_twice = BernoulliNet.ClassificationAccuracy(inputs,label,2)
        self.assertGreaterEqual(Classifier_accuracy_twice, 0)
        self.assertGreaterEqual(Bernoulli_accuracy_twice, 0)
        self.assertLessEqual(Classifier_accuracy_twice,1)
        self.assertLessEqual(Bernoulli_accuracy_twice,1)
        # Note: We should also check setting "times" = 2, will change the results.
        # However, random.seed() doesn't fix the sampled weights and biases.
        # Therefore, the tests described here will not be included in the unittest.

    def test_TopGradient(self):
        inputs = [[1.53243,0.4354657],[0.468873,1.425436557],[1.64353, 2.65756]]
        label = [[0,1],[1,0],[1,0]]
        # Define a different types of networks.
        BernoulliNet = Network([2,2], type='bernoulli',pdw=None,pdb=None)
        ClassifierNet = Network([2,2], type='classifier',pdw=None,pdb=None)
        RegressionNet = Network([2,2], type='regression',pdw=None,pdb=None)
        # Define top gradients for all 3 networks.
        Bernoulli_TopGradient = BernoulliNet.TopGradient(inputs,label)
        Classifier_TopGradient = ClassifierNet.TopGradient(inputs,label)
        Regression_TopGradient = RegressionNet.TopGradient(inputs,label)
        # All 3 top gradients should be the same.
        self.assertEqual(Bernoulli_TopGradient.all(),Classifier_TopGradient.all())
        self.assertEqual(Bernoulli_TopGradient.all(),Regression_TopGradient.all())

class TestSuppliedFunctions(unittest.TestCase):

    def test_NSamples(self):
        x = [1.5,0.3,4]
        np_x = np.array([1.5,0.3,4])
        x2d = [[1.5,0.3,4],[4.3,5.9,6.2]]
        # List compare with numpy array should equal.
        self.assertEqual(NSamples(x),NSamples(np_x))
        # Check 2D list and numpy array, should return 2 since convention is row vector.
        self.assertEqual(NSamples(x2d), 2)
        self.assertEqual(NSamples(np.array(x2d)), 2)

    def test_OneHot(self):
        # Input must be 2D.
        x = [[1.5,0.3,4.7]]
        np_x = np.array(x)
        # Output should be type np.ndarray.
        self.assertIsInstance(OneHot(x), np.ndarray)
        # Check OneHot encoding. assertEqual checking two numpy arrays will fail since
        # the numpy array objects are stored at different locations.
        self.assertEqual(OneHot(x)[0][0], 0.)
        self.assertEqual(OneHot(x)[0][1], 0.)
        self.assertEqual(OneHot(x)[0][2], 1.)
        self.assertEqual(OneHot(np_x)[0][0], 0.)
        self.assertEqual(OneHot(np_x)[0][1], 0.)
        self.assertEqual(OneHot(np_x)[0][2], 1.)

    def test_CrossEntropy(self):
        y = [[0.2,0.8],[0.3865, 0.6135]]
        t = [[0,1],[1,0]]
        np_y = np.array(y)
        np_t = np.array(t)
        # List should return same results as numpy array.
        self.assertEqual(CrossEntropy(y,t), (CrossEntropy(np_y, np_t)))
        # Check CrossEntropy output value.
        self.assertEqual(CrossEntropy(np_y, np_t), 1.17376696226887)

    def test_gradCrossEntropy(self):
        y = [[0.2,0.8],[0.3865, 0.6135]]
        t = [[0,1],[1,0]]
        np_y = np.array(y)
        np_t = np.array(t)
        # List should return same results as numpy array.
        self.assertEqual(gradCrossEntropy(y,t)[0][0], gradCrossEntropy(np_y,np_t)[0][0])
        self.assertEqual(gradCrossEntropy(y,t)[1][0], gradCrossEntropy(np_y,np_t)[1][0])
        # Check gradCrossEntropy output value.
        self.assertEqual(gradCrossEntropy(np_y,np_t)[0][0], 0.625)
        self.assertEqual(gradCrossEntropy(np_y,np_t)[1][0], -1.2936610608020698)

    def test_MSE(self):
        y = [5.4354435, 544.534536]
        t = [6, 545]
        np_y = np.array(y)
        np_t = np.array(t)
        # List should return same results as numpy array.
        self.assertEqual(MSE(y,t), MSE(np_y,np_t))
        # Check MSE output value.
        self.assertEqual(MSE(np_y, np_t), 0.1338451942470619)

    def test_gradMSE(self):
        y = [[5.4354435],[544.534536]]
        t = [[6], [545]]
        np_y = np.array(y)
        np_t = np.array(t)
        # List should return same results as numpy array.
        self.assertEqual(gradMSE(y,t)[0][0], gradMSE(np_y,np_t)[0][0])
        self.assertEqual(gradMSE(y,t)[1][0], gradMSE(np_y,np_t)[1][0])
        # Check MSE output value.
        self.assertEqual(gradMSE(np_y, np_t)[0][0], -0.28227825000000006)
        self.assertEqual(gradMSE(np_y, np_t)[1][0], -0.2327319999999986)

    def test_CategoricalCE(self):
        y = [0.2,0.3,0.5]
        t = [0,1,0]
        np_y = np.array(y)
        np_t = np.array(t)
        # List should return same results as numpy array.
        self.assertEqual(CategoricalCE(y,t),CategoricalCE(np_y, np_t))
        # Check MSE output value.
        self.assertEqual(CategoricalCE(np_y, np_t), 0.40132426810864535)
    
    def test_Shuffle(self):
        y = np.array([[1.87,2.94],[7.324,8.453]])
        t = np.array([[0],[1]])
        # Set seed so the results will be the same.
        random.seed(1)
        shuffled_y, shuffled_t = Shuffle(y,t)
        # Check output is numpy array.
        self.assertIsInstance(shuffled_y, np.ndarray)
        self.assertIsInstance(shuffled_t, np.ndarray)
        # Check shuffled output values.
        self.assertEqual(shuffled_y.all(), y.all())
        self.assertEqual(shuffled_t.all(), t.all())

    def test_MakeBatches(self):
        y = np.array([[1.87,2.94],[7.324,8.453]])
        t = np.array([[0],[1]])
        # Since We checked Shuffle above, therefore here for simplicity,
        # We are going to set shuffle = False, and compare all elements in batch1 and 2.
        batch1 = MakeBatches(y,t,batch_size=1,shuffle=0)
        batch2 = MakeBatches(y,t,batch_size=-1,shuffle =False)
        self.assertEqual(np.array(batch1[0][0]).all(), np.array(batch2[0][0]).all())
        self.assertEqual(np.array(batch1[0][1]).all(), np.array(batch2[0][1]).all())
        self.assertEqual(np.array(batch1[1][0]).all(), np.array(batch2[1][0]).all())
        self.assertEqual(np.array(batch1[1][1]).all(), np.array(batch2[1][1]).all())

if __name__ == '__main__':
    unittest.main()