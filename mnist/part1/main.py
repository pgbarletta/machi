import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################

# def closed_form(X, Y, lambda_factor):
#     """
#     Computes the closed form solution of linear regression with L2 regularization

#     Args:
#         X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
#         Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
#             data point
#         lambda_factor - the regularization constant (scalar)
#     Returns:
#         theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
#         represents the y-axis intercept of the model and therefore X[0] = 1
#     """

#     A = (X.T @ X) + (np.identity(X.shape[1]) * lambda_factor)
#     b = X.T @ Y

#     rtdo = np.linalg.inv(A) @ b
    
#     return rtdo

# def compute_test_error_linear(test_x, Y, theta):
#     test_y_predict = np.round(np.dot(test_x, theta))
#     test_y_predict[test_y_predict < 0] = 0
#     test_y_predict[test_y_predict > 9] = 9
#     return 1 - np.mean(test_y_predict == Y)

# def run_linear_regression_on_MNIST(lambda_factor=.01):
#     """
#     Trains linear regression, classifies test data, computes test error on test set

#     Returns:
#         Final test error
#     """
#     train_x, train_y, test_x, test_y = get_MNIST_data()
#     train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
#     test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
#     theta = closed_form(train_x_bias, train_y, lambda_factor)
#     test_error = compute_test_error_linear(test_x_bias, test_y, theta)
#     return test_error


# Don't run this until the relevant functions in linear_regression.py have been fully implemented.
# print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=1))


#######################################################################
# 3. Support Vector Machine
#######################################################################

# TODO: first fill out functions in svm.py, or the functions below will not work

# def one_vs_rest_svm(train_x, train_y, test_x):
#     """
#     Trains a linear SVM for binary classifciation

#     Args:
#         train_x - (n, d) NumPy array (n datapoints each with d features)
#         train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
#         test_x - (m, d) NumPy array (m datapoints each with d features)
#     Returns:
#         pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
#     """
#     lsvc = LinearSVC(random_state = 0, C=1.)

#     lsvc.fit(train_x, train_y)
    
#     pred_test_y = lsvc.predict(test_x)
    
#     return pred_test_y

# def multi_class_svm(train_x, train_y, test_x):
#     """
#     Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

#     Args:
#         train_x - (n, d) NumPy array (n datapoints each with d features)
#         train_y - (n, ) NumPy array containing the labels (int) for each training data point
#         test_x - (m, d) NumPy array (m datapoints each with d features)
#     Returns:
#         pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
#     """

#     lsvc = LinearSVC(random_state = 0, C=.1, multi_class= 'ovr')

#     lsvc.fit(train_x, train_y)
    
#     pred_test_y = lsvc.predict(test_x)
    
#     return pred_test_y

# def run_svm_one_vs_rest_on_MNIST():
#     """
#     Trains svm, classifies test data, computes test error on test set

#     Returns:
#         Test error for the binary svm
#     """
#     train_x, train_y, test_x, test_y = get_MNIST_data()
#     train_y[train_y != 0] = 1
#     test_y[test_y != 0] = 1
#     pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
#     test_error = compute_test_error_svm(test_y, pred_test_y)
#     return test_error


# print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


# def run_multiclass_svm_on_MNIST():
#     """
#     Trains svm, classifies test data, computes test error on test set

#     Returns:
#         Test error for the binary svm
#     """
#     train_x, train_y, test_x, test_y = get_MNIST_data()
#     pred_test_y = multi_class_svm(train_x, train_y, test_x)
#     test_error = compute_test_error_svm(test_y, pred_test_y)
#     return test_error


# print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

# #######################################################################
# # 4. Multinomial (Softmax) Regression and Gradient Descent
# #######################################################################

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    xt = (theta @ X.T) / temp_parameter
    c = np.tile(np.max(xt, axis = 0), (xt.shape[0], 1))

    xtc = xt - c

    H = np.exp(xtc) / np.sum(np.exp(xtc), axis=0) 

    return H


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    n = X.shape[0]
    k = theta.shape[0]
    d = theta.shape[1]

    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()
    p = compute_probabilities(X, theta, temp_parameter)

    termino_1 = ((M - p)  @ X) / (n * temp_parameter)
    termino_2 = lambda_factor * theta
    gdte = termino_2 - termino_1

    return theta - (alpha * gdte)


def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    # write_pickle_data(theta, "./theta.pkl.gz")

    
    return test_error


# print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1.))

# TODO: Find the error rate for temp_parameter = [.5, 1.0, 2.0]
#      Remember to return the tempParameter to 1, and re-run run_softmax_on_MNIST

# #######################################################################
# # 6. Changing Labels
# #######################################################################

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    return (train_y % 3, test_y % 3)

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    
    y_own = get_classification(X, theta, temp_parameter) % 3
    error = 0
    for x, b in zip(Y, y_own):
        error += (x != b)

    return (error / len(Y))

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def run_softmax_on_MNIST_mod3(temp_parameter=1.):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """

    # train_x, train_y, test_x, test_y = get_MNIST_data()
    # train_y, test_y = update_y(train_y, test_y)
    
    # get_classification(X, theta, temp_parameter)
    # test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    # write_pickle_data(theta, "./theta.pkl.gz")

    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y, test_y = update_y(train_y, test_y)
    theta, cost_function_history = softmax_regression(train_x, train_y, 
    temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    
    train_y, test_y = update_y(train_y, test_y)
    test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
    
    return test_error

# TODO: Run run_softmax_on_MNIST_mod3(), report the error rate
# print('softmax test_error=', run_softmax_on_MNIST_mod3(temp_parameter=1.))

# #######################################################################
# # 7. Classification Using Manually Crafted Features
# #######################################################################

# ## Dimensionality reduction via PCA ##

# # TODO: First fill out the PCA functions in features.py as the below code depends on them.


# n_components = 18

###Correction note:  the following 4 lines have been modified since release.
# train_x_centered, feature_means = center_data(train_x)
# pcs = principal_components(train_x_centered)
# train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
# test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

# temp_parameter = 1.
# train_x, train_y, test_x, test_y = get_MNIST_data()
# theta, cost_function_history = softmax_regression(train_pca, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
# plot_cost_function_over_time(cost_function_history)
# test_error = compute_test_error(test_pca, test_y, theta, temp_parameter)

# print('test_error=', test_error)

# # train_pca (and test_pca) is a representation of our training (and test) data
# # after projecting each example onto the first 18 principal components.


# # TODO: Train your softmax regression model using (train_pca, train_y)
# #       and evaluate its accuracy on (test_pca, test_y).


# # TODO: Use the plot_PC function in features.py to produce scatterplot
# #       of the first 100 MNIST images, as represented in the space spanned by the
# #       first 2 principal components found above.
# plot_PC(train_x[range(000, 100), ], pcs, train_y[range(000, 100)], feature_means)#feature_means added since release


# # TODO: Use the reconstruct_PC function in features.py to show
# #       the first and second MNIST images as reconstructed solely from
# #       their 18-dimensional principal component representation.
# #       Compare the reconstructed images with the originals.
# firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x, feature_means)#feature_means added since release
# plot_images(firstimage_reconstructed)
# plot_images(train_x[0, ])

# secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x, feature_means)#feature_means added since release
# plot_images(secondimage_reconstructed)
# plot_images(train_x[1, ])


## Cubic Kernel ##
# TODO: Find the 10-dimensional PCA representation of the training and test set


# TODO: First fill out cubicFeatures() function in features.py as the below code requires it.

train_x, train_y, test_x, test_y = get_MNIST_data()
train_x_centered, feature_means = center_data(train_x)
test_x_centered, feature_means_test = center_data(test_x)
pcs_train = principal_components(train_x_centered)
pcs_test = principal_components(test_x_centered)

n_components = 10
train_pca10 = project_onto_PC(train_x, pcs_train, n_components, feature_means)
# no sé pq reutilizo los PCs de la train data en vez de usar
# los PCs de la test data
# test_pca10 = project_onto_PC(test_x, pcs_test, n_components, feature_means_test)
test_pca10 = project_onto_PC(test_x, pcs_train, n_components, feature_means)

train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)

temp_parameter = 1.
theta, cost_function_history = softmax_regression(train_cube, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
plot_cost_function_over_time(cost_function_history)
test_error = compute_test_error(test_cube, test_y, theta, temp_parameter)

print('PCA + cubic features - test_error=', test_error)

# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.


# TODO: Train your softmax regression model using (train_cube, train_y)
#       and evaluate its accuracy on (test_cube, test_y).
