import numpy as np

def knn_regression(X, X_test, y, k):
    '''For each test point in X_test we look for the k nearest neighbours
    and predict its output value by averaging all the output values of the neighbours

    Args:
        X: array of x values of training data
        X_test: arrays of of points where we want to perform a LWR
        y: array of y values of training data
        k: number of neighbours

    Returns:
        Yhat: Array of predicted values at points in X_test    
    '''
    Yhat = np.zeros(X_test.shape[0])
    for i, xq in enumerate(X_test):
        dist = np.sqrt((X - xq)**2)
        idx_dist_sorted = np.argsort(dist)
        knns = idx_dist_sorted[:k]
        Yhat[i] = np.mean(y[knns])
    
    return Yhat