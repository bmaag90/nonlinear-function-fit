import numpy as np

def kernel_weights(X, xq, tau):
    ''' Calculates the kernel weights, with width tau, of a point xq in relation to all the points in X 
    
    Args:
        X: array of x values of training data
        xq: single point, which we want to calculate the weights for
        tau: kernel width
    
    Returns:
        W: Diagonal matrix holding the weights for xq in relation to each point in X
    '''
    m = X.shape[0]
    W = np.eye(m)
    for i in range(m):
        W[i,i] = np.exp(-1*(xq-X[i]).T*(xq-X[i])/(2*tau**2) )
        
    return W

def predict_weighted_regression_single_point(X, xq, y, tau):
    ''' Performs a weighted regression for a single point xq

    Args:
        X: array of x values of training data
        xq: single point, which we want to perform the LWR for
        y: array of y values of training data
        tau: kernel width
    
    Returns:
        yhat: Predicted value at point xq
    
    Note: 
        Small values of tau may lead to singular matrices W*Xs and W*y!
    '''
    m = X.shape[0]
    Xs = np.hstack([X.reshape(-1,1), np.ones((m, 1))])
    W = kernel_weights(X, xq, tau)
    
    A = np.matmul(Xs.T, np.matmul(W, Xs))
    B = np.matmul(Xs.T, np.matmul(W, y))
    
    beta = np.linalg.solve(A, B).reshape(-1,1)
    
    yhat = np.matmul(np.array([xq, 1]), beta)
    
    return yhat

def predict_weighted_regression(X, X_test, y, tau):
    ''' Performs a weighted regression for an array of test points X_test

    Args:
        X: array of x values of training data
        X_test: arrays of of points where we want to perform a LWR
        y: array of y values of training data
        tau: kernel width
    
    Returns:
        Yhat: Array of predicted values at points in X_test
    '''
    m = X_test.shape[0]
    Yhat = np.ones((m,1))
    for i, xq in enumerate(X_test):
        Yhat[i, 0] = predict_weighted_regression_single_point(X, xq, y, tau)   

    return Yhat