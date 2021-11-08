import numpy as np

def eval_fx(x, str_fx):
    ''' Evals a string of a function at points in array x   

    Args: 
        str_fx: string describing the underlying function, e.g 0.5*x + np.sin(0.5*x)
            Needs to use "x" describing the values where the function is evaluated at and numpy as np
            to utilize different functions available in numpy

    Returns:
        Fx: Function evaluated at x

    '''
    Fx = eval(str_fx)   
    return Fx

def noise(noise_mu=0.0, noise_sigma=1, m=50):
    ''' Generates m samples of normally-distributed noise with dist-mean mu and standard deviation sigma
    
    Args: 
        noise_mu: mean of normally distributed noise
        noise_sigma: standard devation of normally distributed noise
        m: number of training samples
    
    Returns: 
        N: array of shape of length m of random noise

    '''
    N = np.random.normal(loc=noise_mu, scale=noise_sigma, size=m)
    
    return N

def create_dataset(x_start, x_end, x_step, noise_mu, noise_sigma, m, str_fx):
    ''' Generates dataset

    Args:
        x_start: start of x range
        x_end: end of x range
        x_step: steps to generate points within [x_start, x_end]
        noise_mu: mean of normally distributed noise
        noise_sigma: standard devation of normally distributed noise
        m: number of training samples
        str_fx: str of the underlying function

    Returns:
        X: array of x values of training data
        y: array of y values of training data
        X_linspace: array of equally spaced values within [x_start, x_end]
    '''
    # Generate an equally spaced range within the x-range, e.g. [0, 0.1, 0.2, ..., 10]
    X_linspace = np.linspace(x_start, x_end, int((x_end - x_start)/x_step))
    # Randomly sample points within the x range to use as training data
    X = (x_end - x_start)*np.random.random_sample(size=m) + x_start
    # evaluate the underlying function at X and add noise to use as training data
    y = eval_fx(X, str_fx) + noise(noise_mu, noise_sigma, m)

    return X, y, X_linspace