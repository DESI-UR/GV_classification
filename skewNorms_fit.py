
import numpy as np

from scipy.stats import skewnorm
from scipy.special import gammaln
from scipy.optimize import minimize


################################################################################
################################################################################

def skewNorms2_diff(x, params):
    '''
    Calculate the difference between two skew normal functions
    '''
    
    a, mu_a, sigma_a, skew_a = params[:4]
    b, mu_b, sigma_b, skew_b = params[4:]
    
    return a*skewnorm.pdf(x, skew_a, loc=mu_a, scale=sigma_a) - \
           b*skewnorm.pdf(x, skew_b, loc=mu_b, scale=sigma_b)

################################################################################
################################################################################

def skewNorms2(params, x):
    '''
    Mixture of two skew normal distributions.
    '''
    
    a, mu_a, sigma_a, skew_a = params[:4]
    b, mu_b, sigma_b, skew_b = params[4:]
    
    return a*skewnorm.pdf(x, skew_a, loc=mu_a, scale=sigma_a) + b*skewnorm.pdf(x, skew_b, loc=mu_b, scale=sigma_b)

################################################################################
################################################################################

def skewNorms3(params, x):
    '''
    Mixture of three skew normal distributions
    '''

    a, mu_a, sigma_a, skew_a = params[:4]
    b, mu_b, sigma_b, skew_b = params[4:8]
    c, mu_c, sigma_c, skew_c = params[8:]

    return a*skewnorm.pdf(x, skew_a, loc=mu_a, scale=sigma_a) + \
           b*skewnorm.pdf(x, skew_b, loc=mu_b, scale=sigma_b) + \
           c*skewnorm.pdf(x, skew_c, loc=mu_c, scale=sigma_c)


################################################################################
################################################################################

def logL_skewNorms2(params, m, x):
    """Log-likelihood of the data set for the skew normal mixture model.
    
    Parameters
    ----------
    params : list or ndarray
        List of 8 parameters.
    m : ndarray
        Binned counts in data set.
    x : ndarray
        Bin centers used to construct the histogrammed counts m.
        
    Returns
    -------
    logL : float
        Log likelihood of set m given model parameters.
    """
    
    lambda1 = skewNorms2(params, x)
    lambda1[lambda1<=0] = np.finfo(dtype=np.float64).tiny
    
    return np.sum(m*np.log(lambda1) - lambda1 - gammaln(m + 1))

################################################################################
################################################################################

def logL_skewNorms3(params, m, x):
    """Log-likelihood of the data set for the skew normal mixture model.
    
    Parameters
    ----------
    params : list or ndarray
        List of 12 parameters.
    m : ndarray
        Binned counts in data set.
    x : ndarray
        Bin centers used to construct the histogrammed counts m.
        
    Returns
    -------
    logL : float
        Log likelihood of set m given model parameters.
    """
    
    lambda1 = skewNorms3(params, x)
    lambda1[lambda1<=0] = np.finfo(dtype=np.float64).tiny
    
    return np.sum(m*np.log(lambda1) - lambda1 - gammaln(m + 1))

################################################################################
################################################################################

def nlogL_skewNorms2(params, m, x):
    """Negative log-likelihood, for minimizers."""
    return -logL_skewNorms2(params, m, x)

################################################################################
################################################################################

def nlogL_skewNorms3(params, m, x):
    """Negative log-likelihood, for minimizers."""
    return -logL_skewNorms3(params, m, x)

################################################################################
################################################################################

def uniform(a, b, u):
    """Given u in [0,1], return a uniform number in [a,b]."""
    return a + (b-a)*u

################################################################################
################################################################################

def jeffreys(a, b, u):
    """Given u in [0,1], return a Jeffreys random number in [a,b]."""
    return a**(1-u) * b**u

################################################################################
################################################################################

def prior_xforSkewNorms2(u):
    """
    Priors for the 8 parameters of the skew normal mixture model. 
    Required by the dynesty sampler.
    
    Parameters
    ----------
    u : ndarray
        Array of uniform random numbers between 0 and 1.
        
    Returns
    -------
    priors : ndarray
        Transformed random numbers giving prior ranges on model parameters.
    """
    a       = jeffreys(1., 1e4, u[0])
    mu_a    = uniform(0.2, 0.4, u[1])
    sigma_a = jeffreys(0.001, 3., u[2])
    skew_a  = uniform(-30., 30., u[3])
    b       = jeffreys(1., 1e4, u[4])
    mu_b    = uniform(0.3, 1., u[5])
    sigma_b = jeffreys(0.001, 3., u[6])
    skew_b  = uniform(-20., 20., u[7])
    
    return a, mu_a, sigma_a, skew_a, b, mu_b, sigma_b, skew_b

################################################################################
################################################################################

def prior_xforSkewNorms3(u):
    """Priors for the 8 parameters of the skew normal mixture model. 
    Required by the dynesty sampler.
    
    Parameters
    ----------
    u : ndarray
        Array of uniform random numbers between 0 and 1.
        
    Returns
    -------
    priors : ndarray
        Transformed random numbers giving prior ranges on model parameters.
    """
    a       = jeffreys(1., 1e4, u[0])
    mu_a    = uniform(0.2, 0.4, u[1])
    sigma_a = jeffreys(0.001, 3., u[2])
    skew_a  = uniform(-20., 20., u[3])
    b       = jeffreys(1., 1e4, u[4])
    mu_b    = uniform(0.3, 1., u[5])
    sigma_b = jeffreys(0.001, 3., u[6])
    skew_b  = uniform(-20., 20., u[7])
    c       = jeffreys(1., 1e4, u[4])
    mu_c    = uniform(0.3, 1., u[5])
    sigma_c = jeffreys(0.001, 3., u[6])
    skew_c  = uniform(-20., 20., u[7])
    
    return a, mu_a, sigma_a, skew_a, b, mu_b, sigma_b, skew_b, c, mu_c, sigma_c, skew_c

################################################################################
################################################################################

def skewNorms2_fit(x_data, y_data, param_bounds):
    '''
    Fit skew normal mixture model to data.


    PARAMETERS
    ==========

    x_data : ndarray
        Array-like object containing x-data to be fit

    y_data : ndarray
        Array-like object containing y-data to be fit.  Same length as x_data.

    param_bounds : ndarray of shape (N,2)
        Array-like object containing N parameter bounds.


    RETURNS
    =======

    bestfit : scipy.optimize.minimize object
        Contains all information corresponding to the best fit of the skew 
        normal mixture model to the data.  If value is None, then no fit 
        converged.
    '''

    bestfit = None

    ############################################################################
    # Generate 30 random seeds for the minimizer.
    # Store the result with the lowest -ln(L) in bestfit.
    #---------------------------------------------------------------------------
    for i in range(30):
        p0 = [np.random.uniform(b[0], b[1]) for b in param_bounds]
        result = minimize(nlogL_skewNorms2, 
                          p0, 
                          method='L-BFGS-B', 
                          args=(y_data, x_data), 
                          bounds=param_bounds)
        
        if result.success:
            if bestfit is None:
                bestfit = result
            else:
                if result.fun < bestfit.fun:
                    bestfit = result
    ############################################################################

    return bestfit

################################################################################
################################################################################

def skewNorms3_fit(x_data, y_data, param_bounds):
    '''
    Fit skew normal mixture model to data.


    PARAMETERS
    ==========

    x_data : ndarray
        Array-like object containing x-data to be fit

    y_data : ndarray
        Array-like object containing y-data to be fit.  Same length as x_data.

    param_bounds : ndarray of shape (N,2)
        Array-like object containing N parameter bounds.


    RETURNS
    =======

    bestfit : scipy.optimize.minimize object
        Contains all information corresponding to the best fit of the skew 
        normal mixture model to the data.  If value is None, then no fit 
        converged.
    '''

    bestfit = None

    ############################################################################
    # Generate 30 random seeds for the minimizer.
    # Store the result with the lowest -ln(L) in bestfit.
    #---------------------------------------------------------------------------
    for i in range(30):
        p0 = [np.random.uniform(b[0], b[1]) for b in param_bounds]
        result = minimize(nlogL_skewNorms3, 
                          p0, 
                          method='L-BFGS-B', 
                          args=(y_data, x_data), 
                          bounds=param_bounds)
        
        if result.success:
            if bestfit is None:
                bestfit = result
            else:
                if result.fun < bestfit.fun:
                    bestfit = result
    ############################################################################

    return bestfit