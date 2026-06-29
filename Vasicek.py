#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import pandas as pd
from scipy.optimize import minimize
from Interest_rate_base import InterestRateModel
from SDE_discretization import EulerMaruyama, Milstein


# In[2]:


class Vasicek(InterestRateModel):
    """
    Class for Vasicek interest rate model of the form
    dr(t) = (theta-alpha*r(t))dt + sigma dW_t

    Parameters
    ----------
    theta: float
    alpha: float
    sigma: float
    """
    def __init__(self, theta, alpha, sigma):
        self.theta = theta
        self.alpha = alpha
        self.sigma = sigma

    def simulation(self, method, T, N, r0):
        """
        Simulates the path of the discretized process

        Parameters
        ----------
        method: str
            method of discretizing SDE to be used 
            either "euler-maruyama" or "milstein"
        T: int
            time horizon of the simulated process in days
        N: int
            number of steps to be used in the simulation
        r0: float
            initial rate

        Returns
        -------
        tuple of numpy.array with time points and numpy.array of simulated process values
        """
        if method == "euler-maruyama":
            sim = EulerMaruyama(lambda x, y: self.theta-self.alpha*x, lambda x, y: self.sigma, T, N, r0)
        if method == "milstein":
            sim = Milstein(lambda x, y: self.theta-self.alpha*x, lambda x, y: self.sigma, T, N, r0)
        return sim.simulation()

    def calibrate(self, data):
        """
        Calibrates the model parameters to data

        Parameters
        ----------
        data: pandas.core.frame.DataFrame
            data to which the model should be calibrated
        """
        def vol(expiry, tenor, a, sigma):
            term1 = np.sqrt( (1.0 - np.exp(-2.0 * a * expiry)) / (2.0 * a * expiry) )
            term2 = ( (1.0 - np.exp(-a * tenor)) / (a * tenor) )
            return sigma * term1 * term2
        def objective(params, data):
            a, sigma = params
            model_vols = vol(data["expiry_years"].values, data["swap_tenor_years"].values, a, sigma)
            market_vols = data["market_vol"].values
            err = model_vols - market_vols
            return np.sum(err**2)
        result = minimize(objective, x0=[0.03, 0.01], args=(data,), bounds=[(0.001, 1.0), (0.0001, 0.10)])
        a_hat, sigma_hat = result.x
        self.theta = -np.log(data['discount_factor_to_expiry'][0])
        self.alpha = a_hat
        self.sigma = sigma_hat


# In[ ]:




