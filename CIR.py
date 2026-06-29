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


class CIR(InterestRateModel):
    """
    Class for CIR interest rate model of the form
    dr(t) = (theta-alpha*r(t))dt + sigma\sqrt(r(t))dW_t

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
            sim = EulerMaruyama(lambda x, y: self.theta-self.alpha*x, lambda x, y: self.sigma*np.sqrt(x), T, N, r0)
        if method == "milstein":
            sim = Milstein(lambda x, y: self.theta-self.alpha*x, lambda x, y: self.sigma*np.sqrt(x), T, N, r0)
        return sim.simulation()


# In[ ]:




