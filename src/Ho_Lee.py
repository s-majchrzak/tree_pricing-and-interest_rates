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


class HoLee(InterestRateModel):
    """
    Class for Ho-Lee interest rate model of the form
    dr(t) = theta(t)dt + sigma(t)dW_t

    Parameters
    ----------
    theta: callable
        function of time
    sigma: callable
        function of time
    """
    def __init__(self, theta, sigma, dsigma = lambda t: 0):
        self.theta = theta
        self.sigma = sigma
        self.dsigma = dsigma

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
            sim = EulerMaruyama(lambda x, y: self.theta(y), lambda x, y: self.sigma(y), T, N, r0)
        if method == "milstein":
            sim = Milstein(lambda x, y: self.theta(y), lambda x, y: self.sigma(y), T, N, r0, lambda x, y: self.dsigma(y))
        return sim.simulation()

    def calibrate(self, data):
        """
        Calibrates the model parameters to data

        Parameters
        ----------
        data: pandas.core.frame.DataFrame
            data to which the model should be calibrated
        """
        expiries = np.sort(data["expiry_years"].unique())
        def vol(expiry, sigma):
            return sigma * np.sqrt(expiry)
        def objective(sigmas, data, expiries):
            market_vols = data["market_vol"].values
            sigma_map = dict(zip(expiries, sigmas))
            row_sigmas = data["expiry_years"].map(sigma_map).values
            model_vols = vol(data["expiry_years"].values, row_sigmas)
            return np.sum((model_vols - market_vols) ** 2)
        x0 = np.full(len(expiries), 0.01)
        bounds = [(0.0001, 0.10)] * len(expiries)
        res = minimize(objective, x0=x0, args=(data, expiries), bounds=bounds)
        def sig(x):
            if x<=expiries[0]*252:
                return res.x[0]
            if x>expiries[-1]*252:
                return res.x[-1]
            for i in range(len(expiries)-1):
                if expiries[i]*252<x<=expiries[i+1]*252:
                    A = (res.x[i+1]-res.x[i])/(expiries[i+1]*252-expiries[i]*252)
                    B = res.x[i]-A*expiries[i]*252
                    return A*x+B
        self.sigma = sig
        def dsig(x):
            if x<=expiries[0]*252:
                return 0
            if x>expiries[-1]*252:
                return 0
            for i in range(len(expiries)-1):
                if expiries[i]*252<x<=expiries[i+1]*252:
                    return (res.x[i+1]-res.x[i])/(expiries[i+1]*252-expiries[i]*252)
        self.dsigma = dsig


# In[ ]:




