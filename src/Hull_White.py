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


class HullWhite(InterestRateModel):
    """
    Class for Hull_white interest rate model of the form
    dr(t) = (theta(t)-alpha*r(t))dt + sigma(t)dW_t

    Parameters
    ----------
    theta: callable
        function of time
    alpha: float
    sigma: callable
        function of time
    """
    def __init__(self, theta, alpha, sigma, dsigma = lambda t: 0):
        self.theta = theta
        self.alpha = alpha
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
            sim = EulerMaruyama(lambda x, y: self.theta(y)-self.alpha*x, lambda x, y: self.sigma(y), T, N, r0)
        if method == "milstein":
            sim = Milstein(lambda x, y: self.theta(y)-self.alpha*x, lambda x, y: self.sigma(y), T, N, r0, lambda x, y: self.dsigma(y))
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
        n_expiries = len(expiries)
        expiry_to_idx = {e: i for i, e in enumerate(expiries)}
        def vol(expiry, tenor, a, sigma):
            term1 = np.sqrt((1.0 - np.exp(-2.0 * a * expiry)) / (2.0 * a * expiry))
            term2 = (1.0 - np.exp(-a * tenor)) / (a * tenor)
            return sigma * term1 * term2
        def objective(params, data):
            a = params[0]
            sigmas = params[1:]
            expiry = data["expiry_years"].values
            tenor = data["swap_tenor_years"].values
            sigma_vec = np.array([sigmas[expiry_to_idx[e]] for e in expiry])
            model_vols = vol(expiry, tenor, a, sigma_vec)
            market_vols = data["market_vol"].values
            err = model_vols - market_vols
            return np.sum(err**2)
        x0 = np.concatenate([[0.03], np.full(n_expiries, 0.01)])
        bounds = [(0.001, 1.0)] + [(0.0001, 0.10)] * n_expiries
        result = minimize(objective, x0=x0, args=(data,), bounds=bounds)
        self.alpha = result.x[0]
        sigma_hats = result.x[1:]
        def sig(x):
            if x<=expiries[0]*252:
                return sigma_hats[0]
            if x>expiries[-1]*252:
                return sigma_hats[-1]
            for i in range(len(expiries)):
                if expiries[i]*252<x<=expiries[i+1]*252:
                    A = (sigma_hats[i+1]-sigma_hats[i])/(expiries[i+1]*252-expiries[i]*252)
                    B = sigma_hats[i]-A*expiries[i]*252
                    return A*x+B
        self.sigma = sig
        def dsig(x):
            if x<=expiries[0]*252:
                return 0
            if x>expiries[-1]*252:
                return 0
            for i in range(len(expiries)):
                if expiries[i]*252<x<=expiries[i+1]*252:
                    return (sigma_hats[i+1]-sigma_hats[i])/(expiries[i+1]*252-expiries[i]*252)
        self.dsigma = dsig


# In[ ]:




