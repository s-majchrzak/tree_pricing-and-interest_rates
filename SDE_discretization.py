#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import pandas as pd


# In[2]:


class DiscretizationMethod(ABC):
    """
    Base class for discretization methods of SDEs of the form
    dX_t = a(X_t,t)dt + b(X_t,t)dW_t

    Parameters
    ----------
    a: Callable[[float, float], float]
        A drift function of two variables: first being the state and second the time
    b: Callable[[float, float], float]
        A volatility function of two variables: first being the state and second the time
    T: float
        time horizon for the simulation of the discretized process
    N: int
        number of steps in the simulation
    y0: float
        the initial value of the process
    """
    
    def __init__(self, a, b, T, N, y0):
        self.a = a
        self.b = b
        self.T = T
        self.N = N
        self.y0 = y0
        self.dt = T/N

    @abstractmethod
    def simulation(self):
        pass

class EulerMaruyama(DiscretizationMethod):
    """
    Class for Euler-Maruyama method of discretizing SDEs
    """
    def simulation(self):
        """
        Simulates the path of the process with Euler-Maruyama algorithm

        Returns
        -------
        tuple consisting of
            numpy.array
                array with time points in the simulation
            numpy.array
                array with path of the simulated process
        """
        times = np.arange(0, self.T + self.dt, self.dt)
        path = np.zeros(self.N+1)
        path[0] = self.y0
        for i in range(self.N):
            dw = np.random.normal(loc=0.0, scale=np.sqrt(self.dt))
            path[i+1] = path[i] + self.a(path[i], times[i])*self.dt + self.b(path[i], times[i])*dw
        return times, path

class Milstein(DiscretizationMethod):
    """
    Class for Milstein method of discretizing SDEs

    Parametars
    ----------
    db: Callable[[float, float], float]
        the derivative of b function with respect to time
        equal 0 by default
    """
    def __init__(self, a, b, T, N, y0, db = lambda x, y: 0):
        super().__init__(a, b, T, N, y0)
        self.db = db

    def simulation(self):
        """
        Simulates the path of the process with Milstein algorithm
        
        Returns
        -------
        tuple consisting of
            numpy.array
                array with time points in the simulation
            numpy.array
            array with path of the simulated process
        """
        times = np.arange(0,self.T + self.dt, self.dt)
        path = np.zeros(self.N+1)
        path[0] = self.y0
        for i in range(self.N):
            dw = np.random.normal(loc=0.0, scale=np.sqrt(self.dt))
            s1 = self.a(path[i], times[i])*self.dt + self.b(path[i], times[i])*dw
            s2 = 0.5*self.b(path[i], times[i])*self.db(path[i], times[i])*(dw**2-self.dt)
            path[i+1] = path[i] + s1 + s2
        return times, path


# In[ ]:




