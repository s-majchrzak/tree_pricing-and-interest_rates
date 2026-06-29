#!/usr/bin/env python
# coding: utf-8

# In[1]:


from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


class InterestRateModel(ABC):
    """
    Base class for interest rate models
    """
    @abstractmethod
    def simulation(self, method, T, N, r0):
        pass

    def paths_plot(self, method, T, N, r0, paths_number):
        """
        Plots the paths of a given number of simulations

        Parameters
        ----------
        paths_number: int
            number of simulations to be ploted
        """
        for i in range(paths_number):
            plt.plot(*self.simulation(method, T, N, r0), linewidth=1)
        plt.xlabel("time")
        plt.show()


# In[ ]:




