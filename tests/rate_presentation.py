#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import pandas as pd
from scipy.optimize import minimize
from Interest_rate_base import InterestRateModel
from SDE_discretization import EulerMaruyama, Milstein
from Vasicek import Vasicek
from CIR import CIR
from Hull_White import HullWhite
from Ho_Lee import HoLee


# In[4]:


df = pd.read_csv("../eur_swaption_dataset(Swaption_Quotes).csv", sep=";", decimal=",")
df = df[(df['quote_quality'] == 'OK') & (df['moneyness_bps'] == 0)]
df = df.reset_index(drop=True)
df["market_vol"] = df["normal_vol_mid_bps"] / 10000.0


# In[5]:


#dummy functions to initiate models and to be replaced in data calibration
def theta(x):
    return -np.log(0.998)/(x+1)
def sigma(x):
    return 1


# In[24]:


vasicek = Vasicek(1,1,1)
vasicek.calibrate(df)
plt.title("Vasicek rates")
vasicek.paths_plot("euler-maruyama", 500, 500*4, 0.02, 10)


# In[25]:


cir = CIR(vasicek.theta, vasicek.alpha, vasicek.sigma)
plt.title("CIR rates with Vasicek parameters")
cir.paths_plot("euler-maruyama", 500, 500*4, 0.02, 10)


# In[19]:


hullwhite = HullWhite(theta, 1, sigma)
hullwhite.calibrate(df)
plt.title("Hull-White rates with Euler-Maruyama")
hullwhite.paths_plot("euler-maruyama", 500, 4*500, 0.02, 15)
plt.title("Hull-White rates with Milstein")
hullwhite.paths_plot("milstein", 500, 4*500, 0.02, 15)


# In[23]:


holee = HoLee(theta, sigma)
holee.calibrate(df)
plt.title("Ho-Lee rates with Euler-Maruyama")
holee.paths_plot("euler-maruyama", 500, 4*500, 0.02, 25)
plt.title("Ho-Lee rates with Milstein")
holee.paths_plot("milstein", 500, 4*500, 0.02, 25)


# In[ ]:




