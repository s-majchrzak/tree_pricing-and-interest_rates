#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import pandas as pd
from scipy.optimize import minimize
from src.Interest_rate_base import InterestRateModel
from src.SDE_discretization import EulerMaruyama, Milstein
from src.Vasicek import Vasicek
from src.CIR import CIR
from src.Hull_White import HullWhite
from src.Ho_Lee import HoLee


# In[2]:


df = pd.read_csv("../eur_swaption_dataset(Swaption_Quotes).csv", sep=";", decimal=",")
df = df[(df['quote_quality'] == 'OK') & (df['moneyness_bps'] == 0)]
df = df.reset_index(drop=True)
df["market_vol"] = df["normal_vol_mid_bps"] / 10000.0


# In[3]:


#dummy functions to initiate models and to be replaced in data calibration
def theta(x):
    return -np.log(0.998)/(x+1)
def sigma(x):
    return 1


# In[4]:


vasicek = Vasicek(1,1,1)
vasicek.calibrate(df)
sims = []
means = np.zeros(4*500+1)-1
means_next = np.zeros(4*500+1)
eps = 0.001
sims_number = 0
while max([means_next[i] - means[i] for i in range(len(means))])>eps:
    for j in range(100):
        times, path = vasicek.simulation("euler-maruyama", 500, 4*500, 0.02)
        sims.append(path)
    means = means_next
    means_next = [np.mean([path[j] for path in sims]) for j in range(4*500+1)]
    plt.plot(times, means_next, linewidth=1)
    sims_number = sims_number + 100
print(f"Number of simulations to achieve <{eps} path accuracy: {sims_number}")
plt.title("Vasicek rate average path each 100 of simulations")
plt.show()


# In[5]:


cir = CIR(vasicek.theta, vasicek.alpha, vasicek.sigma)
sims = []
means = np.zeros(4*500+1)-1
means_next = np.zeros(4*500+1)
eps = 0.001
sims_number = 0
while max([means_next[i] - means[i] for i in range(len(means))])>eps:
    for j in range(100):
        times, path = cir.simulation("euler-maruyama", 500, 4*500, 0.02)
        sims.append(path)
    means = means_next
    means_next = [np.mean([path[j] for path in sims]) for j in range(4*500+1)]
    plt.plot(times, means_next, linewidth=1)
    sims_number = sims_number + 100
print(f"Number of simulations to achieve <{eps} path accuracy: {sims_number}")
plt.title("CIR rate average path each 100 of simulations")
plt.show()


# In[6]:


holee = HoLee(theta, sigma)
holee.calibrate(df)
sims = []
means = np.zeros(4*500+1)-1
means_next = np.zeros(4*500+1)
eps = 0.0005
sims_number = 0
while max([means_next[i] - means[i] for i in range(len(means))])>eps:
    for j in range(100):
        times, path = holee.simulation("euler-maruyama", 500, 4*500, 0.02)
        sims.append(path)
    means = means_next
    means_next = [np.mean([path[j] for path in sims]) for j in range(4*500+1)]
    plt.plot(times, means_next, linewidth=1)
    sims_number = sims_number + 100
print(f"Number of simulations to achieve <{eps} path accuracy with Euler-Maruyama: {sims_number}")
plt.title("Ho-Lee rate average path each 100 of simulations with Euler-Maruyama")
plt.show()


# In[7]:


sims = []
means = np.zeros(4*500+1)-1
means_next = np.zeros(4*500+1)
sims_number = 0
while max([means_next[i] - means[i] for i in range(len(means))])>eps:
    for j in range(100):
        times, path = holee.simulation("milstein", 500, 4*500, 0.02)
        sims.append(path)
    means = means_next
    means_next = [np.mean([path[j] for path in sims]) for j in range(4*500+1)]
    plt.plot(times, means_next, linewidth=1)
    sims_number = sims_number + 100
print(f"Number of simulations to achieve <{eps} path accuracy with Milstein: {sims_number}")
plt.title("Ho-Lee rate average path each 100 of simulations with Milstein")
plt.show()


# In[10]:


hullwhite = HullWhite(theta, 1, sigma)
hullwhite.calibrate(df)
sims = []
means = np.zeros(4*500+1)-1
means_next = np.zeros(4*500+1)
eps = 0.0005
sims_number = 0
while max([means_next[i] - means[i] for i in range(len(means))])>eps:
    for j in range(100):
        times, path = hullwhite.simulation("euler-maruyama", 500, 4*500, 0.02)
        sims.append(path)
    means = means_next
    means_next = [np.mean([path[j] for path in sims]) for j in range(4*500+1)]
    plt.plot(times, means_next, linewidth=1)
    sims_number = sims_number + 100
print(f"Number of simulations to achieve <{eps} path accuracy with Euler-Maruyama: {sims_number}")
plt.title("Hull-White rate average path each 100 of simulations with Euler-Maruyama")
plt.show()


# In[11]:


sims = []
means = np.zeros(4*500+1)-1
means_next = np.zeros(4*500+1)
sims_number = 0
while max([means_next[i] - means[i] for i in range(len(means))])>eps:
    for j in range(100):
        times, path = hullwhite.simulation("milstein", 500, 4*500, 0.02)
        sims.append(path)
    means = means_next
    means_next = [np.mean([path[j] for path in sims]) for j in range(4*500+1)]
    plt.plot(times, means_next, linewidth=1)
    sims_number = sims_number + 100
print(f"Number of simulations to achieve <{eps} path accuracy with Milstein: {sims_number}")
plt.title("Hull-White rate average path each 100 of simulations with Milstein")
plt.show()


# In[ ]:




