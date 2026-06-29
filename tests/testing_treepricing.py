#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from src.CRR_model import CRRModel, Tree, Option


# In[2]:


option_call = Option(100, "call", "european")
steps = 1
price_call = -1
price_next = 0
eps = 0.002
while abs(price_call-price_next)>eps:
    model = CRRModel(100, 0.2, 0.05, 1, steps, up=None, down=None, kids=2, lista=[])
    price_call = price_next
    price_next = option_call.price(model)
    steps = steps+105
    if steps>5000:
        steps = "Pricing reached 5000 steps"
        break
print(f"Option 'call' price: {price_call}, Steps to convergence: {steps}")

option_put = Option(100, "put", "european")
steps = 1
price_put = -1
price_next = 0
while abs(price_put-price_next)>eps:
    model = CRRModel(100, 0.2, 0.05, 1, steps, up=None, down=None, kids=2, lista=[])
    price_put = price_next
    price_next = option_put.price(model)
    steps = steps+105
    if steps>5000:
        steps = "Pricing reached 5000 steps"
        break
print(f"Option 'put' price: {price_put}, Steps to convergence: {steps}")

print(f"Put-call parity according to Black-Scholes: {100-100*np.exp(-0.05)}")
print(f"Put-call parity according to our model: {price_call-price_put}")


# In[7]:


prices_call = []
prices_put = []
steps = []
i = 75
while i<2500:
    model = CRRModel(100, 0.2, 0.05, 1, i, up=None, down=None, kids=2, lista=[])
    prices_call.append(option_call.price(model))
    prices_put.append(option_put.price(model))
    steps.append(i)
    i = i+15
f, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].plot(steps, prices_call, color="blue")
ax[0].set_title("Call price")
ax[0].set_xlabel("tree pricing iterations")
ax[0].set_ylabel("price")
ax[1].plot(steps, prices_put, color="orange")
ax[1].set_title("Put price")
ax[1].set_xlabel("tree pricing iterations")
ax[1].set_ylabel("price")
plt.show()


# In[ ]:




