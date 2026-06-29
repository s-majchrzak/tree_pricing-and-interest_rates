#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from src.CRR_model import CRRModel, Tree, Option


# In[2]:


#By default model uses binomial tree, calculates up and down factors and assignes probabilities
model1 = CRRModel(1, 0.2, 0.05, 1, 3)
print(f"Up factor = {model1.up}, Down factor = {model1.down}")
tree1 = model1._build_tree()
print(f"Risk neutral probability: {tree1.probability(model1.lista)}")
generations1 = tree1.generations()
for i in range(len(generations1)):
    print(f"Generation {i}: {generations1[i]}")
tree1.plot(8)


# In[3]:


#We can customize up and down factors with u*d not necessarily equal 1
#We can use larger than trinomial trees and assign probabilities
model2 = CRRModel(1, 0.2, 0.05, 1, 2, up=1.2, down=0.9, kids=4, lista=[0.2, 0.2])
print(f"Up, mid1, mid2, down factors: {model2.means}")
tree2 = model2._build_tree()
print(f"Risk neutral probability: {tree2.probability(model2.lista)}")
generations2 = tree2.generations()
for i in range(len(generations2)):
    print(f"Generation {i}: {generations2[i]}")
tree2.plot(10)


# In[7]:


#Warning
#The model may yield incoherent values when the provided sigma doesn't produce the risk neutral probability
model3 = CRRModel(1, 0.02, 0.05, 1, 3, up=None, down=None, kids=3, lista=[])
print(f"Up, mid, down factors: {model3.means}")
tree3 = model3._build_tree()
print(f"This is not a probability!: {tree3.probability(model3.lista)}")

#or if the provided probability values do not correspond to a risk neutral measure
model4 = CRRModel(1, 0.2, 0.05, 1, 3, up=1.4, down=0.9, kids=3, lista=[0.9])
print(f"Up, mid, down: {model4.means}")
tree4 = model4._build_tree()
print(f"This is not a probability!: {tree4.probability(model4.lista)}")


# In[4]:


#Option pricing
option1 = Option(1, "call", "european")
print(option1.price(model1))
option2 = Option(1, "put", "american")
print(option2.price(model2))


# In[ ]:




