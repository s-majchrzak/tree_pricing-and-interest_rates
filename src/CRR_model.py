#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


class CRRModel:
    """
    Class for the generalized CRR pricing model capable of generating the extension 
    of binomial and trinomial trees

    Parameters
    ----------
    price: float
        spot price of the stock
    sigma: float
        volatility of the stock
        used only to generate the default move factors
    rate: float
        risk free interest rate
    maturity: int
        maturity of the option to be price
    steps: int
        number of time steps to be used in the pricing
    up: float
        the up factor of the stock move
        To be set as exp(sigma*sqrt(delta_t)) if None
    down: float
        the down factor of the stock move
        To be set as 1/up if None
    kids: int
        number of children that each tree node produces in the model
        2 by default
    lista: list
        list of kids-2 probabilities assigned to the first kids-2 "up" moves
        empty by default to be set as matching the first two moments of the 
        geometric Brownian motion with the first kids-3 values equally probable
    """
    def __init__(self, price, sigma, rate, maturity, steps, up=None, down=None, kids=2, lista=[]):
        self.price = price
        self.sigma = sigma
        self.rate = rate
        self.maturity = maturity
        self.steps = steps
        self.delta_t = maturity/steps
        if up == None:
            up = np.exp(sigma*np.sqrt(self.delta_t))
        if down == None:
            down = 1/up
        self.up = up
        self.down = down
        self.kids = kids
        self.means = self._proportional_means(up, down, kids)
        if lista == []:
            if kids > 2:
                means = self.means
                prob = [1/kids for i in range(kids-3)]
                nominator1 = np.exp((2*rate+sigma**2)*self.delta_t) - down**2 - (np.exp(rate*self.delta_t)-down)*(means[-2]+down)
                nominator2 = sum([prob[i]*(means[i]**2-down**2- (means[i]-down)*(means[-2]+down)) for i in range(len(prob))])
                denominator = means[-3]**2 - down**2 - (means[-3]-down)*(means[-2]+down)
                prob.append((nominator1-nominator2)/denominator)
                lista = prob
        self.lista = lista

    def _proportional_means(self, u, d, n):
        """
        Calculates the proportional means, so that tree can recombine

        Parameters
        ----------
        u: float
            up factor
        d: float
            down factor
        n: int
            number of means to be determined

        Returns
        -------
        list
            list of proportional means including up and down factors
        """
        r = (d/u)**(1/(n-1))
        sequence = [u * r**i for i in range(n)]
        return sequence

    def _build_tree(self):
        """
        Builds the tree
        """
        return Tree(self.price, self.rate, self.delta_t, self.steps, self.means, self.kids)

class Tree:
    """
    Class for the general pricing tree (n-nomial) capable of calculating risk neutral measures

    Parameters
    ----------
    price: float
        spot price of the stock
    rate: float
        interest rate (used in discounting)
    delta_t: float
        length of the time step
    steps: float
        number of generations of the tree that are to be built
    means: list
        list of proportional means of up and down factors
        the first term of the list is the up factor
        the second term of the list is the down factor
        necessary for the tree to recombine
    kids: int
        number of children that each node has
    """
    def __init__(self, price, rate, delta_t, steps, means, kids):
        self.price = price
        self.rate = rate
        self.delta_t = delta_t
        self.steps = steps
        self.means = means
        self.kids = kids
        
    def generations(self):
        """
        Calculates the possible stock moves as the tree generations

        Returns
        -------
        list of generations (lists) of the possible stock value at each time step
        """
        generations = [[self.price]]
        n = self.steps + 1
        m = self.kids - 1
        vect = self.means
        for i in range(1,n):
            gen1 = [vect[0]*s for s in generations[i-1]]
            gen2 = [generations[i-1][-1]*v for v in vect[1:]]
            generations.append(gen1+gen2)
        return generations

    def probability(self, lista):
        """
        Calculates the risk neutral probability

        Parameters
        ----------
        lista: list
            list of n-2 probabilities assigned to the first n-2 "up" moves
            where n is the number of children each node has

        Returns
        -------
        list enlarged by the remaining two missing probabilities, so that the 
        whole array contains the risk neutral measure
        """
        values = self.means
        d = values[-1]
        r = self.rate
        delta_t = self.delta_t
        prob = lista
        if len(lista)>0:
            prob.append( (np.exp(r*delta_t) - d - sum(lista[i]*(values[i] - d) for i in range(len(lista)))) / (values[-2] - d) )
        else:
            prob.append( (np.exp(r*delta_t) - d) / (values[-2] - d) )
        prob.append(1 - sum(prob[i] for i in range(len(prob))))
        return prob

    def plot(self, npaths):
        """
        Plots the (distinct) random paths that a stock may attain in the tree

        Parameters
        ----------
        npaths: int
            number of paths to be plotted

        Returns
        -------
        plot with the desired number of paths
        """
        generations = self.generations()
        path_container = [0]
        for i in range(npaths):
            path = 0
            while path in path_container:
                path = [self.price]
                index = 0
                for j in range(self.steps):
                    index_next = np.random.choice(range(index,index+self.kids))
                    path.append(generations[j+1][index_next])
                    index = index_next
            path_container.append(path)
            plt.plot(range(self.steps+1), path, color="black", linewidth=1)
        plt.show()

class Option:
    """
    Class for the option

    Parameters
    ----------
    strike: float
        strike of the option
    kind: "call" or "put"
        the kind of the option
    style: "european" or "american"
        style of the option
    """
    def __init__(self, strike, kind, style):
        self.strike = strike
        self.kind = kind
        self.style = style

    def intrinsic_value(self, spot):
        """
        the payoff function to be used in the pricing

        Parameters
        ----------
        spot: float
            the spot value of the stock

        Returns
        -------
        float
            the payoff at given spot
        """
        if self.kind == "call":
            return max(spot - self.strike, 0)
        return max(self.strike - spot, 0)

    def price(self, model):
        """
        Prices the option

        Parameters
        ----------
        model: CRRModel
            model to be used in the pricing

        Returns
        -------
        price of the option
        """
        tree = model._build_tree()
        r = model.rate
        delta_t = model.delta_t
        gen = tree.generations()
        q = tree.probability(model.lista)
        n = len(gen)-1
        m = model.kids
        values = [self.intrinsic_value(gen[n][j]) for j in range(len(gen[n]))]
        if self.style == "european":
            for i in range(model.steps):
                values_next = [np.exp(-r*delta_t)*(sum(q[k]*values[j+k] for k in range(m))) for j in range(len(gen[n-1]))]
                values = values_next
                n=n-1
        if self.style == "american":
            for i in range(model.steps):
                values_next = [max(self.intrinsic_value(gen[n-1][j]), np.exp(-r*delta_t)*(sum(q[k]*values[j+k] for k in range(m))) )
                              for j in range(len(gen[n-1]))]
                values = values_next
                n=n-1
        return values[0]


# In[ ]:




