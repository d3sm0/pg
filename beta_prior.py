import matplotlib.pyplot as plt
import numpy as np
import pdb

def beta(x, alpha=2, beta=2):
    out = x**(alpha-1) *(1-x)**(beta-1)
    return out/out.sum()

def softmax(x):
    return np.exp(x)/np.exp(x).sum()


def gaussian_kernel(x, gamma=1):
    x = np.expand_dims(x,-1)
    d = np.linspace(-2., 2., 10).reshape((1,-1))
    return np.exp(-gamma* (x-d)**2).sum(1)

def beta_kernel(x, alpha=2, beta=2, b=1):
    x = np.expand_dims(x,-1)
    alpha = x/b +  1
    beta = (1-x)/b + 1
    d = np.linspace(0., 1., 10).reshape((1,-1))
    out = (d)**(alpha-1)*(1-d)**(beta-1)
    return out.sum(1)
    #return out/out.sum()
    #return np.exp(-gamma* (x-d)**2).sum(1)

x = np.linspace(-4., 4.)


fig, ax  = plt.subplots(1,1)
# ax.plot(x,gaussian_kernel(x))
ax.plot(x,beta_kernel(x))
#ax.plot(x,softmax(x))
plt.show(block=False)
plt.pause(5)
plt.close()
