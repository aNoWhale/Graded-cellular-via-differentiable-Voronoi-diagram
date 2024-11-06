import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-10,10,1000)
def normal_distribution(x,mu=0.,sigma=1.):
    return 1./(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))
sigma=1./4
mu=1
k=1/normal_distribution(1,mu,sigma)
print(k)
y=normal_distribution(x,mu=mu,sigma=sigma)*k
plt.plot(x,y)
plt.show()