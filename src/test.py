import numpy as np
import matplotlib.pyplot as plt
sites=np.random.randint(0,100,size=(10,2))
cauchy=np.random.randint(0,100,size=(10,2))

plt.scatter(sites[:,0],sites[:,1],marker="o")
plt.scatter(cauchy[:,0],cauchy[:,1],marker="+")


plt.plot([sites[:,0], cauchy[:,0]], [sites[:,1], cauchy[:,1]], 'b-')
plt.show()