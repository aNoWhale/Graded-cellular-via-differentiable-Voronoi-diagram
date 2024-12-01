import jax.numpy as np


ones=np.ones((2,2))

print(ones.at[-2:,:].set(ones[-2:,:]+1))




