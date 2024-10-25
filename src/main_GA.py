from softVoronoi import generate_voronoi
from linear_fem import linear_fem
import jax.numpy as np
import os
import numpy as onp

if __name__=='__main__':
    # Specify mesh-related information (second-order tetrahedron element).
    Lx, Ly = 60., 30.
    Nx,Ny=60,30
    coordinates = np.indices((Nx, Ny))

    onp.random.seed(0)
    sites_num=50
    dim=2
    def generate_points(Nx, Ny, n):
        # uniform points in 0,Nx 0,Ny
        x = onp.random.uniform(0, Nx, n)
        y = onp.random.uniform(0, Ny, n)
        return np.column_stack((x, y))

    sites = generate_points(Nx, Ny, sites_num)
    optimizationParams = {"coordinates":coordinates,"sites_num":sites_num,"Dm_dim":dim}
    Dm = np.tile(np.array(([1, 0], [0, 1])), (sites.shape[0], 1, 1))  # Nc*dim*dim
    cauchy_points=sites.copy()
    p= np.concatenate((np.ravel(sites),np.ravel(Dm),np.ravel(cauchy_points)),axis=0)# 1-d array contains flattened: sites,Dm,cauchy points


    linear_fem(Nx,Ny,Lx,Ly,optimizationParams,p,"u")

