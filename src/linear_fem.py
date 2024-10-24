# Import some useful modules.
import glob

import jax.numpy as np
import os
import numpy as onp
import jax
import glob
import matplotlib.pyplot as plt

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_fem import logger

import logging

from generate_mesh import rectangle_mesh

logger.setLevel(logging.DEBUG)



# Weak forms.
class LinearElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM
    # solves -div(f(u_grad)) = b. Here, we have f(u_grad) = sigma.
    def custom_init(self):
        # Override base class method.
        # Set up 'self.fe.flex_inds' so that location-specific TO can be realized.
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))

    def get_tensor_map(self):
        def stress(u_grad, theta):
            # Plane stress assumption
            # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
            Emax = 70.e3
            Emin = 1e-3 * Emax
            nu = 0.3
            penal = 3.
            E = Emin + (Emax - Emin) * theta[0] ** penal
            epsilon = 0.5 * (u_grad + u_grad.T)
            eps11 = epsilon[0, 0]
            eps22 = epsilon[1, 1]
            eps12 = epsilon[0, 1]
            sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
            sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
            sig12 = E / (1 + nu) * eps12
            sigma = np.array([[sig11, sig12], [sig12, sig22]])
            return sigma

        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 100.])

        return [surface_map]

    def set_voronoi(self,op,p):
        thetas = generate_voronoi(op, p)
        thetas = thetas.reshape(-1, 1)
        theta = np.repeat(thetas[:, None, :], self.fe.num_quads, axis=1)
        self.internal_vars=[theta]
        return theta

# Specify mesh-related information (second-order tetrahedron element).
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 60., 30.
Nx,Ny=60,30
meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
from softVoronoi import generate_voronoi
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
optimizationParams = {'maxIters': 51, 'movelimit': 10.,"coordinates":coordinates,"sites_num":sites_num,"Dm_dim":dim}
Dm = np.tile(np.array(([1, 0], [0, 1])), (sites.shape[0], 1, 1))  # Nc*dim*dim
cauchy_points=sites.copy()
p= np.concatenate((np.ravel(sites),np.ravel(Dm),np.ravel(cauchy_points)),axis=0)# 1-d array contains flattened: sites,Dm,cauchy points




# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)


def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)


# Define Dirichlet boundary values.
# This means on the 'left' side, we apply the function 'zero_dirichlet_val'
# to all components of the displacement variable u.
def zero_dirichlet_val(point):
    return 0.


dirichlet_bc_info = [[left] * 2, [0, 1], [zero_dirichlet_val] * 2]

# Define Neumann boundary locations.
# This means on the 'right' side, we will perform the surface integral to get
# the tractions with the function 'get_surface_maps' defined in the class 'LinearElasticity'.
location_fns = [right]

# Create an instance of the problem.
problem = LinearElasticity(mesh,
                           vec=2,
                           dim=2,
                           ele_type=ele_type,
                           dirichlet_bc_info=dirichlet_bc_info,
                           location_fns=location_fns)
thetas=problem.set_voronoi(optimizationParams,p)

# Solve the defined problem.
sol_list = solver(problem, solver_options={'umfpack_solver': {}})

# Postprocess for stress evaluations
# Do some cleaning work. Remove old solution files.
data_path = os.path.join(os.path.dirname(__file__), 'data')
files = glob.glob(os.path.join(data_path, f'vtk/*'))
for f in files:
    os.remove(f)

# (num_cells, num_quads, vec, dim)
u_grad = problem.fes[0].sol_to_grad(sol_list[0])
epsilon = 0.5 * (u_grad + u_grad.transpose(0, 1, 3, 2))
# (num_cells, bnum_quads, 1, 1) * (num_cells, num_quads, vec, dim)
# -> (num_cells, num_quads, vec, dim)
Emax = 70.e3
Emin = 1e-3 * Emax
nu = 0.3
penal = 3.
E = (Emin + (Emax - Emin) * thetas ** penal)
eps11 = epsilon[:,:,0, 0,None]
eps22 = epsilon[:,:,1, 1,None]
eps12 = epsilon[:,:,0, 1,None]
sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
sig12 = E / (1 + nu) * eps12
sigma_1 = np.concatenate([sig11, sig12], axis=-1)  # (num_cells, num_quads, 2)
sigma_2 = np.concatenate([sig12, sig22], axis=-1)  # (num_cells, num_quads, 2)

# 然后再沿另一个新轴拼接成 2x2 矩阵
sigma = np.concatenate([sigma_1[:, :, None, :], sigma_2[:, :, None, :]], axis=2)


# (num_cells, num_quads)
cells_JxW = problem.JxW[:, 0, :]
# (num_cells, num_quads, vec, dim) * (num_cells, num_quads, 1, 1) ->
# (num_cells, vec, dim) / (num_cells, 1, 1)
#  --> (num_cells, vec, dim)
sigma_average = np.sum(sigma * cells_JxW[:, :,None,None], axis=1) / np.sum(cells_JxW, axis=1)[:,None,None]

# Von Mises stress
# (num_cells, dim, dim)
s_dev = (sigma_average - 1 / problem.dim * np.trace(sigma_average, axis1=1, axis2=2)[:, None, None]
         * np.eye(problem.dim)[None, :, :])
# (num_cells,)
vm_stress = np.sqrt(3. / 2. * np.sum(s_dev * s_dev, axis=(1, 2)))
thetas=np.mean(thetas.squeeze(-1), axis=-1)
# Store the solution to local file.
vtk_path = os.path.join(data_path, 'vtk/u.vtu')
save_sol(problem.fes[0], sol_list[0], vtk_path, cell_infos=[('vm_stress', vm_stress),("theta",thetas)])
