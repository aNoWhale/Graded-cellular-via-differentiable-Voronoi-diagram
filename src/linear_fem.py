# Import some useful modules.
import glob
from typing import Callable

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
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh,rectangle_mesh
from jax_fem import logger
import tqdm as tqdm
import logging

from softVoronoi import generate_voronoi, generate_gene_random


def linear_fem(Nx,Ny,Lx,Ly,optimizationParams,p,filename,load:np.array,load_location:Callable,fixed_location:Callable):
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
                return load

            return [surface_map]

        def set_voronoi(self, op, p):
            thetas = generate_voronoi(op, p)
            thetas = thetas.reshape(-1, 1)
            theta = np.repeat(thetas[:, None, :], self.fe.num_quads, axis=1)
            self.internal_vars = [theta]
            return thetas

        def compute_compliance(self, sol):
            # Surface integral
            boundary_inds = self.boundary_inds_list[0]
            _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
            # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)
            u_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :] * self.fe.face_shape_vals[
                                                                                  boundary_inds[:, 1]][
                                                                              :, :, :, None]
            u_face = np.sum(u_face, axis=2)  # (num_selected_faces, num_face_quads, vec)
            # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)

            # subset_quad_points = self.get_physical_surface_quad_points(boundary_inds)

            subset_quad_points = self.physical_surface_quad_points[0]

            neumann_fn = self.get_surface_maps()[0]
            traction = -jax.vmap(jax.vmap(neumann_fn))(u_face,
                                                       subset_quad_points)  # (num_selected_faces, num_face_quads, vec)
            val = np.sum(traction * u_face * nanson_scale[:, :, None])
            return val

    logger.setLevel(logging.DEBUG)
    # Specify mesh-related information (second-order tetrahedron element).
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # Define boundary locations.
    # def fixed_location(point):
    #     return np.isclose(point[0], 0., atol=1e-5)

    # def load_location(point):
    #     return np.isclose(point[0], Lx, atol=1e-5)

    def dirichlet_val(point):
        return 0.

    dirichlet_bc_info = [[fixed_location] * 2, [0, 1], [dirichlet_val] * 2]

    location_fns = [load_location]

    # Create an instance of the problem.
    problem = LinearElasticity(mesh,vec=2,dim=2,ele_type=ele_type,dirichlet_bc_info=dirichlet_bc_info,location_fns=location_fns)
    thetas=problem.set_voronoi(optimizationParams,p)

    # Solve the defined problem.
    sol_list = solver(problem, solver_options={'umfpack_solver': {}})

    # Postprocess for stress evaluations
    # Do some cleaning work. Remove old solution files.
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    files = glob.glob(os.path.join(data_path, f'vtk/*'))
    # for f in files:
    #     os.remove(f)

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
    eps11 = epsilon[:,:,0, 0]
    eps22 = epsilon[:,:,1, 1]
    eps12 = epsilon[:,:,0, 1]
    sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
    sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
    sig12 = E / (1 + nu) * eps12
    sigma = np.array([[sig11, sig12], [sig12, sig22]])
    sigma= np.transpose(sigma, (2, 3, 0, 1))
    # (num_cells, num_quads)
    cells_JxW = problem.JxW[:, 0, :]
    # (num_cells, num_quads, vec, dim) * (num_cells, num_quads, 1, 1) ->
    # (num_cells, vec, dim) / (num_cells, 1, 1)
    #  --> (num_cells, vec, dim)
    sigma_average = np.sum(sigma * cells_JxW[:, :,None,None], axis=1) / np.sum(cells_JxW, axis=1)[:,None,None]

    # Von Mises stress
    trace_sigma_avg = np.trace(sigma_average, axis1=1, axis2=2)  # (num_cells,)
    identity_matrix = np.eye(problem.dim)  # (dim, dim)
    s_dev = sigma_average - (1 / problem.dim) * trace_sigma_avg[:, None, None] * identity_matrix
    # (num_cells, dim, dim)
    # s_dev = (sigma_average - 1 / problem.dim * np.trace(sigma_average, axis1=1, axis2=2)* np.eye(problem.dim))
    # (num_cells,)
    vm_stress = np.sqrt(3. / 2. * np.sum(s_dev * s_dev, axis=(1, 2)))
    thetas=thetas.squeeze(-1)
    # compliance
    compliance=problem.compute_compliance(sol_list[0])

    # Store the solution to local file.
    vtk_path = os.path.join(data_path, f'vtk/{filename}.vtu')
    save_sol(problem.fes[0], sol_list[0], vtk_path, cell_infos=[('vm_stress', vm_stress),("theta",thetas)])
    tqdm.tqdm.write(f'{filename}.vtu done,compliance:{compliance}')
    return compliance

if __name__ == '__main__':
    Nx, Ny = 60, 30
    Lx, Ly = 60., 30.
    sites_num = 50
    dim = 2
    margin = 10
    coordinates = np.indices((Nx, Ny))
    optimizationParams = {"coordinates": coordinates, "sites_num": sites_num, "Dm_dim": dim, "margin": margin}
    p=generate_gene_random(optimizationParams,Nx,Ny)
    def load_location_x(point):
         return np.isclose(point[0], Lx, atol=1e-5)
    def fixed_location_x(point):
        return np.isclose(point[0], 0., atol=1e-5)
    val=linear_fem(Nx, Ny, Lx, Ly, optimizationParams, p, f"x_", load=np.array([100., 0.]),
               load_location=load_location_x, fixed_location=fixed_location_x)
    print(val)