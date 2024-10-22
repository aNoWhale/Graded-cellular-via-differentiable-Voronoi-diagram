"""Solve a 3D linear elasticity problem with one variable (vec=3)
This is to verify that the implementation of multi-variable problem is correct.
Compare the results with jax-fem/applications/stokes/example_2vars.py
"""
import jax
import jax.numpy as np
import jax.flatten_util
import os

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh
from jax_fem.utils import save_sol
from jax_fem.problem import Problem


class LinearElasticity(Problem):

    def get_universal_kernel(self):

        def stress(u_grad):
            E = 70e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma

        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_shape_grads = cell_shape_grads[:, :self.fes[0].num_nodes, :]
            cell_sol = cell_sol_list[0]
            cell_JxW = cell_JxW[0]
            cell_v_grads_JxW = cell_v_grads_JxW[:, :self.fes[0].num_nodes, :, :]

            vec = self.fes[0].vec
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, vec, self.dim)  # (num_quads, vec, dim)
            # (num_quads, vec, dim)
            u_physics = jax.vmap(stress)(u_grads_reshape, *cell_internal_vars).reshape(u_grads.shape)
            # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))

            return jax.flatten_util.ravel_pytree(val)[0]

        return universal_kernel


def problem():
    """Can be used to test the memory limit of JAX-FEM
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    ele_type = 'HEX8'
    meshio_mesh = box_mesh_gmsh(2, 1, 1, 1., 1., 1., data_dir, ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def dirichlet_val(point):
        return 0.1

    dirichlet_bc_info = [[left, left, left, right, right, right], 
                         [0, 1, 2, 0, 1, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                          dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]
 
    problem = LinearElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)

    sol_list = solver(problem)
    vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
    save_sol(problem.fes[0], sol_list[0], vtk_path)

 
if __name__ == "__main__":
    problem()