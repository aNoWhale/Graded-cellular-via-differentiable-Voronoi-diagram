# Import some useful modules.
import time

import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.mma import optimize
# Define constitutive relationship.
# Generally, JAX-FEM solves -div.(f(u_grad,alpha_1,alpha_2,...,alpha_N)) = b.
# Here, we have f(u_grad,alpha_1,alpha_2,...,alpha_N) = sigma(u_grad, theta),
# reflected by the function 'stress'. The functions 'custom_init'and 'set_params'
# override base class methods. In particular, set_params sets the design variable theta.

from softVoronoi import generate_voronoi
from src.softVoronoi import generate_voronoi_separate


class Elasticity(Problem):
    def custom_init(self):
        # Override base class method.
        # Set up 'self.fe.flex_inds' so that location-specific TO can be realized.
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))
        self.target = 0
    def get_tensor_map(self):
        def stress(u_grad, theta):
            # Plane stress assumption
            # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
            Emax = 70.e3
            Emin = 1e-5 * Emax
            nu = 0.3
            penal = 1.
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
            # load define
            return np.array([-200., 0.])

        return [surface_map]

    def set_params(self, params):
        # Override base class method.
        """edited from tianxu xue"""
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]
    def compute_compliance(self, sol):
        # Surface integral
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)
        u_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :] * self.fe.face_shape_vals[boundary_inds[:, 1]][
                                                                          :, :, :, None]
        u_face = np.sum(u_face, axis=2)  # (num_selected_faces, num_face_quads, vec)
        # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
        # subset_quad_points = self.get_physical_surface_quad_points(boundary_inds)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face,
                                                   subset_quad_points)  # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return np.sqrt(np.square(val-self.target))
    def setTarget(self,target):
        self.target = target


######################################################################
# Do some cleaning work. Remove old solution files.
data_path = os.path.join(os.path.dirname(__file__), 'data')
files = glob.glob(os.path.join(data_path, f'vtk/*'))
for f in files:
    os.remove(f)

# Specify mesh-related information. We use first-order quadrilateral element.
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)

Nx = 100
Ny = 50
Lx, Ly = 100., 50.

meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# Define boundary conditions and values.
def fixed_location(point):
    return np.isclose(point[0], 0., atol=1e-5)


def load_location(point):
    return np.logical_and(np.isclose(point[0], Lx, atol=1e-5),np.logical_and(
        np.greater(point[1],0),np.less(point[1],Ly/2)
    ))

def dirichlet_val(point):
    return 0.


dirichlet_bc_info = [[fixed_location] * 2, [0, 1], [dirichlet_val] * 2]

location_fns = [load_location]

# Define forward problem.
problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info,
                     location_fns=location_fns)
problem2 = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info,
                     location_fns=location_fns)


# Apply the automatic differentiation wrapper.
# This is a critical step that makes the problem solver differentiable.
fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})
fwd_pred2 = ad_wrapper(problem2, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})


# Define the objective function 'J_total(theta)'.
# In the following, 'sol = fwd_pred(params)' basically says U = U(theta).
def J_total(params):
    """
    目标函数
    :param params:
    :return:
    """
    # J(u(theta), theta)
    sol_list = fwd_pred(params)
    compliance = problem.compute_compliance(sol_list[0])
    """指定目标"""
    # compliance = problem.compute_compliance_target(sol_list[0],target=0)
    return compliance

def J_total2(params):
    """
    目标函数
    :param params:
    :return:
    """
    # J(u(theta), theta)
    sol_list = fwd_pred2(params)
    compliance = problem2.compute_compliance(sol_list[0])
    """指定目标"""
    # compliance = problem.compute_compliance_target(sol_list[0],target=0)
    return compliance



# Output solution files to local disk
outputs = []


def output_sol(params, obj_val):
    print(f"\nOutput solution - need to solve the forward problem again...")
    sol_list = fwd_pred(params)
    sol = sol_list[0]
    vtu_path = os.path.join(data_path, f'vtk/sol_{output_sol.counter:03d}.vtu')
    save_sol(problem.fe, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path,
             cell_infos=[('theta', problem.full_params[:, 0])])
    print(f"compliance or var = {obj_val}")
    outputs.append(obj_val)
    output_sol.counter += 1

def output_sol2(params, obj_val):
    print(f"\nOutput solution - need to solve the forward problem again...")
    sol_list = fwd_pred2(params)
    sol = sol_list[0]
    vtu_path = os.path.join(data_path, f'vtk/sol_{output_sol.counter:03d}.vtu')
    save_sol(problem2.fe, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path,
             cell_infos=[('theta', problem2.full_params[:, 0])])
    print(f"compliance or var = {obj_val}")
    outputs.append(obj_val)
    output_sol.counter += 1

output_sol.counter = 0


# Prepare J_total and dJ/d(theta) that are required by the MMA optimizer.
def objectiveHandle(p):
    """
    定义目标函数和梯度计算 (MMA 使用)
    :param p:
    :return:
    """
    # MMA solver requires (J, dJ) as inputs
    # J has shape ()
    # dJ has shape (...) = p.shape
    J, dJ = jax.value_and_grad(J_total)(p)
    output_sol(p, J)
    return J, dJ

def objectiveHandle2(p):
    """
    定义目标函数和梯度计算 (MMA 使用)
    :param p:
    :return:
    """
    # MMA solver requires (J, dJ) as inputs
    # J has shape ()
    # dJ has shape (...) = p.shape
    J, dJ = jax.value_and_grad(J_total2)(p)
    output_sol2(p, J)
    return J, dJ


# Prepare g and dg/d(theta) that are required by the MMA optimizer.
def consHandle(rho):
    """
    定义约束
    :rho:
    :return:
    """
    # MMA solver requires (c, dc) as inputs
    # c should have shape (numConstraints,)
    # dc should have shape (numConstraints, ...)
    def computeGlobalVolumeConstraint(rho):
        # thetas = generate_voronoi(op, p)
        # thetas = thetas.reshape(-1, 1)
        g = np.mean(rho) / vf - 1.
        return g
    c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(rho)
    c, gradc = c.reshape((1,)), gradc[None, ...]
    return c, gradc


# Finalize the details of the MMA optimizer, and solve the TO problem.
vf = 0.5




sites_num=30
dim=2
margin=0
coordinates = np.indices((Nx, Ny))

def generate_points(Nx, Ny, sx,sy):
    # uniform points in 0,Nx 0,Ny
    x = np.linspace(0-margin, Nx+margin, sx)
    y = np.linspace(0-margin, Ny+margin, sy)
    points=np.meshgrid(x, y)
    xa=points[0].flatten()
    ya=points[1].flatten()
    points=np.column_stack((xa, ya))
    return points


sites = generate_points(Nx, Ny, 10,3)
time_start=time.time()

sites_low = np.tile(np.array([0 - margin, 0 - margin]), (sites_num, 1))
sites_up = np.tile(np.array([Nx + margin, Ny + margin]),(sites_num, 1))
Dm_low = np.tile(np.array([[0, 0], [0, 0]]), (sites_low.shape[0], 1, 1))
Dm_up = np.tile(np.array([[2, 2], [2, 2]]), (sites_low.shape[0], 1, 1))
cauchy_points_low = sites_low
cauchy_points_up = sites_up
bound_low = np.concatenate((np.ravel(sites_low), np.ravel(Dm_low), np.ravel(cauchy_points_low)), axis=0)[:, None]
bound_up = np.concatenate((np.ravel(sites_up), np.ravel(Dm_up), np.ravel(cauchy_points_up)), axis=0)[:, None]

Dm = np.tile(np.array(([1, 0], [0, 1])), (sites.shape[0], 1, 1))  # Nc*dim*dim
cauchy_points=sites.copy()
numConstraints = 1

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# optimizationParams = {'maxIters': 249, 'movelimit': 0.1,"coordinates":coordinates,"sites_num":sites_num,"Dm_dim":dim,
#                       "Nx":Nx,"Ny":Ny,"margin":margin,
#                       "heaviside":True,"cauchy":False,
#                       "bound_low":bound_low,"bound_up":bound_up,"paras_at":(0,sites_num*6),
#                       "cauchy_points":cauchy_points,"immortal":["cauchy_points"]}
#
# problem.op=optimizationParams
# # p_ini= np.concatenate((np.ravel(sites),np.ravel(Dm),np.ravel(cauchy_points)),axis=0)# 1-d array contains flattened: sites,Dm,cauchy points
# p_ini= np.concatenate((np.ravel(sites),np.ravel(Dm)),axis=0)# 1-d array contains flattened: sites,Dm,cauchy points
# p_oped,j=optimize(problem.fe, p_ini, optimizationParams, objectiveHandle, consHandle, numConstraints,generate_voronoi_separate)
""""""""""""""""""""""""""""""""""""""""""""""""""




# sites=p_oped[0:sites_num*dim].reshape((sites_num, dim))
# Dm=p_oped[sites_num*dim:].reshape((sites_num,dim,dim))
optimizationParams2 = {'maxIters': 99, 'movelimit': 0.5,"coordinates":coordinates,"sites_num":sites_num,"Dm_dim":dim,
                      "Nx":Nx,"Ny":Ny,"margin":margin,
                      "heaviside":True,"cauchy":True,
                      "bound_low":bound_low,"bound_up":bound_up,"paras_at":(sites_num*6,sites_num*8),
                      "sites":sites,"Dm":Dm,"immortal":["sites","Dm"]}
problem2.op=optimizationParams2
# problem.setTarget(j*1.5)
cauchy_points=sites.copy()
p_ini2= np.ravel(cauchy_points)# 1-d array contains flattened: sites,Dm,cauchy points
optimize(problem2.fe, p_ini2, optimizationParams2, objectiveHandle2, consHandle, numConstraints,generate_voronoi_separate)


print(f"As a reminder, compliance = {J_total(np.ones((len(problem.fe.flex_inds), 1)))} for full material")
print(f"running time:{time.time() - time_start}")
# Plot the optimization results.
obj = onp.array(outputs)
plt.figure(figsize=(10, 8))
plt.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
plt.xlabel(r"Optimization step", fontsize=20)
plt.ylabel(r"Objective value", fontsize=20)
plt.tick_params(labelsize=20)
plt.tick_params(labelsize=20)
plt.show()
