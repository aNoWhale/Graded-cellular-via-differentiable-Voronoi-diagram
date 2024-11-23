# Import some useful modules.
import time

import numpy as onp
import jax
import jax.numpy as np
import os
import sys
import glob
import matplotlib
from scipy.ndimage import zoom

matplotlib.use('Qt5Agg') #for WSL
import matplotlib.pyplot as plt
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jax_fem_voronoi_dir = os.path.join(parent_dir, 'jax-fem-voronoi')
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(jax_fem_voronoi_dir)
sys.path.append(src_dir)
# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.mma import optimize
from jax_fem.mma_original import optimize_rho
# Define constitutive relationship.
# Generally, JAX-FEM solves -div.(f(u_grad,alpha_1,alpha_2,...,alpha_N)) = b.
# Here, we have f(u_grad,alpha_1,alpha_2,...,alpha_N) = sigma(u_grad, theta),
# reflected by the function 'stress'. The functions 'custom_init'and 'set_params'
# override base class methods. In particular, set_params sets the design variable theta.

from softVoronoi_cell import generate_voronoi_separate,generate_para_rho
plt.ion()
fig, ax = plt.subplots()

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
            Emax = 70e3
            Emin = 1e-5 * Emax
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
            # load define
            return np.array([0., -100.])

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
        return np.sqrt(np.square(val - self.target))

    def setTarget(self, target):
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
resolution=0.03
Lx, Ly = Nx*resolution, Ny*resolution




# Define boundary conditions and values.
def fixed_location(point):
    return np.isclose(point[0], 0., atol=1e-5)
    # return np.logical_or(np.logical_and(np.isclose(point[0], 0., atol=0.1*Lx+1e-5),np.isclose(point[1], 0., atol=0.1*Ly+1e-5)),
    #                      np.logical_and(np.isclose(point[0], Lx, atol=0.1*Lx+1e-5),np.isclose(point[1], 0., atol=0.1*Ly+1e-5)))


def load_location(point):
    return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0, atol=0.1 * Ly + 1e-5))
    # return  np.logical_and(np.isclose(point[0], Lx/2, atol=0.1*Lx+1e-5),
    #                        np.isclose(point[1], Ly, atol=0.1*Ly+1e-5))


def dirichlet_val(point):
    return 0.


dirichlet_bc_info = [[fixed_location] * 2, [0, 1], [dirichlet_val] * 2]

location_fns = [load_location]







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
             cell_infos=[('theta', problem.full_params[:, 0])], )
    # point_infos = [ ("sites", params[0:problem.op["sites_num"] * 2].reshape(problem.op["sites_num"], problem.op["Dm_dim"]))]
    print(f"compliance or var = {obj_val}")
    outputs.append(obj_val)
    output_sol.counter += 1


def output_sol2(params, obj_val):
    print(f"\nOutput solution - need to solve the forward problem again...")
    sol_list = fwd_pred2(params)
    sol = sol_list[0]
    vtu_path = os.path.join(data_path, f'vtk/sol_{output_sol.counter:03d}.vtu')
    save_sol(problem2.fe, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path,
             cell_infos=[('theta', problem2.full_params[:, 0])], )
    # point_infos = [("sites", params[0:problem2.op["sites_num"] * 2].reshape(problem2.op["sites_num"], problem2.op["Dm_dim"]))]
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


def consHandle1(rho):

    # MMA solver requires (c, dc) as inputs
    # c should have shape (numConstraints,)
    # dc should have shape (numConstraints, ...)
    def computeGlobalVolumeConstraint(rho):
        g = np.mean(rho) / vf - 1.
        return g

    c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(rho)
    c, gradc = c.reshape((1,)), gradc[None, ...]
    return c, gradc
def consHandle2(p):

    # MMA solver requires (c, dc) as inputs
    # c should have shape (numConstraints,)
    # dc should have shape (numConstraints, ...)
    def computeGlobalVolumeConstraint(rho):
        # thetas = generate_voronoi(op, p)
        # thetas = thetas.reshape(-1, 1)
        g = np.mean(rho) / vf - 1.
        return g
    c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(p)
    c, gradc = c.reshape((1,)), gradc[None, ...]
    return c, gradc



# Finalize the details of the MMA optimizer, and solve the TO problem.
vf = 0.5

dim = 2
margin = 5
coordinates = np.indices((Nx, Ny))*resolution


def generate_points(Lx, Ly, sx, sy):
    # uniform points in 0,Nx 0,Ny
    x = np.linspace(0 , Lx , sx)
    y = np.linspace(0 , Ly , sy)
    points = np.meshgrid(x, y)
    xa = points[0].flatten()
    ya = points[1].flatten()
    points = np.column_stack((xa, ya))
    return points



time_start = time.time()

""""""""""""""""first step"""""""""""""""""""""
meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info,
                     location_fns=location_fns)
fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})

numConstraints = 1
optimizationParams = {'maxIters': 30, 'movelimit': 0.1, "lastIters":0,"stage":0,
                      "coordinates": coordinates, "sites_num": 0,
                      "dim": dim,
                      "Nx": Nx, "Ny": Ny, "margin": margin,}
problem.op = optimizationParams
rho_ini = vf*np.ones((Nx*Ny, 1))

rho_oped, j = optimize_rho(problem.fe, rho_ini, optimizationParams, objectiveHandle, consHandle1, numConstraints, )
"""""""""""""""""""""""""""""""""scale up"""""""""""""""""""""""""""""""""


# 计算缩放比例
scale_y = 2
scale_x = 2
rho_oped=rho_oped.reshape(Nx,Ny)
Nx2,Ny2=Nx*scale_x,Ny*scale_y
coordinates = np.indices((Nx2, Ny2))*resolution
# 使用 zoom 进行缩放
rho_oped = np.array(zoom(rho_oped, (scale_x, scale_y), order=1))  # order=1 表示线性插值
rho_oped=rho_oped.ravel()


"""""""""""""""""""""""""""second step"""""""""""""""""""""""""""""
meshio_mesh2 = rectangle_mesh(Nx=Nx2, Ny=Ny2, domain_x=Lx, domain_y=Ly)
mesh2 = Mesh(meshio_mesh2.points, meshio_mesh2.cells_dict[cell_type])
problem2 = Elasticity(mesh2, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info,
                      location_fns=location_fns)
fwd_pred2 = ad_wrapper(problem2, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})
# sites=generate_points(Lx,Ly,10,3)
# sites_low = np.tile(np.array([0 - margin, 0 - margin]), (sites_num, 1))
# sites_up = np.tile(np.array([Nx + margin, Ny + margin]), (sites_num, 1))
#
# Dm_low = np.tile(np.array([[0, 0], [0, 0]]), (sites_low.shape[0], 1, 1))
# Dm_up = np.tile(np.array([[2000, 2000], [2000, 2000]]), (sites_low.shape[0], 1, 1))
# cp_low = sites_low
# cp_up = sites_up
# rho_low=np.zeros((Nx*Ny))
# rho_up=np.ones((Nx*Ny))
# bound_low = np.concatenate((np.ravel(rho_low), np.ravel(Dm_low), np.ravel(cp_low)), axis=0)[:, None]
# bound_up = np.concatenate((np.ravel(rho_up), np.ravel(Dm_up), np.ravel(cp_up)), axis=0)[:, None]
# Dm = np.tile(np.array(([1000, 0], [0, 1000])), (sites.shape[0], 1, 1))  # Nc*dim*dim

# cp = sites.copy()

# Dm = rho_oped[sites_num * dim:].reshape((sites_num, dim, dim))
optimizationParams2 = {'maxIters': 100, 'movelimit': 0.5, "lastIters":optimizationParams['maxIters'],"stage":1,
                       "coordinates": coordinates,"resolution":resolution,
                       # "sites_num": sites_num,
                       "dim": dim,
                       "Nx": Nx2, "Ny": Ny2, "margin": margin,
                       "heaviside": False, "control": False,
                       # "bound_low": bound_low, "bound_up": bound_up, "paras_at": (0, bound_low.shape[0]),
                        "immortal": []}
"""""""""""""""""""""""""""""""""revise para"""""""""""""""""""""""""""""""""
p_ini2,optimizationParams2=generate_para_rho(optimizationParams2, rho_oped)

problem2.op = optimizationParams2
problem2.setTarget(0)
# cauchy_points=sites.copy()

p_final,j_now =optimize(problem2.fe, p_ini2, optimizationParams2, objectiveHandle2, consHandle2, numConstraints,
         generate_voronoi_separate)


print(f"As a reminder, compliance = {J_total(np.ones((len(problem.fe.flex_inds), 1)))} for full material")
print(f"previous J/compliance :{j}\n now error:{j_now}")
print(f"running time:{time.time() - time_start}")
# Plot the optimization results.
obj = onp.array(outputs)
fig=plt.figure(figsize=(16, 8))
ax1=fig.add_subplot(1, 1, 1)
ax1.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
plt.draw()
plt.savefig(f'data/vtk/result.png', dpi=300, bbox_inches='tight')
# plt.show()
