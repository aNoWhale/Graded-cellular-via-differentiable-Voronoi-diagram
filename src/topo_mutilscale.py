# Import some useful modules.
import time

import numpy as onp
import jax
import jax.numpy as np
import os
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=80"
import sys
import glob
import matplotlib
from scipy import ndimage
from scipy.ndimage import zoom, sobel


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
import softVoronoi
from softVoronoi_cell import generate_voronoi_separate,generate_para_rho
import ultilies as ut


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
"""global parameters"""
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
# Output solution files to local disk
outputs = []
outputs2 = []
# Finalize the details of the MMA optimizer, and solve the TO problem.
dim = 2



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
margin = 2

""""""""""""""""first step"""""""""""""""""""""
"""define model"""
#Nx*Ny should %100 = 0
Nx = 100
Ny = 50
resolution=1
Lx, Ly = Nx*resolution, Ny*resolution
coordinates = np.indices((Nx, Ny))*resolution
meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
"""define problem"""
# Define boundary conditions and values.
def fixed_location(point):
    return np.isclose(point[0], 0., atol=1e-5)
    # return np.logical_or(np.logical_and(np.isclose(point[0], 0., atol=0.1*Lx+1e-5),np.isclose(point[1], 0., atol=0.1*Ly+1e-5)),
    #                      np.logical_and(np.isclose(point[0], Lx, atol=0.1*Lx+1e-5),np.isclose(point[1], 0., atol=0.1*Ly+1e-5)))
def load_location(point):
    # return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0, atol=0.1 * Ly + 1e-5))
    return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], Ly/2, atol=0.1 * Ly/2 + 1e-5))
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
vf=0.3 #0.3
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

problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info,
                     location_fns=location_fns)
fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})
# fwd_pred = ad_wrapper(problem, solver_options = {'jax_solver': {}})


"""define parameters"""
sx,sy=8,2
sites=generate_points(Lx,Ly,sx,sy)
sites_num=sx*sy
sites_low = np.tile(np.array([0 - margin, 0 - margin]), (sites_num, 1))*resolution
sites_up = np.tile(np.array([Nx + margin, Ny + margin]), (sites_num, 1))*resolution
Dm_low = np.tile(np.array([[0.5, 0], [0, 0.5]]), (sites_low.shape[0], 1, 1))
Dm_up = np.tile(np.array([[2, 2], [2, 2]]), (sites_low.shape[0], 1, 1))
cp_low = sites_low
cp_up = sites_up
bound_low = np.concatenate((np.ravel(sites_low), np.ravel(Dm_low),np.ravel(cp_low)), axis=0)[:, None]
bound_up = np.concatenate((np.ravel(sites_up), np.ravel(Dm_up),np.ravel(cp_up)), axis=0)[:, None]
Dm = np.tile(np.array(([1, 0], [0, 1])), (sites.shape[0], 1, 1))  # Nc*dim*dim
cp = sites.copy()

optimizationParams = {'maxIters': 70, 'movelimit': 0.1, "lastIters":0,"stage":0,
                      "coordinates": coordinates, "sites_num": sites_num,"resolution":resolution,
                      "dim": dim,
                      "Nx": Nx, "Ny": Ny, "margin": margin,
                      "heaviside": True, "control": False,
                      "bound_low": bound_low, "bound_up": bound_up, "paras_at": (0, sites_num * 6),
                      "immortal": []}
problem.op = optimizationParams
p_ini=np.concatenate((sites.ravel(), Dm.ravel()))
numConstraints = 1
if False:
    p_oped, j ,rho_oped= optimize(problem.fe, p_ini, optimizationParams, objectiveHandle, consHandle1, numConstraints,softVoronoi.generate_voronoi_separate )
    np.save("data/p_oped.npy", p_oped)
    np.save("data/j.npy", j)
    np.save("data/rho_oped.npy", rho_oped)
else:
    p_oped=np.load("data/p_oped.npy")
    j=np.load("data/j.npy")
    rho_oped=np.load("data/rho_oped.npy")
# rho_ini = vf*np.ones((len(problem.fe.flex_inds), 1))
# rho,j=optimize_rho(problem.fe, rho_ini, optimizationParams, objectiveHandle, consHandle1, numConstraints )
"""""""""""""""""""""""""""""""""scale up"""""""""""""""""""""""""""""""""

# 计算缩放比例
resolution2=0.01
scale_y = 3
scale_x = 3
Nx2,Ny2=Nx*scale_x,Ny*scale_y
Lx2,Ly2=Nx2*resolution2,Ny2*resolution2
padding_size=20 # pixel
coordinates = np.indices((Nx2, Ny2+padding_size*2))*resolution2

# 使用 zoom 进行缩放
rho_oped=rho_oped.reshape(Nx,Ny)
rho_oped = np.array(zoom(rho_oped, (scale_x, scale_y), order=1))  # order=1 表示线性插值
padding=np.zeros((Nx2,padding_size))
rho_oped=np.concatenate((padding,rho_oped,padding ), axis=1)
rho_oped=rho_oped.ravel()
"""""""""""""""""""""""""""""""""infill reconstruct"""""""""""""""""""""""""""""""""
rho=rho_oped.reshape((Nx2, Ny2))
last_vf=np.mean(rho_oped)
#硬边界
# rho_mask = rho
# structure = ndimage.generate_binary_structure(2, 2)  # 定义结构元素
# binary_matrix = (rho_mask > 0.5)
# boundary = binary_matrix ^ ndimage.binary_erosion(binary_matrix, structure=structure)
# # 软边界
# rho_mask=ut.blur_edges(rho,blur_sigma=1.)
# boundary=ut.extract_continuous_boundary(rho,threshold=0.5)
sites_boundary=p_oped[:optimizationParams["sites_num"]*2].reshape((-1,2))
sites_boundary=sites_boundary.at[:,0].set(sites_boundary[:,0]*scale_x*resolution2/resolution)
sites_boundary=sites_boundary.at[:,1].set(sites_boundary[:,1]*scale_y*resolution2/resolution)
sites_boundary=sites_boundary.at[:,1].set(sites_boundary[:,1]+padding_size*resolution2)
Dm_boundary=p_oped[optimizationParams["sites_num"]*2:].reshape((-1,2,2))*50 #50
"""""""""""""""""""""""""""""""""""""""""""""second step"""""""""""""""""""""""""""""""""""""""""""""
"""define model"""
meshio_mesh2 = rectangle_mesh(Nx=Nx2, Ny=Ny2, domain_x=Lx2, domain_y=Ly2)
mesh2 = Mesh(meshio_mesh2.points, meshio_mesh2.cells_dict[cell_type])
"""define problem"""
# Define boundary conditions and values.
def fixed_location2(point):
    return np.isclose(point[0], 0., atol=1e-5)
    # return np.logical_or(np.logical_and(np.isclose(point[0], 0., atol=0.1*Lx+1e-5),np.isclose(point[1], 0., atol=0.1*Ly+1e-5)),
    #                      np.logical_and(np.isclose(point[0], Lx, atol=0.1*Lx+1e-5),np.isclose(point[1], 0., atol=0.1*Ly+1e-5)))
def load_location2(point):
    # return np.logical_and(np.isclose(point[0], Lx2, atol=1e-5), np.isclose(point[1], 0, atol=0.1 * Ly2 + 1e-5))
    return np.logical_and(np.isclose(point[0], Lx2, atol=1e-5), np.isclose(point[1], Ly2/2., atol=0.1 * Ly2/2 + 1e-5))
    # return  np.logical_and(np.isclose(point[0], Lx/2, atol=0.1*Lx+1e-5),
    #                        np.isclose(point[1], Ly, atol=0.1*Ly+1e-5))
def dirichlet_val2(point):
    return 0.
dirichlet_bc_info2 = [[fixed_location2] * 2, [0, 1], [dirichlet_val2] * 2]
location_fns2 = [load_location2]
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
def output_sol2(params, obj_val):
    print(f"\nOutput solution - need to solve the forward problem again...")
    sol_list = fwd_pred2(params)
    sol = sol_list[0]
    vtu_path = os.path.join(data_path, f'vtk/sol_{output_sol.counter:03d}.vtu')
    save_sol(problem2.fe, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path,
             cell_infos=[('theta', problem2.full_params[:, 0])], )
    # point_infos = [("sites", params[0:problem2.op["sites_num"] * 2].reshape(problem2.op["sites_num"], problem2.op["Dm_dim"]))]
    print(f"compliance or var = {obj_val}")
    outputs2.append(obj_val)
    output_sol.counter += 1
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
vf=last_vf
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
problem2 = Elasticity(mesh2, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info2,
                      location_fns=location_fns2)
fwd_pred2 = ad_wrapper(problem2, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})

# sites=p_oped[:optimizationParams["sites_num"]*2].reshape((optimizationParams["sites_num"], 2))
# Dm=p_oped[-optimizationParams["sites_num"]*4:].reshape((optimizationParams["sites_num"], 2,2))
# print(f"Dm_boundary:{Dm_boundary}")
optimizationParams2 = {'maxIters': 10, 'movelimit': 0.2, "lastIters":optimizationParams['maxIters'],"stage":1,
                       "coordinates": coordinates,"resolution":resolution2,
                       "sites_boundary":sites_boundary,"Dm_boundary":Dm_boundary,"padding_size":padding_size,
                       # "sites_num": sites_num,
                       "dim": dim,
                       "Nx": Nx2, "Ny": Ny2, "margin": margin,"Lx":Lx2, "Ly":Ly2,
                       "heaviside": True, "control": True,
                       # "bound_low": bound_low, "bound_up": bound_up, "paras_at": (0, bound_low.shape[0]),
                       "immortal": []}
"""revise para"""
p_ini2,optimizationParams2=generate_para_rho(optimizationParams2, rho_oped)
# optimizationParams2["sites_num"]=sites.shape[0]
# p_ini2=cp.ravel()
# sites_low = np.tile(np.array([0 - optimizationParams2["margin"], 0 - optimizationParams2["margin"]]), (optimizationParams2["sites_num"], 1)) * optimizationParams2["resolution"]
# sites_up = np.tile(np.array([optimizationParams2["Nx"] + optimizationParams2["margin"], optimizationParams2["Ny"] + optimizationParams2["margin"]]), (optimizationParams2["sites_num"], 1)) * optimizationParams2["resolution"]
# Dm = np.tile(np.array(([100, 0], [0, 100])), (sites.shape[0], 1, 1))  # Nc*dim*dim
# Dm_low = np.tile(np.array([[0.1, 0], [0, 0.1]]), (sites_low.shape[0], 1, 1))
# Dm_up = np.tile(np.array([[200, 200], [200, 200]]), (sites_low.shape[0], 1, 1))
# cp = sites.copy()
# cp_low = cp.ravel()-30
# cp_up = cp.ravel()+30
# optimizationParams2["bound_low"] = np.concatenate((np.ravel(sites_low), np.ravel(Dm_low), np.ravel(cp_low)), axis=0)[:, None]
# optimizationParams2["bound_up"] = np.concatenate((np.ravel(sites_up), np.ravel(Dm_up), np.ravel(cp_up)), axis=0)[:, None]
# optimizationParams2["paras_at"] = (optimizationParams2["sites_num"] * 6, optimizationParams2["sites_num"] * 8)
#
# problem2.op = optimizationParams2
problem2.setTarget(j*0)
# cauchy_points=sites.copy()

p_final,j_now,_ =optimize(problem2.fe, p_ini2, optimizationParams2, objectiveHandle2, consHandle2, numConstraints,
         generate_voronoi_separate)


"""""""""""""""""""""""""""""""""""""""""""""""""""plot result"""""""""""""""""""""""""""""""""""""""""""""""""""
print(f"As a reminder, compliance = {J_total(np.ones((len(problem.fe.flex_inds), 1)))} for full material")
print(f"previous J/compliance :{j}\n now error:{j_now}")
print(f"running time:{time.time() - time_start}")
# Plot the optimization results.
obj = onp.array(outputs)
obj2 = onp.array(outputs2)
fig=plt.figure(figsize=(16, 8))
ax1=fig.add_subplot(1, 2, 1)
ax1.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
ax1=fig.add_subplot(1, 2, 2)
ax1.plot(onp.arange(len(obj2)) + 1, obj2, linestyle='-', linewidth=2, color='black')
plt.draw()
plt.savefig(f'data/vtk/result.png', dpi=600, bbox_inches='tight')
# plt.show()
