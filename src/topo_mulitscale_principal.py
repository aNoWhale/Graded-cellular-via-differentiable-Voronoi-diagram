# Import some useful modules.
import logging
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
from jax_fem import logger
from jax_fem.solver import solver

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
            Emax = 70e3 #70e3 MPa?
            Emin = 1e-5 * Emax
            nu = 0.3
            penal = 3. #1 freeend3
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

    #changkun sun added here
    def get_tensor_strain_map(self):
        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon
        return strain

    def get_surface_maps(self):
        def surface_map(u, x):
            # load define
            return np.array([0., -100.]) #0 -100

        return [surface_map]

    def set_params(self, params):
        # Override base class method.
        """edited from tianxu xue"""
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]

    def set_rho(self,rho):
        thetas = rho.reshape(-1, 1)
        full_params = np.ones((self.fe.num_cells, rho.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(rho)
        theta = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [theta]
        return thetas

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

    def compute_stiffness(self, sol):
        # 获取所有单元索引
        # cell_inds = np.arange(len(self.fe.cells))

        # 获取形函数梯度和 Jacobian 行列式
        cell_grads, jacobian_det = self.fe.get_shape_grads()  # 假设不需要 cell_inds 参数

        # 位移梯度: strain = grad(u)
        # u_cell = sol[self.fe.cells]  # (num_cells, num_nodes, dim)，提取单元位移
        # strain = np.einsum('cnd,cqnd->cqd', u_cell, cell_grads)  # (num_cells, num_quads, dim, dim)
        u_grad = self.fes[0].sol_to_grad(sol)
        #非线性
        # strain = 0.5 * np.einsum('cndm,cqnd->cqdm', u_grad, cell_grads) # (num_cells, num_quads, dim, dim)
        #线性
        strain_fn=self.get_tensor_strain_map()
        strain = jax.vmap(jax.vmap(strain_fn))(u_grad)
        # strainmax=strain.max()
        # strainmin=strain.min()
        # 获取应力计算函数
        stress_fn = self.get_tensor_map()  # 使用实例化的 `get_tensor_map` 函数，直接获取应力计算函数

        # 计算应力
        # stress = jax.vmap(jax.vmap(stress_fn))(strain, self.internal_vars[0])  # 应力计算，theta 和 strain
        stress = jax.vmap(jax.vmap(stress_fn))(u_grad, self.internal_vars[0])  # 应力计算，theta 和 strain
        # stressmax=stress.max()
        # stressmin=stress.min()
        # 计算能量密度: W = 0.5 * stress : strain
        energy_density = 0.5 * np.einsum('cqij,cqij->cq', stress, strain)  # (num_cells, num_quads)


        # 通过 Jacobian 行列式积分计算总刚度
        stiffness = np.sum(energy_density * jacobian_det)  # 标量
        return stiffness
        # return np.sqrt(np.square(stiffness - self.target))

    def compute_first_principal_stress(self, sol):
        # 获取形函数梯度和 Jacobian 行列式
        cell_grads, _ = self.fe.get_shape_grads()
        # 计算位移梯度
        u_grad = self.fes[0].sol_to_grad(sol)
        # 计算应力张量
        stress_fn = self.get_tensor_map()
        stress = jax.vmap(jax.vmap(stress_fn))(u_grad, self.internal_vars[0])  # (num_cells, num_quads, dim, dim)
        # 计算主应力和主方向
        def compute_first_principal_stress(sigma):
            # 使用 jax 进行特征值分解
            eigvals, eigvecs = np.linalg.eigh(sigma)  # eigvals 是特征值, eigvecs 是特征向量
            # 查看 eigvals 和 eigvecs 的形状，确认它们的维度
            # print("eigvals:", eigvals)
            # print("eigvecs:", eigvecs)
            # 如果是二维矩阵，每个 sigma 可能有两个特征值
            # 选择每个矩阵的最大特征值
            principal_stress = eigvals[ -1]  # 获取每个矩阵的最大特征值（第一主应力）
            # 获取对应最大特征值的特征向量
            principal_direction = eigvecs[:, -1]  # 获取对应的特征向量
            return principal_stress, principal_direction

        # 使用 jax.vmap 对所有单元和积分点进行计算
        principal_stress, principal_directions = jax.vmap(jax.vmap(compute_first_principal_stress))(stress)
        return principal_stress, principal_directions

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

Lx, Ly =0.5,0.5#100 50
Nx = 50 #100
Ny = 50 #50
resolution=Lx/Nx
print(f"Nx = {Nx}, Ny = {Ny}")
assert Nx*Ny %100 == 0
coordinates = np.indices((Nx, Ny))*resolution
meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
"""define problem"""
# Define boundary conditions and values.
def fixed_location(point):
    return np.isclose(point[0], 0., atol=1e-5)
    # return np.logical_or(np.logical_and(np.isclose(point[0], 0., atol=0.1*Lx/2+1e-5),np.isclose(point[1], 0., atol=0.1*Ly/2+1e-5)),
    #                      np.logical_and(np.isclose(point[0], Lx, atol=0.1*Lx/2+1e-5),np.isclose(point[1], 0., atol=0.1*Ly/2+1e-5)))
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
vf=0.3 #0.3 0.35
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
# sx,sy=9,2 #8,2
# sites=generate_points(Lx,Ly,sx,sy)
# sites=sites[:-2,:]
onp.random.seed(0)
sites_x=onp.random.uniform(0,Lx,size=(25,1)) #20
sites_y=onp.random.uniform(0,Ly,size=(25,1))
sites=np.concatenate((sites_x, sites_y), axis=-1)
sites_num=sites.shape[0]
sites_low = np.tile(np.array([0 - margin, 0 - margin]), (sites_num, 1))*resolution
sites_up = np.tile(np.array([Nx + margin, Ny + margin]), (sites_num, 1))*resolution
Dm_low = np.tile(np.array([[0.3, 0], [0, 0.3]]), (sites_low.shape[0], 1, 1)) #0.5 0.4
Dm_up = np.tile(np.array([[1.5, 1.5], [1.5, 1.5]]), (sites_low.shape[0], 1, 1)) #2    /resolution
cp_low = sites_low
cp_up = sites_up
bound_low = np.concatenate((np.ravel(sites_low), np.ravel(Dm_low),np.ravel(cp_low)), axis=0)[:, None]
bound_up = np.concatenate((np.ravel(sites_up), np.ravel(Dm_up),np.ravel(cp_up)), axis=0)[:, None]
Dm = np.tile(np.array(([1, 0], [0, 1])), (sites.shape[0], 1, 1))/resolution  # Nc*dim*dim
cp = sites.copy()

optimizationParams = {'maxIters': 2, 'movelimit': 0.1, "lastIters":0,"stage":0,
                      "coordinates": coordinates, "sites_num": sites_num,"reso":resolution,
                      "dim": dim,
                      "Nx": Nx, "Ny": Ny, "margin": margin,
                      "heaviside": True, "control": False,
                      "bound_low": bound_low, "bound_up": bound_up, "paras_at": (0, sites_num * 6),
                      "immortal": []}
problem.op = optimizationParams
p_ini=np.concatenate((sites.ravel(), Dm.ravel()))
numConstraints = 1
"""first step"""
if True:
    p_oped, j ,rho_oped= optimize(problem.fe, p_ini, optimizationParams, objectiveHandle, consHandle1, numConstraints,softVoronoi.generate_voronoi_separate )
    np.save("data/p_oped.npy", p_oped)
    np.save("data/j.npy", j)
    np.save("data/rho_oped.npy", rho_oped)
    np.save("data/vtk/p_oped.npy", p_oped)
    np.save("data/vtk/j.npy", j)
    np.save("data/vtk/rho_oped.npy", rho_oped)
else:
    p_oped=np.load("data/p_oped.npy")
    j=np.load("data/j.npy")
    rho_oped=np.load("data/rho_oped.npy")
# rho_ini = vf*np.ones((len(problem.fe.flex_inds), 1))
# rho,j=optimize_rho(problem.fe, rho_ini, optimizationParams, objectiveHandle, consHandle1, numConstraints )
first_full=J_total(np.ones((len(problem.fe.flex_inds), 1)))
logger.info(f"As a reminder, compliance = {first_full} for full material")
"""""""""""""""""""""""""""""""""scale up"""""""""""""""""""""""""""""""""
print(f"zooming up......")
# 计算缩放比例
scale = 3
resolution2=round(resolution/scale,4)
padding_size=10 # pixel
Lx2,Ly2=Lx,Ly+padding_size*2*resolution2
Nx2,Ny2= Nx * scale, Ny * scale + padding_size * 2
print(f"Nx2 = {Nx2}, Ny2 = {Ny2}")
coordinates = np.indices((Nx2, Ny2))*resolution2

# 使用 zoom 进行缩放
rho_oped=rho_oped.reshape(Nx,Ny)
rho_oped = np.array(zoom(rho_oped, (scale, scale), order=1))  # order=1 表示线性插值
padding=np.zeros((Nx2,padding_size))
rho_oped=np.concatenate((padding,rho_oped,padding ), axis=1)
rho_oped=rho_oped.ravel()
"""""""""""""""""""""""""""""""""infill reconstruct"""""""""""""""""""""""""""""""""
rho=rho_oped.reshape((Nx2, Ny2))
# last_vf=np.mean(rho_oped)
last_vf=vf

#硬边界
# rho_mask = rho
# structure = ndimage.generate_binary_structure(2, 2)  # 定义结构元素
# binary_matrix = (rho_mask > 0.5)
# boundary = binary_matrix ^ ndimage.binary_erosion(binary_matrix, structure=structure)
# # 软边界
# rho_mask=ut.blur_edges(rho,blur_sigma=1.)
# boundary=ut.extract_continuous_boundary(rho,threshold=0.5)
sites_boundary=p_oped[:optimizationParams["sites_num"]*2].reshape((-1,2))
sites_boundary=sites_boundary.at[:,0].set(sites_boundary[:,0])
sites_boundary=sites_boundary.at[:,1].set(sites_boundary[:,1])
sites_boundary=sites_boundary.at[:,1].set(sites_boundary[:,1]+padding_size*resolution2)
Dm_boundary=p_oped[optimizationParams["sites_num"]*2:].reshape((-1,2,2))*resolution2*6 #50 *resolution *0.3 #6
first_step_time=time.time()
"""""""""""""""""""""""""""""""""""""""""""""second step"""""""""""""""""""""""""""""""""""""""""""""
"""define model"""
meshio_mesh2 = rectangle_mesh(Nx=Nx2, Ny=Ny2, domain_x=Lx2, domain_y=Ly2)
mesh2 = Mesh(meshio_mesh2.points, meshio_mesh2.cells_dict[cell_type])
"""define problem"""
# Define boundary conditions and values.
def fixed_location2(point):
    return np.isclose(point[0], 0., atol=1e-5)
    # return np.logical_or(np.logical_and(np.isclose(point[0], 0., atol=0.1*Lx2/2+1e-5),np.isclose(point[1], 0., atol=0.1*Ly2/2+1e-5)),
    #                      np.logical_and(np.isclose(point[0], Lx2, atol=0.1*Lx2/2+1e-5),np.isclose(point[1], 0., atol=0.1*Ly2/2+1e-5)))
def load_location2(point):
    # return np.logical_and(np.isclose(point[0], Lx2, atol=1e-5), np.isclose(point[1], 0, atol=0.1 * Ly2 + 1e-5))
    return np.logical_and(np.isclose(point[0], Lx2, atol=1e-5), np.isclose(point[1], Ly2/2., atol=0.1 * Ly2/2 + 1e-5))
    # return  np.logical_and(np.isclose(point[0], Lx2/2, atol=0.1*Lx2+1e-5),
    #                        np.isclose(point[1], Ly2, atol=0.1*Ly2+1e-5))
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
    # compliance = problem2.compute_compliance(sol_list[0])
    stiffness = problem2.compute_stiffness(sol_list[0])

    """指定目标"""
    # compliance = problem.compute_compliance_target(sol_list[0],target=0)
    return stiffness
def output_sol2(params, obj_val):
    print(f"\nOutput solution - need to solve the forward problem again...")
    sol_list = fwd_pred2(params)
    sol = sol_list[0]
    vtu_path = os.path.join(data_path, f'vtk/sol_{output_sol.counter:03d}.vtu')
    save_sol(problem2.fe, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path,
             cell_infos=[('theta', problem2.full_params[:, 0])], )
    # point_infos = [("sites", params[0:problem2.op["sites_num"] * 2].reshape(problem2.op["sites_num"], problem2.op["Dm_dim"]))]
    print(f"stiffness or var = {obj_val}")
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
optimizationParams2 = {'maxIters': 2, 'movelimit': 0.1, "lastIters":optimizationParams['maxIters'],"stage":1, #limit0.2
                       "coordinates": coordinates,"reso":resolution2,
                       "sites_boundary":sites_boundary,"Dm_boundary":Dm_boundary,
                       "padding_size":padding_size,
                       # "sites_num": sites_num,
                       "dim": dim,
                       "Nx": Nx2, "Ny": Ny2, "margin": margin,"Lx":Lx2, "Ly":Ly2,
                       "heaviside": True, "control": False,
                       # "bound_low": bound_low, "bound_up": bound_up, "paras_at": (0, bound_low.shape[0]),
                       "immortal": []}
"""revise para"""
p_ini2,optimizationParams2=generate_para_rho(optimizationParams2, rho_oped)

problem2.setTarget(0.1)

"""second step"""
if True:
    p_final, j_now, rho_oped2 = optimize(problem2.fe, p_ini2, optimizationParams2, objectiveHandle2, consHandle2,
                                         numConstraints,
                                         generate_voronoi_separate)
    np.save("data/p_final.npy", p_final)
    np.save("data/j_now.npy", j_now)
    np.save("data/rho_oped2.npy", rho_oped2)
    np.save("data/vtk/p_final.npy", p_final)
    np.save("data/vtk/j_now.npy", j_now)
    np.save("data/vtk/rho_oped2.npy", rho_oped2)
else:
    p_final=np.load("data/p_final.npy")
    j_now=np.load("data/j_now.npy")
    rho_oped2=np.load("data/rho_oped2.npy")

"""""""""""""""""""""""""""""""""""""""""""distortion"""""""""""""""""""""""""""""""""""""""""""
"""define model"""
meshio_mesh3 = rectangle_mesh(Nx=Nx2, Ny=Ny2, domain_x=Lx2, domain_y=Ly2)
mesh3 = Mesh(meshio_mesh2.points, meshio_mesh2.cells_dict[cell_type])
"""define problem"""
outputs3=[]
def output_sol3(sol_list, obj_val,principal_stress, principal_directions,name=None):
    print(f"\nOutput solution - need to solve the forward problem again...")
    sol = sol_list[0]
    arrow_start_points = problem3.fe.points[problem3.fe.cells] # 获取所有单元的节点信息
    arrow_start_points = np.concatenate((arrow_start_points,np.zeros((arrow_start_points.shape[0],1))),axis=-1)
    arrow_directions = np.concatenate((principal_directions,np.zeros((principal_directions.shape[0],1))),axis=-1)  # 主应力方向
    # 定义VTU文件路径
    filename=f"sol_{output_sol.counter:03d}" if name is None else name
    vtu_path = os.path.join(data_path, f'vtk/{filename}.vtu')
    # 将解及第一主应力和方向存储到VTU文件中
    save_sol(problem3.fe, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path,
             cell_infos=[
                 ('theta', problem3.full_params[:, 0]),
                 ('PrincipalStress', principal_stress),  # 添加第一主应力
                 # ('PrincipalDirection', principal_directions)  # 添加第一主应力方向
             ],point_infos=[('arrow_start_points',arrow_start_points),
                            ('arrow_directions',arrow_directions)])
    # print(f"stiffness or var = {obj_val}")
    # outputs2.append(obj_val)
    # output_sol.counter += 1
    return arrow_start_points,arrow_directions,principal_stress


logging.info("Calculating stress problem")
problem3 = Elasticity(mesh3, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info2,
                      location_fns=location_fns2)
problem3.set_rho(rho_oped2)
sol_list = solver(problem3, solver_options={'umfpack_solver': {}})
principal_stress, principal_directions=problem3.compute_first_principal_stress(sol_list[0])

max_principal_stress = principal_stress.max(axis=1)  # (num_cells,)，每个单元的最大主应力
max_principal_directions = principal_directions[np.arange(principal_directions.shape[0]),
                                               np.argmax(principal_stress, axis=1), :]  # (num_cells, 2)


logging.info(f"Principal stress.shape: {principal_stress.shape}")
logging.info(f"Principal directions.shape: {principal_directions.shape}")
plt.clf()
plt.imshow(max_principal_stress.reshape(Nx2,Ny2),cmap="rainbow")
plt.draw()
plt.savefig(f'data/vtk/max_principal_stress.png', dpi=600, bbox_inches='tight')

# array_start_point,array_direction,_=output_sol3(sol_list,0,max_principal_stress,point_direction,"first_principal_stress")
# fwd_pred2 = ad_wrapper(problem2, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})
"""""""""""""""""""""""""""""""""""""""""where max stress at"""""""""""""""""""""""""""""""""""""""""
num_sites=optimizationParams2["sites_num"]
sites_ori=p_final[:num_sites*2].reshape((num_sites,2))
Dm=p_final[num_sites*2:num_sites*6].reshape((num_sites,2,2))
max_add_num=200 #150

max_principal_stress.reshape(Nx2,Ny2)
indices = np.argsort(max_principal_stress, axis=None)[int(-1*max_add_num):] #最大第一主应力所在的单元索引
# rows, cols = np.unravel_index(indices, max_principal_stress.shape)
arrow_start_points = problem3.fe.points[problem3.fe.cells]
max_stress_position=arrow_start_points[indices,:,:]
max_stress_position=np.mean(max_stress_position,axis=1)
max_stress_direction=max_principal_directions[indices,:]

max_stress_position,max_stress_direction=ut.remove_nearby_points(max_stress_position,max_stress_direction,threshold=resolution2*15) #15 18 15

max_stress_direction=np.stack((max_stress_direction[:,1]*-1,max_stress_direction[:,0]),axis=1) # vertical vector
logging.info(f"max_stress_position.shape: {max_stress_position.shape}")
print(f"max_stress_position.shape:{max_stress_position.shape[0]}")
cp_ori=sites_ori.copy()
cp=np.concatenate((cp_ori,max_stress_position+max_stress_direction*10),axis=0)
sites=np.concatenate((sites_ori,max_stress_position),axis=0)

Dm= np.concatenate((Dm,np.tile(np.array(([0.7,0],[0,0.7]))/resolution2,reps=(max_stress_position.shape[0],1,1))),axis=0) #0.9
optimizationParams3 = {'maxIters': 1, 'movelimit': 0.1, "lastIters":optimizationParams['maxIters'],"stage":1, #limit0.2
                       "coordinates": coordinates,"reso":resolution2,
                       "sites_boundary":sites_boundary,"Dm_boundary":Dm_boundary,
                       "padding_size":0,
                       "sites_num": sites.shape[0],
                       "dim": dim,
                       "Nx": Nx2, "Ny": Ny2, "margin": margin,"Lx":Lx2, "Ly":Ly2,
                       "heaviside": True, "control": True,
                       # "bound_low": bound_low, "bound_up": bound_up, "paras_at": (0, bound_low.shape[0]),
                       "immortal": []}


p=np.concatenate((np.ravel(sites),np.ravel(Dm),np.ravel(cp)),axis=0)
field=generate_voronoi_separate(optimizationParams3,p,epoch=20)



plt.clf()
plt.imshow(field,cmap="viridis")
plt.colorbar()
plt.draw()
plt.savefig(f'data/vtk/inserted_np.png', dpi=600, bbox_inches='tight')
plt.scatter(max_stress_position[:,1]/resolution2,max_stress_position[:,0]/resolution2,marker='+',c='violet')
plt.draw()
plt.savefig(f'data/vtk/inserted.png', dpi=600, bbox_inches='tight')
plt.scatter(sites_ori[:,1]/resolution2,sites_ori[:,0]/resolution2,marker='+',c='r')
plt.draw()
plt.savefig(f'data/vtk/inserted_sites.png', dpi=600, bbox_inches='tight')


"""""""""""""""""""""""""""""""""""""""""""""""""""plot result"""""""""""""""""""""""""""""""""""""""""""""""""""
print(f"As a reminder, 1st_compliance = {first_full} for full material")
print(f"As a reminder, 2rd_compliance = {J_total(np.ones((len(problem.fe.flex_inds), 1)))} for full material")
print(f"previous J/compliance :{j}\n now J/compliance:{j_now}")
print(f"Principal stress.shape: {principal_stress.shape}")
print(f"Principal directions.shape: {principal_directions.shape}")
print(f"first step time:{first_step_time-time_start}")
print(f"second step time:{time.time()-first_step_time}")
print(f"total running time:{time.time() - time_start}")
print(f"first elements:{Nx}*{Ny}")
print(f"second elements:{Nx2}*{Ny2}")


# Plot the optimization results.
obj = onp.array(outputs)
onp.savetxt(f"data/vtk/output1.csv", obj, delimiter=",")
obj2 = onp.array(outputs2)
onp.savetxt(f"data/vtk/output2.csv", obj2, delimiter=",")
fig=plt.figure(figsize=(16, 8))
ax1=fig.add_subplot(1, 2, 1)
ax1.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
ax1=fig.add_subplot(1, 2, 2)
ax1.plot(onp.arange(len(obj2)) + 1, obj2, linestyle='-', linewidth=2, color='black')
plt.draw()
plt.savefig(f'data/vtk/result.png', dpi=600, bbox_inches='tight')
# plt.show()
