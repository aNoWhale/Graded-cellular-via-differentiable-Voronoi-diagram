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
from numpy.ma.core import indices
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
# import softVoronoi
# from softVoronoi_cell import generate_voronoi_separate,generate_para_rho
from softVoronoi_cell import *
import ultilies as ut
from scipy.spatial import KDTree
def merge_close_points(points, threshold):
    tree = KDTree(points)
    # 记录每个点是否已被访问
    visited = onp.zeros(len(points), dtype=bool)
    merged_points = []
    for i, point in enumerate(points):
        if visited[i]:
            continue
        indices = tree.query_ball_point(point, r=threshold)
        cluster_points = points[indices]
        merged_point = cluster_points.mean(axis=0)
        merged_points.append(merged_point)
        visited[indices] = True
    return onp.array(merged_points)
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

    # def setTarget(self, target):
    #     self.target = target

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

time_start = time.time()
margin = 2

Lx, Ly =1,1#100 50
Nx = 100 #100
Ny = 100 #50
resolution=Lx/Nx
print(f"Nx = {Nx}, Ny = {Ny}")
print(f"resolution = {resolution}")
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

"""""""""""""""""""""""""""""1st"""""""""""""""""""""""""""""
onp.random.seed(0)
sites_x=onp.random.uniform(0,Lx,size=(64,1)) #20
sites_y=onp.random.uniform(0,Ly,size=(64,1))
sites=onp.concatenate((sites_x, sites_y), axis=-1)
sites=merge_close_points(sites,threshold=resolution*5)
sites=np.array(sites)
sites_num=sites.shape[0]
print(f"sites num:{sites_num}")
Dm = np.tile(np.array(([1.5, 0], [0, 1.5])), (sites.shape[0], 1, 1))/resolution  # Nc*dim*dim 1.2
cp = sites.copy()

coordinates = np.stack(coordinates, axis=-1)
rho = voronoi_field(coordinates, sites,rho_cell_m, Dm=Dm).reshape(Nx,Ny)
rho = heaviside_projection(rho, eta=0.5, epoch=50)
plt.imshow(rho, cmap='viridis')  # 使用 'viridis' 颜色映射
plt.colorbar(label='Pixel Value')  # 添加颜色条用于显示值的范围
# plt.title("Pixel Values Visualized with Colors")
plt.imsave(f"block/1_VD_np_img.png", rho, cmap='viridis')
plt.savefig(f"block/2_VD_np.png")
plt.draw()
plt.scatter(sites[:, 1] // resolution, sites[:, 0] // resolution, marker='+', color='r')
plt.savefig(f"block/3_VD.png")
plt.draw()

print(f"zooming up......")
# 计算缩放比例
scale = 4
resolution2=round(resolution/scale,4)
padding_size=0 # pixel
Lx2,Ly2=Lx,Ly+padding_size*2*resolution2
Nx2,Ny2= Nx * scale, Ny * scale + padding_size * 2
print(f"Nx2 = {Nx2}, Ny2 = {Ny2}")
coordinates = np.indices((Nx2, Ny2))*resolution2
coordinates = np.stack(coordinates, axis=-1)
# 使用 zoom 进行缩放
rho_oped=rho.reshape(Nx,Ny)
rho_oped = np.array(zoom(rho_oped, (scale, scale), order=1))  # order=1 表示线性插值
padding=np.zeros((Nx2,padding_size))
rho_oped=np.concatenate((padding,rho_oped,padding ), axis=1)
rho_oped=rho_oped.ravel()
"""""""""""""""""""""""""""""""""infill reconstruct"""""""""""""""""""""""""""""""""
rho=rho_oped.reshape((Nx2, Ny2))

rho=rho.reshape((Nx2,Ny2))
"""""""""""""""""""""""""""""""""""""""multiscale"""""""""""""""""""""""""""""""""""""""
# ## 整齐的生成sites
# density_x=20 #pixel 15
# density_y=20 #pixel 10
# matrix = np.ones((Nx2,Ny2), dtype=int)
# step_x = max(1, density_x)  # 行方向步长
# step_y = max(1, density_y)  # 列方向步长
# row_indices, col_indices = np.meshgrid(np.arange(0, Nx2, step_x), np.arange(0, Ny2, step_y), indexing='ij')
# row_indices = row_indices.ravel()  # 展平
# col_indices = col_indices.ravel()  # 展平
# matrix = matrix.at[row_indices, col_indices].set(0)
# sites2 = np.argwhere(matrix+0.5 < rho) * resolution2
# Dm2 = np.tile(np.array(([0.8, 0], [0, 0.8])), (sites2.shape[0], 1, 1))/resolution2  # Nc*dim*dim
# sites=np.concatenate((sites, sites2), axis=0)
# Dm=np.concatenate((Dm*1,Dm2),axis=0)
#
# rho = voronoi_field(coordinates, sites,rho_cell_m, Dm=Dm).reshape(Nx2,Ny2)
# rho = heaviside_projection(rho, eta=0.5, epoch=200)
# plt.clf()
# plt.imshow(rho, cmap='viridis')  # 使用 'viridis' 颜色映射
# plt.colorbar(label='Pixel Value')  # 添加颜色条用于显示值的范围
# # plt.title("Pixel Values Visualized with Colors")
# plt.imsave(f"block/4_VD2_np_img.png", rho, cmap='viridis')
# plt.savefig(f"block/5_VD2_np.png")
# plt.draw()
# plt.scatter(sites[:, 1] // resolution2, sites[:, 0] // resolution2, marker='+', color='r')
# plt.scatter(sites2[:, 1] // resolution2, sites2[:, 0] // resolution2, marker='+', color='w')
# plt.savefig(f"block/6_VD2.png")
# plt.draw()


"""define model"""
meshio_mesh2 = rectangle_mesh(Nx=Nx2, Ny=Ny2, domain_x=Lx2, domain_y=Ly2)
mesh3 = Mesh(meshio_mesh2.points, meshio_mesh2.cells_dict[cell_type])
"""define problem"""
def fixed_location2(point):
    return np.isclose(point[0], 0., atol=1e-5)
    # return np.logical_or(np.logical_and(np.isclose(point[0], 0., atol=0.1*Lx/2+1e-5),np.isclose(point[1], 0., atol=0.1*Ly/2+1e-5)),
    #                      np.logical_and(np.isclose(point[0], Lx, atol=0.1*Lx/2+1e-5),np.isclose(point[1], 0., atol=0.1*Ly/2+1e-5)))
def load_location2(point):
    # return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0, atol=0.1 * Ly + 1e-5))
    return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], Ly/2, atol=0.1 * Ly/2 + 1e-5))
    # return  np.logical_and(np.isclose(point[0], Lx/2, atol=0.1*Lx+1e-5),
    #                        np.isclose(point[1], Ly, atol=0.1*Ly+1e-5))
def dirichlet_val2(point):
    return 0.
dirichlet_bc_info2 = [[fixed_location2] * 2, [0, 1], [dirichlet_val2] * 2]
location_fns2 = [load_location2]

logging.info("Calculating stress problem")
problem3 = Elasticity(mesh3, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info2,
                      location_fns=location_fns2)
problem3.set_rho(rho.ravel()[:,None])
sol_list = solver(problem3, solver_options={'umfpack_solver': {}})
principal_stress, principal_directions=problem3.compute_first_principal_stress(sol_list[0])

max_principal_stress = principal_stress.max(axis=1)  # (num_cells,)，每个单元的最大主应力
max_principal_directions = principal_directions[np.arange(principal_directions.shape[0]),
                                               np.argmax(principal_stress, axis=1), :]  # (num_cells, 2)


logging.info(f"Principal stress.shape: {principal_stress.shape}")
logging.info(f"Principal directions.shape: {principal_directions.shape}")
plt.clf()
plt.imshow(max_principal_stress.reshape(Nx2,Ny2),cmap="rainbow")
plt.colorbar()
plt.draw()
plt.savefig(f'block/7_max_principal_stress.png', dpi=600, bbox_inches='tight')

# array_start_point,array_direction,_=output_sol3(sol_list,0,max_principal_stress,point_direction,"first_principal_stress")
# fwd_pred2 = ad_wrapper(problem2, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})
"""""""""""""""""""""""""""""""""""""""""where max stress at"""""""""""""""""""""""""""""""""""""""""
sites_ori=sites
max_add_num=350 #150 200
# max_principal_stress.reshape(Nx2,Ny2)
start_x=[int(Nx2/2)-15,int(Nx2*1/4)-10]
end_x=[int(Nx2/2)+5,int(Nx2*1/4)+10]
indices=np.array([]).astype(int)
for i in range(len(start_x)):
    start=start_x[i]
    end=end_x[i]
    local_stress=max_principal_stress[start*Ny2:end*Ny2+1]
    local_directions=max_principal_directions[start*Ny2:end*Ny2+1]
    indi = np.argsort(local_stress, axis=None)[int(-1*max_add_num):]
    indi = indi+start*Ny2
    indices=np.concatenate((indices, indi),axis=0)
# indices = np.argsort(max_principal_stress, axis=None)[int(-1*max_add_num):] #最大第一主应力所在的单元索引
# min_indices = np.argsort(max_principal_stress, axis=None)[:int(max_add_num)] #最大第一主应力所在的单元索引
# indices=np.concatenate((indices, min_indices),axis=0)
# rows, cols = np.unravel_index(indices, max_principal_stress.shape)
arrow_start_points = problem3.fe.points[problem3.fe.cells]
max_stress_position=arrow_start_points[indices,:,:]
max_stress_position=np.mean(max_stress_position,axis=1)
max_stress_direction=max_principal_directions[indices,:]


max_stress_position,max_stress_direction=ut.remove_nearby_points(max_stress_position,max_stress_direction,threshold=resolution2*25) #15 18 15

max_stress_direction=np.stack((max_stress_direction[:,1]*-1,max_stress_direction[:,0]),axis=1) # vertical vector
logging.info(f"max_stress_position.shape: {max_stress_position.shape}")
print(f"max_stress_position.shape:{max_stress_position.shape[0]}")

cp_ori=sites_ori.copy()
cp=np.concatenate((cp_ori,max_stress_position+max_stress_direction*10),axis=0)
sites=np.concatenate((sites_ori,max_stress_position),axis=0)
Dm3=np.tile(np.array(([0.5,0],[0,0.5]))/resolution2,reps=(max_stress_position.shape[0],1,1)) #0.6
Dm= np.concatenate((Dm,Dm3),axis=0) #0.9

rho = voronoi_field(coordinates, sites,rho_cell_mm, Dm=Dm,cp=cp).reshape(Nx2,Ny2)
rho = heaviside_projection(rho, eta=0.5, epoch=400)
plt.imshow(rho, cmap='viridis')  # 使用 'viridis' 颜色映射
plt.colorbar(label='Pixel Value')  # 添加颜色条用于显示值的范围
# plt.title("Pixel Values Visualized with Colors")
plt.imsave(f"block/8_VD3_np_img.png", rho, cmap='viridis')
plt.savefig(f"block/9_VD3_np.png")
plt.draw()
plt.scatter(sites[:, 1] // resolution2, sites[:, 0] // resolution2, marker='+', color='r')
plt.scatter(sites_ori[:, 1] // resolution2, sites_ori[:, 0] // resolution2, marker='+', color='violet')
plt.savefig(f"block/10_VD3.png")
plt.draw()
shadow=np.ones_like(rho)
for i in range(len(start_x)):
    start=start_x[i]
    end=end_x[i]
    shadow=shadow.at[start:end,:].set(2)
rho_s=rho*shadow
plt.imshow(rho_s, cmap='viridis')
plt.savefig(f"block/local.png")
plt.draw()
print(f"mean:{np.mean(rho)}")