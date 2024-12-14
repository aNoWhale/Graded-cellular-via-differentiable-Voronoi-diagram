import time
from typing import Callable

import jax
import numpy as np
# import jax.numpy as np
from matplotlib import pyplot as plt
import tqdm

from src import ultilies


def heaviside_projection(field, eta=0.5, epoch=0):
    gamma = 2 ** (epoch // 50)
    field = (np.tanh(gamma * eta) + np.tanh(gamma * (field - eta))) / (
                np.tanh(gamma * eta) + np.tanh(gamma * (1 - eta)))
    return field





def normal_distribution(x,mu=0.,sigma=1.):
    return 1./(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))


def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_euclidean_cell(cell, sites, *args):
    diff = sites[:, None, :] - cell[:, None, :]  # n*1*dim
    return (diff @ diff.swapaxes(1, 2)).squeeze()


def d_mahalanobis_cell(cell, sites, Dm, *args):
    Dm=Dm[0]
    diff = sites[:, None, :] - cell[None, None, :]  # n*1*dim
    dist_m_cell = np.sqrt((diff @ Dm.swapaxes(1, 2) @ Dm @ diff.swapaxes(1, 2))).squeeze()
    return dist_m_cell


def d_mahalanobis_masked_cell(cell, sites, Dm, cp, *args):
    alot = 1e-9  # avoid nan
    # Dm = Dm[0]
    Dm_inv=args[0]
    diff_sx = cell[None, None, :] - sites[:, None, :]  # N*1*dim
    dist_m_cell = np.sqrt((diff_sx @ Dm.swapaxes(1, 2) @ Dm @ diff_sx.swapaxes(1, 2))).squeeze()
    # dist_m_sx_inv = np.sqrt((diff_sx @ Dm_inv.swapaxes(1, 2) @ Dm_inv @ diff_sx.swapaxes(1, 2))).squeeze()
    diff_sc = cp[:, None, :] - sites[:, None, :]  # N*1*dim
    dist_sc = np.linalg.norm(diff_sc, axis=-1).squeeze() + alot  # N
    # dist_m_sc_inv = np.sqrt((diff_sc @ Dm_inv.swapaxes(1, 2) @ Dm_inv @ diff_sc.swapaxes(1, 2))).squeeze() + alot  # N
    dist_sx = np.linalg.norm(diff_sx, axis=-1).squeeze() + alot  # N
    cos = np.abs((diff_sc @ diff_sx.swapaxes(-1, -2)).squeeze() / (dist_sx * dist_sc)).squeeze()  # N
    sigma = 1. / 30  # 1/100   1/3   1/30
    mu = 1
    kr=1
    # mushroom = kr * np.power(dist_sx,0.5) + 1
    mushroom = 1
    k = (1 / normal_distribution(mu, mu, sigma)) * mushroom
    cos = normal_distribution(cos, mu=mu, sigma=sigma) * k + 1
    cos_mask = sigmoid(0.1*(dist_sc-dist_sx))
    return (cos ** cos_mask) * dist_m_cell


def rho_cell_mm(cell, sites, *args):
    dist_f = d_mahalanobis_masked_cell(cell,sites,*args)  # N
    # etas = np.array([1e-20])
    # dist = np.concatenate((dist_f, etas), axis=0)
    dist=dist_f
    negative_dist= -1*dist
    exp_matrices = np.exp(negative_dist-np.max(negative_dist))  # N
    # exp_matrices = np.exp(np.concatenate((negative_dist,np.array([-30]))))  # N
    sum_vals = np.sum(exp_matrices, axis=0, keepdims=True)  # 1
    soft = exp_matrices / sum_vals  # N
    beta = 5  # 10 #5 razer 7
    rho = 1 - np.sum(soft ** beta, axis=0)
    return rho


def rho_cell_m(cell, sites, *args):
    dist_f = d_mahalanobis_cell(cell,sites,*args)  # N
    # etas = np.array([1e-10])
    # dist_f = np.concatenate((dist_f, etas), axis=0)
    dist=dist_f
    negative_dist= -1*dist
    exp_matrices = np.exp(negative_dist-np.max(negative_dist))  # N
    sum_vals = np.sum(exp_matrices, axis=0, keepdims=True)  # 1
    soft = exp_matrices / sum_vals  # N
    beta = 5  # 10 #5 razer 7
    rho = 1 - np.sum(soft ** beta, axis=0)
    return rho


def voronoi_field(field, sites, rho_fn: Callable, **kwargs):
    assert field.shape[-1] == 2
    cell = field.reshape(-1, 2)  # cell_num * 2
    calc_rho=lambda cell,sites: rho_fn(cell, sites, *kwargs.values())
    rh=[]
    for i in tqdm.tqdm(range(0,cell.shape[0])):
        rh.append(calc_rho(cell[i,:], sites))
    # rh = jax.vmap(calc_rho, in_axes=(0, None))(cell, sites)
    return np.array(rh)




def generate_voronoi_separate(para, p, **kwargs):
    ###interprate p and para into design variables
    coordinates = para["coordinates"]
    sites_num = para["sites_num"]
    dim = para["dim"]
    shapes = [(sites_num, dim), (sites_num, dim, dim), (sites_num, dim)]
    sites_len = (shapes[0][0] * shapes[0][1]) if "sites" not in para else 0
    Dm_len = shapes[1][0] * shapes[1][1] * shapes[1][2] if "Dm" not in para else 0
    c_len = shapes[2][0] * shapes[2][1] if "cp" not in para else 0
    if p.shape[0] == 1 and p.shape[1] != 1:
        p = p[0]
    sites = para["sites"] if "sites" in para else p[0:sites_len].reshape(shapes[0][0], shapes[0][1])
    Dm = para["Dm"] if "Dm" in para else p[sites_len:sites_len + Dm_len].reshape(shapes[1][0], shapes[1][1],
                                                                                    shapes[1][2])
    Dm_inv = ultilies.inv_2d(Dm)

    coordinates = np.stack(coordinates, axis=-1)
    if "cp" in para or para["control"]:
        cp = para["cp"] if "cp" in para else (
            p[sites_len + Dm_len:sites_len + Dm_len + c_len].reshape(shapes[2][0], shapes[2][1]))
        field = voronoi_field(coordinates, sites,rho_cell_mm, Dm=Dm, cp=cp,Dm_inv=Dm_inv)
    else:
        field = voronoi_field(coordinates, sites,rho_cell_m, Dm=Dm)

    if "heaviside" in para and para["heaviside"] is True:
        field = heaviside_projection(field, eta=0.5, epoch=kwargs['epoch'])

    return field.reshape(para["Nx"],para["Ny"])




if __name__ == '__main__':
    start_time = time.time()  # 计时起点
    np.random.seed(0)
    print(f"running！")
    Nx,Ny=500,500
    resolution=0.1
    x_len = Nx*resolution
    y_len = Ny*resolution
    coords = np.indices((Nx, Ny))*resolution
    coordinates = np.stack(coords, axis=-1)
    cauchy_field = coordinates.copy()
    # coordination
    # sites=np.array(([20,50],[80,50]))*reso
    # cp=np.array(([50,50],[80,50]))*reso

    sites_x = np.random.randint(low=0, high=Nx, size=(30, 1))*resolution
    sites_y = np.random.randint(low=0, high=Ny, size=(30, 1))*resolution
    sites=np.concatenate((sites_x, sites_y), axis=-1)
    cp = sites.copy()
    cp = cp + np.random.normal(loc=0, scale=5, size=cp.shape)

    Dm = np.tile(np.array(([5, 5.], [0., 5])), (sites.shape[0], 1, 1))  # Nc*dim*dim
    # Dm =Dm + np.random.normal(loc=-0.5, scale=0.5, size=Dm.shape)
    Dm_inv = ultilies.inv_2d(Dm)
    # Dm[0] = np.array(([1, 0], [0, 1]))
    # Dm[1] = np.array(([1, 0], [0, 1]))


    # dist_field=voronoi_field(coordinates, sites, Dm=Dm, sigmoid_sites=sigmoid_sites, sigmoid_field=sigmoid_field)
    # field=voronoi_field(coordinates, sites, Dm=Dm)
    # field = voronoi_field(coordinates, sites,rho_cell_mm, Dm=Dm, cp=cp)
    field = voronoi_field(coordinates, sites,rho_cell_mm, Dm=Dm,cp=cp,Dm_inv = Dm_inv).reshape(Nx,Ny)
    field=heaviside_projection(field, eta=0.5, epoch=120)

    print(f"algorithm use ：{time.time() - start_time:.6f} 秒")
    print(f"max.field:{np.max(field)}")
    plt.imshow(field, cmap='viridis')  # 使用 'viridis' 颜色映射
    plt.colorbar(label='Pixel Value')  # 添加颜色条用于显示值的范围
    # plt.title("Pixel Values Visualized with Colors")
    plt.scatter(sites[:, 1]//resolution, sites[:, 0]//resolution, marker='+', color='r')
    print(f"total used：{time.time() - start_time:.6f} 秒")
    plt.show()
