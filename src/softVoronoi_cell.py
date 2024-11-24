import time
from typing import Callable

import jax
import numpy as onp
import jax.numpy as np
from matplotlib import pyplot as plt
import tqdm

@jax.jit
def heaviside_projection(field, eta=0.5, epoch=0):
    gamma = 2 ** (epoch // 5)
    field = (np.tanh(gamma * eta) + np.tanh(gamma * (field - eta))) / (
                np.tanh(gamma * eta) + np.tanh(gamma * (1 - eta)))
    return field


def batch_softmax(matrices,**kwargs):  # 形状 (1, 100, 100)
    exp_matrices = np.exp(-1 * matrices)  # (2,100,100)
    if "etas" in kwargs:
        if kwargs["etas"] is not None:
            s0=np.full_like(exp_matrices[0,:,:], kwargs["etas"],)
            exp_matrices=np.concatenate((exp_matrices, s0[None,:]), axis=0)
    sum_vals = np.sum(exp_matrices, axis=0, keepdims=True)  # 形状 (1, 100, 100)
    soft = exp_matrices / sum_vals
    return soft


def relu(x, y=0.):
    return np.maximum(y, x)


def d_euclidean(x, xm,*args):
    """
    euclidean distance
    :param x:
    :param xm:
    :return:
    """
    diff = x[np.newaxis, :, :, :] - xm[:, np.newaxis, np.newaxis, :]  # Nc*n*n*dim
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))  # Nc*n*n
    return dist_matrix


def d_mahalanobis(x, xm, Dm):
    """
    mahalanobis distance
    :param x:
    :param xm:
    :return:
    """
    diff = x[np.newaxis, :, :, np.newaxis, :] - xm[:, np.newaxis, np.newaxis, np.newaxis, :]  # Nc*n*n*dim

    nor = np.einsum("ijklm,ijkml->ijk", np.einsum('ijklm,imn->ijkln', diff, Dm.swapaxes(1, 2)), np.einsum('imn,ijknl->ijkml', Dm, diff.swapaxes(-1, -2)))
    dist_matrix = np.sqrt(nor)  # Nc*n*n*dim Nc*dim*dim
    return dist_matrix

def normal_distribution(x,mu=0.,sigma=1.):
    return 1./(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def d_mahalanobis_masked(x, xm, xs,Dm):

    alot=1e-9 # avoid nan
    diff_xxm = x[np.newaxis, :, :, np.newaxis, :] - xm[:, np.newaxis, np.newaxis, np.newaxis, :]  # Nc*n*n*1*dim

    nor = np.einsum("ijklm,ijkml->ijk", np.einsum('ijklm,imn->ijkln', diff_xxm, Dm.swapaxes(1, 2)), np.einsum('imn,ijknl->ijkml', Dm, diff_xxm.swapaxes(-1, -2)))
    dist_matrix = np.sqrt(nor)+alot  # Nc*n*n


    diff_xmxs = xm[:,None,:] - xs[:,None,:] #N*1*dim


    dist_exmxs=np.linalg.norm(diff_xmxs, axis=-1)[:,None,None,:]+alot
    dist_exxm=np.linalg.norm(diff_xxm, axis=-1)+alot

    cos= np.abs(np.einsum("ijk,ilmkn->ilmjn", diff_xmxs, diff_xxm.swapaxes(-1,-2)).squeeze()/(dist_exmxs*dist_exxm).squeeze()) #Nc*n*n


    ##### peach
    sigma = 1. / 30 #1/100   1/3   1/30
    mu = 1
    scale = 1 # 1
    k = (1 / normal_distribution(mu, mu, sigma)) * scale
    cos = normal_distribution(cos, mu=mu, sigma=sigma) * k + 1



    cos_mask=1
    return (cos**cos_mask)*dist_matrix

def sigmoid(x):
    return 1/(1+np.exp(-x))



def d_euclidean_cell(cell, sites, *args):
    diff = sites[:, None, :] - cell[:, None, :]  # n*1*dim
    return (diff @ diff.swapaxes(1, 2)).squeeze()


def d_mahalanobis_cell(cell, sites, Dm, *args):
    Dm=Dm[0]
    diff = sites[:, None, :] - cell[None, None, :]  # n*1*dim
    dist_m_cell = np.sqrt((diff @ Dm.swapaxes(-1, -2) @ Dm @ diff.swapaxes(-1, -2))).squeeze()
    return dist_m_cell


def d_mahalanobis_masked_cell(cell, sites, Dm, cp, *args):
    alot = 1e-9  # avoid nan
    Dm = Dm[0]
    diff_sx = cell[None, None, :] - sites[:, None, :]  # N*1*dim
    dist_m_cell = np.sqrt((diff_sx @ Dm.swapaxes(-1, -2) @ Dm @ diff_sx.swapaxes(-1, -2))).squeeze()
    diff_sc = cp[:, None, :] - sites[:, None, :]  # N*1*dim
    dist_sc = np.linalg.norm(diff_sc, axis=-1).squeeze() + alot  # N
    dist_sx = np.linalg.norm(diff_sx, axis=-1).squeeze() + alot  # N
    cos = np.abs((diff_sc @ diff_sx.swapaxes(-1, -2)) / (dist_sx * dist_sc)).squeeze()  # N
    sigma = 1. / 30  # 1/100   1/3   1/30
    mu = 1
    scale = 1  # 1
    k = (1 / normal_distribution(mu, mu, sigma)) * scale
    cos = normal_distribution(cos, mu=mu, sigma=sigma) * k + 1
    cos_mask = 1
    return (cos ** cos_mask) * dist_m_cell

@jax.jit
def rho_cell_mm(cell, sites, *args):
    dist_f = d_mahalanobis_masked_cell(cell,sites,*args)  # N
    # etas = np.array([1e-50])
    # dist = np.concatenate((dist_f, etas), axis=0)
    dist=dist_f
    negative_dist= -1*dist
    exp_matrices = np.exp(negative_dist-np.max(negative_dist))  # N
    sum_vals = np.sum(exp_matrices, axis=0, keepdims=True)  # 1
    soft = exp_matrices / sum_vals  # N
    beta = 3  # 10 #5 razer 7
    rho = 1 - np.sum(soft ** beta, axis=0)
    return rho

@jax.jit
def rho_cell_m(cell, sites, *args):
    dist_f = d_mahalanobis_cell(cell,sites,*args)  # N
    # etas = np.array([1e-50])
    # dist = np.concatenate((dist_f, etas), axis=0)
    dist=dist_f
    negative_dist= -1*dist
    exp_matrices = np.exp(negative_dist-np.max(negative_dist))  # N
    sum_vals = np.sum(exp_matrices, axis=0, keepdims=True)  # 1
    soft = exp_matrices / sum_vals  # N
    beta = 3  # 10 #5 razer 7
    rho = 1 - np.sum(soft ** beta, axis=0)
    return rho


def voronoi_field(field, sites, rho_fn: Callable, **kwargs):
    assert field.shape[-1] == 2
    cell = field.reshape(-1, 2)  # cell_num * 2

    def calc_rho(cell, sites, *args):
        return rho_fn(cell, sites, *args)

    # 检测可用设备
    devices = jax.devices()
    print(f"devices:{devices}")
    num_devices = len(devices)
    if num_devices == 1:
        # 使用 vmap 和 jit 加速计算
        calc_rho_batch = jax.jit(jax.vmap(calc_rho, in_axes=(0, None,None)))
        # 分块处理，避免内存占用过高
        batch_size = kwargs.get('batch_size', 100)  # 支持通过 kwargs 控制 batch_size，默认为 100
        rho_list = []
        for i in tqdm.tqdm(range(0, cell.shape[0], batch_size), desc="Calculating Voronoi Diagram"):
            batch_cells = jax.device_put(cell[i:i + batch_size])  # 将当前批次放到设备上
            rho_batch = calc_rho_batch(batch_cells, sites,tuple(kwargs.values()))  # 批量计算
            rho_list.append(rho_batch)
    else:
        # 使用 pmap 实现多设备并行计算
        calc_rho_pmap = jax.pmap(
            jax.vmap(calc_rho, in_axes=(0, None, None)),  # 设备内批量处理
            in_axes=(0, None, None)  # 设备间数据分片
        )
        # 批次大小调整为设备数的倍数
        batch_size = kwargs.get('batch_size', 100)
        effective_batch_size = batch_size * num_devices
        rho_list = []
        # 按批次处理数据
        for i in tqdm.tqdm(range(0, cell.shape[0], effective_batch_size), desc="Calculating Voronoi Diagram"):
            batch_cells = cell[i:i + effective_batch_size]
            # 将数据分割为设备数量的子批次
            sub_batches = np.array_split(batch_cells, num_devices)
            # 填充不足的子批次，保持一致长度
            while len(sub_batches) < num_devices:
                sub_batches.append(np.zeros_like(sub_batches[0]))
            # 转为 JAX 数组并放置在对应设备上
            sub_batches = jax.device_put(np.array(sub_batches))
            # 使用 pmap 并行计算
            rho_batch = calc_rho_pmap(sub_batches, sites, tuple(kwargs.values()))
            # 收集计算结果，合并所有设备的输出
            rho_list.append(rho_batch.reshape(-1, rho_batch.shape[-1]))

    # 合并所有批次结果
    rho = np.concatenate(rho_list, axis=0)
    return rho.reshape(field.shape[:-1])


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

    coordinates = np.stack(coordinates, axis=-1)
    if "cp" in para and para["control"]:
        cp = para["cp"] if "cp" in para else (
            p[sites_len + Dm_len:sites_len + Dm_len + c_len].reshape(shapes[2][0], shapes[2][1]))
        field = voronoi_field(coordinates, sites,rho_cell_mm, Dm=Dm, cp=cp)
    else:
        field = voronoi_field(coordinates, sites,rho_cell_m, Dm=Dm)

    if "heaviside" in para and para["heaviside"] is True:
        field = heaviside_projection(field, eta=0.5, epoch=kwargs['epoch'])

    return field.reshape(para["Nx"],para["Ny"])


def generate_para_rho(para, rho_p, **kwargs):
    # generate seed
    rho=rho_p
    rho=rho.reshape(para["Nx"],para["Ny"])
    key = jax.random.PRNGKey(0)
    random_numbers = jax.random.uniform(key, shape=rho.shape, minval=0.00, maxval=100.00)
    # rho*float + x determines the point generation rate.
    void=0.05
    entity=3.0
    sites = np.argwhere(random_numbers < (rho*(entity-void))+void )*para["resolution"]

    para["sites_num"]=sites.shape[0]
    move_around=30 # seed movement
    sites_low = sites.ravel()-move_around*para["resolution"]
    sites_up = sites.ravel()+move_around*para["resolution"]
    Dm = np.tile(np.array(([30, 0], [0, 30])), (sites.shape[0], 1, 1))  # Nc*dim*dim
    Dm_low = np.tile(np.array([[0, 0], [0, 0]]), (sites_low.shape[0], 1, 1))
    Dm_up = np.tile(np.array([[100, 100], [100, 100]]), (sites_low.shape[0], 1, 1))
    cp = sites.copy()
    cp_low = sites_low
    cp_up = sites_up
    para["bound_low"] = np.concatenate((np.ravel(sites_low), np.ravel(Dm_low), np.ravel(cp_low)), axis=0)[:, None]
    para["bound_up"] = np.concatenate((np.ravel(sites_up), np.ravel(Dm_up), np.ravel(cp_up)), axis=0)[:, None]
    para["paras_at"] = (0, para["sites_num"]*6)
    # p = np.concatenate((sites.ravel(), Dm.ravel(),cp.ravel()))
    p = np.concatenate((sites.ravel(), Dm.ravel()))
    print("seeds:",sites.shape[0])
    return p,para


if __name__ == '__main__':
    start_time = time.time()  # 计时起点
    print(f"running！")
    Nx, Ny = 1000, 1000
    resolution = 0.03
    x_len = Nx * resolution
    y_len = Ny * resolution
    coords = np.indices((Nx, Ny)) * resolution
    coordinates = np.stack(coords, axis=-1)
    cauchy_field = coordinates.copy()
    ###### coordination
    sites=np.array(([5,5],[3,5]))
    cp=np.array(([4,2],[5,6]))

    # sites = np.random.randint(low=0 - 20, high=Nx + 20, size=(20, 2)) * resolution
    # cp = sites.copy()
    # cp = cp + np.random.normal(loc=0, scale=5, size=cp.shape)

    Dm = np.tile(np.array(([1, 0], [0, 1])), (sites.shape[0], 1, 1))  # Nc*dim*dim
    Dm=Dm.at[0].set(np.array(([1, 0], [0, 1])))

    # dist_field=voronoi_field(coordinates, sites, Dm=Dm, sigmoid_sites=sigmoid_sites, sigmoid_field=sigmoid_field)
    # field=voronoi_field(coordinates, sites, Dm=Dm)
    field = voronoi_field(coordinates, sites,rho_cell_m, Dm=Dm).reshape(Nx,Ny)
    field=heaviside_projection(field, eta=0.5, epoch=120)

    print(f"algorithm use ：{time.time() - start_time:.6f} 秒")

    plt.imshow(field, cmap='viridis')  # 使用 'viridis' 颜色映射
    plt.colorbar(label='Pixel Value')  # 添加颜色条用于显示值的范围
    plt.title("Pixel Values Visualized with Colors")
    plt.scatter(sites[:, 1]//resolution, sites[:, 0]//resolution, marker='^', color='r')
    print(f"total used：{time.time() - start_time:.6f} 秒")
    plt.show()
