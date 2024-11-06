import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv

def heaviside_projection(field, eta=0.5, epoch=0):
    gamma = 2 ** (epoch // 50)
    field = (np.tanh(gamma * eta) + np.tanh(gamma * (field - eta))) / (
                np.tanh(gamma * eta) + np.tanh(gamma * (1 - eta)))
    return field


def batch_softmax(matrices,**kwargs):  # 形状 (1, 100, 100)
    k=kwargs['k'] if 'k' in kwargs.keys() else 1
    exp_matrices = np.exp(-k * matrices)  # (2,100,100)
    if "etas" in kwargs:
        if kwargs["etas"] is not None:
            s0=np.full_like(exp_matrices[0,:,:], kwargs["etas"],)
            exp_matrices=np.concatenate((exp_matrices, s0[None,:]), axis=0)
    sum_vals = np.sum(exp_matrices, axis=0, keepdims=True)  # 形状 (1, 100, 100)
    soft = exp_matrices / sum_vals
    return soft


def relu(x, y=0.):
    return np.maximum(y, x)


def d_euclidean(x, xm):
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
    diff_xxm = x[np.newaxis, :, :, np.newaxis, :] - xm[:, np.newaxis, np.newaxis, np.newaxis, :]  # Nc*n*n*dim
    dot1 = np.einsum('ijklm,imn->ijkln', diff_xxm, Dm.swapaxes(1, 2))
    dot2 = np.einsum('imn,ijknl->ijkml', Dm, diff_xxm.swapaxes(-1, -2))
    nor = np.einsum("ijklm,ijkml->ijk", dot1, dot2)
    dist_matrix = np.sqrt(nor)  # Nc*n*n*dim Nc*dim*dim
    return dist_matrix

def d_mahalanobis_masked(x, xm, xs,Dm):
    """
    mahalanobis distance
    :param xs: additional seed points Ns*dim
    :param x: coordinate field m*n*dim
    :param xm: voronoi sites Nm*dim
    :return:
    """
    alot=1e-9
    diff_xxm = x[np.newaxis, :, :, np.newaxis, :] - xm[:, np.newaxis, np.newaxis, np.newaxis, :]  # Nc*n*n*1*dim
    dot1 = np.einsum('ijklm,imn->ijkln', diff_xxm, Dm.swapaxes(1, 2))
    dot2 = np.einsum('imn,ijknl->ijkml', Dm, diff_xxm.swapaxes(-1, -2))
    nor = np.einsum("ijklm,ijkml->ijk", dot1, dot2)
    dist_matrix = np.sqrt(nor)+alot  # Nc*n*n

    diff_xmxs = xm[:,None,:] - xs[:,None,:] #N*1*dim
    dot1 = np.einsum('ijk,ikl->ijl', diff_xmxs, Dm.swapaxes(1, 2))
    dot2 = np.einsum('ijk,ikl->ijl', Dm, diff_xmxs.swapaxes(-1, -2))
    nor = np.einsum("ijk,ikl->ijl", dot1, dot2)
    dist_xmxs = (np.sqrt(nor)+alot)  # Nc*1*1
    cos= np.abs(np.einsum("ijk,ilmkn->ilmjn", diff_xmxs, diff_xxm.swapaxes(-1,-2)).squeeze()/(dist_xmxs*dist_matrix)) #Nc*n*n
    sigma = 1. / 100
    mu = 1
    k = 1 / normal_distribution(mu, mu, sigma)*0.5
    cos=normal_distribution(cos,mu=mu,sigma=sigma)*k+1
    return cos*dist_matrix



def cauchy_distribution(x, **kwargs):
    x0 = kwargs['x0'] if 'x0' in kwargs.keys() else 0
    gamma = kwargs['gamma'] if 'gamma' in kwargs.keys() else 1
    scale = kwargs['scale'] if 'scale' in kwargs.keys() else 1
    cauchy = (1. * scale / np.pi) * (gamma / ((x - x0) ** 2 + gamma ** 2))
    return cauchy

def normal_distribution(x,mu=0.,sigma=1.):
    return 1./(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))


def cauchy_mask(dist_field, point: np.array, mask_field,gamma=5,scale=10,**kwargs):

    if "Dm" in kwargs:
        mask_field=d_mahalanobis(mask_field, point, kwargs['Dm'])
    else:
        mask_field = d_euclidean(mask_field, point)  # Ns*n*n
    cauchy = cauchy_distribution(mask_field, gamma=gamma, scale=scale)
    assert cauchy.shape == dist_field.shape
    masked = dist_field[:, :, :] * (cauchy + 1)[:, :, :]  # Nc*n*n , Ns*n*n ,Nc==Ns-> Nc*n*n
    return masked  # Nc*n*n


def voronoi_field(field, sites, **kwargs):
    if "Dm" in kwargs:
        dist = d_mahalanobis(field, sites, kwargs["Dm"])
    else:
        dist = d_euclidean(field, sites)  # Nc,r,c
    if "quadratic" in kwargs and kwargs["quadratic"]:
        dist=quadratic(dist)
    if "cauchy_field" in kwargs and "cauchy_points" in kwargs:
        if "Dm" in kwargs:
            Dm_inv = np.array([np.linalg.inv(kwargs["Dm"][i]) for i in range(kwargs["Dm"].shape[0])])
            dist = cauchy_mask(dist, kwargs["cauchy_points"], kwargs["cauchy_field"],Dm=Dm_inv)
        else:
            dist = cauchy_mask(dist, kwargs["cauchy_points"], kwargs["cauchy_field"])

    soft = batch_softmax(dist,etas=kwargs["etas"] if "etas" in kwargs.keys() else None)
    beta = 1
    rho = 1 - np.sum(soft ** beta, axis=0)
    return rho


def generate_voronoi(para, p: np.array, **kwargs):
    """

    :param para:
    :param p: a params 1-d array, can be divided into sites,Dm,cauchy_points.
    :return:
    """
    coordinates = para["coordinates"]
    sites_num = para["sites_num"]
    Dm_dim = para["Dm_dim"]

    shapes = [(sites_num, Dm_dim), (sites_num, Dm_dim, Dm_dim), (sites_num, Dm_dim)]
    sites_len = (shapes[0][0] * shapes[0][1])
    Dm_len = shapes[1][0] * shapes[1][1] * shapes[1][2]
    cauchy_len = shapes[2][0] * shapes[2][1]

    if p.shape[0] == 1 and p.shape[1] != 1:
        p = p[0]
    sites = p[0:sites_len].reshape(shapes[0][0], shapes[0][1])
    Dm = p[sites_len:sites_len + Dm_len].reshape(shapes[1][0], shapes[1][1], shapes[1][2])

    if "cauchy" in para and para["cauchy"]:
        cauchy_points = p[sites_len + Dm_len:sites_len + Dm_len + cauchy_len].reshape(shapes[2][0], shapes[2][1])
        coordinates = np.stack(coordinates, axis=-1)
        cauchy_field = coordinates.copy()
        field = voronoi_field(coordinates, sites, Dm=Dm, cauchy_field=cauchy_field, cauchy_points=cauchy_points)
    else:
        field = voronoi_field(coordinates, sites, Dm=Dm)

    if "heaviside" in para and para["heaviside"] is True:
        field = heaviside_projection(field, eta=0.5, epoch=kwargs['epoch'])
    return field


def generate_voronoi_separate(para, p, **kwargs):
    coordinates = para["coordinates"]
    sites_num = para["sites_num"]
    Dm_dim = para["Dm_dim"]

    shapes = [(sites_num, Dm_dim), (sites_num, Dm_dim, Dm_dim), (sites_num, Dm_dim)]
    sites_len = (shapes[0][0] * shapes[0][1]) if "sites" not in para else 0
    Dm_len = shapes[1][0] * shapes[1][1] * shapes[1][2] if "Dm" not in para else 0
    cauchy_len = shapes[2][0] * shapes[2][1] if "cauchy_points" not in para else 0

    if p.shape[0] == 1 and p.shape[1] != 1:
        p = p[0]
    sites = para["sites"] if "sites" in para else p[0:sites_len].reshape(shapes[0][0], shapes[0][1])
    Dm = para["Dm"] if "Dm" in para else p[sites_len:sites_len + Dm_len].reshape(shapes[1][0], shapes[1][1],
                                                                                    shapes[1][2])

    coordinates = np.stack(coordinates, axis=-1)
    if "cauchy" in para and para["cauchy"]:
        cauchy_points = para["cauchy_points"] if "cauchy_points" in para else (
            p[sites_len + Dm_len:sites_len + Dm_len + cauchy_len].reshape(shapes[2][0], shapes[2][1]))
        cauchy_field = coordinates.copy()
        field = voronoi_field(coordinates, sites, Dm=Dm, cauchy_field=cauchy_field, cauchy_points=cauchy_points,etas=kwargs['etas'] if 'etas' in kwargs.keys() else None)
    else:
        field = voronoi_field(coordinates, sites, Dm=Dm,etas=kwargs['etas'] if 'etas' in kwargs.keys() else None)

    if "heaviside" in para and para["heaviside"] is True:
        field = heaviside_projection(field, eta=0.5, epoch=kwargs['epoch'])
    # plt.imshow(field, cmap='viridis')  # 使用 'viridis' 颜色映射
    # plt.colorbar(label='Pixel Value')  # 添加颜色条用于显示值的范围
    # plt.scatter(sites[:,1],sites[:,0],c='r')
    # plt.title("Pixel Values Visualized with Colors")
    # plt.show()
    return field


def generate_gene_random(op, Nx, Ny) -> np.ndarray:
    sites_num = op["sites_num"]
    margin = op["margin"]
    sites = np.ones((sites_num, 2))
    sites[:, 0] = np.random.randint(low=0 - margin, high=Nx + margin, size=sites_num)
    sites[:, 1] = np.random.randint(low=0 - margin, high=Ny + margin, size=sites_num)
    Dm = np.tile(np.array(([1, 0], [0, 1])), (sites.shape[0], 1, 1))  # Nc*dim*dim
    cauchy_points = sites.copy()
    p = np.concatenate((np.ravel(sites), np.ravel(Dm), np.ravel(cauchy_points)),
                       axis=0)  # 1-d array contains flattened: sites,Dm,cauchy points
    return p


if __name__ == '__main__':
    start_time = time.time()  # 计时起点
    x_len = 100
    y_len = 100
    coords = np.indices((x_len, y_len))
    coordinates = np.stack(coords, axis=-1)
    cauchy_field = coordinates.copy()

    # sites=np.array(([30,50],[80,50],))
    # cauchy_points=np.array(([70,30],[90,50]))
    np.random.seed(0)
    sites = np.random.randint(low=0, high=100, size=(40, 2))
    cauchy_points = sites.copy()
    cauchy_points = cauchy_points+cauchy_points *np.random.normal(loc=0, scale=1, size=cauchy_points.shape)

    Dm = np.tile(np.array(([1, 0], [0, 1])), (sites.shape[0], 1, 1))  # Nc*dim*dim
    Dm[0] = np.array(([1, 0], [0, 1]))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    plotter = pv.Plotter()
    combined_mesh = pv.UnstructuredGrid()
    X = np.arange(coordinates.shape[0])
    Y = np.arange(coordinates.shape[1])
    X, Y = np.meshgrid(X, Y)

    dist = d_mahalanobis_masked(coordinates, sites, cauchy_points,Dm)
    # d0=np.full_like(dist[0,:,:],15)
    # dist=np.concatenate((dist,d0[None,:]),axis=0)
    Dm_inv = np.array([np.linalg.inv(Dm[i]) for i in range(Dm.shape[0])])


    # dist = cauchy_mask(dist, cauchy_points, cauchy_field,gamma=5,scale=3,Dm=Dm_inv)

    soft = batch_softmax(dist,etas=None,k=1)

    mask_field=coordinates
    mask_field = d_euclidean(mask_field, cauchy_points)  # Ns*n*n
    cauchy = cauchy_distribution(mask_field, gamma=10, scale=np.pi*10)
    # soft=soft*(cauchy+1)
    beta = 3
    rho = np.sum(soft ** beta, axis=0)
    # rho=heaviside_projection(rho,eta=0.5, epoch=100)



    for i in range(len(dist)):
        di=dist[i]
        points = np.vstack((X.ravel(), Y.ravel(), di.ravel())).T
        grid = pv.StructuredGrid(X, Y, di)
        combined_mesh = combined_mesh.merge(grid)
        plotter.add_mesh(grid, scalars=di.ravel(), cmap='viridis')
    grid_rho = pv.StructuredGrid(X, Y, rho*10-20)
    plotter.add_mesh(grid_rho, scalars=rho.ravel(), cmap='Greys',opacity=0.9)

    print(f"代码运行时间：{time.time() - start_time:.6f} 秒")
    combined_mesh = combined_mesh.merge(grid_rho)
    combined_mesh.save("combine.vtk")
    plotter.add_axes()
    plotter.show()
    # plt.show()

