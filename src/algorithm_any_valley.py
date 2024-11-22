import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from scipy.special import softmax


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

    diff_xmxs = xm[:,None,None,:] - xs[:,:,None,:] #N*Ncau*1*dim
    # dot1 = np.einsum('ijk,ikl->ijl', diff_xmxs, Dm.swapaxes(1, 2))
    # dot2 = np.einsum('ijk,ikl->ijl', Dm, diff_xmxs.swapaxes(-1, -2))
    # nor = np.einsum("ijk,ikl->ijl", dot1, dot2)
    # dist_xmxs = (np.sqrt(nor)+alot)  # Nc*1*1
    dist_exmxs=np.linalg.norm(diff_xmxs, axis=-1)[:,:,None,None,:]+alot
    dist_exxm=np.linalg.norm(diff_xxm, axis=-1)[:,None,:,:,:]+alot
    # cos= np.abs(np.einsum("ijk,ilmkn->ilmjn", diff_xmxs, diff_xxm.swapaxes(-1,-2)).squeeze()/(dist_xmxs*dist_matrix)) #Nc*n*n
    upper=np.einsum("ijkl,imnlk->ijmnk", diff_xmxs, diff_xxm.swapaxes(-1,-2))
    lower=dist_exmxs*dist_exxm

    cos= (((upper/lower+1)/2.)) # \in 0~1
    sigma = 1. / 30
    mu = 1
    k = (1 / normal_distribution(mu, mu, sigma))
    cos=normal_distribution(cos,mu=mu,sigma=sigma)*k+1 #\in 1~2
    cos= np.prod(cos,axis=1)-1 # \in 0~?
    cos= ((cos-np.min(cos))/(np.max(cos)-np.min(cos))).squeeze(axis=-1) # \in 0~1
    ##### 奥特曼形
    # sigma = 1. / 30
    # mu = 1
    # scale = 1
    # k = (1 / normal_distribution(mu, mu, sigma)) * scale
    # cos = normal_distribution(cos, mu=mu, sigma=sigma) * k * (-1) + scale + 1
    ##### 桃子
    sigma = 1. / 300
    mu = 1
    scale = 1
    k = (1 / normal_distribution(mu, mu, sigma)) * scale
    cos = normal_distribution(cos, mu=mu, sigma=sigma) * k + 1
    #### 正态range
    # sigma_mask = 20.
    # mu_mask = 0
    # scale_mask = 1
    # k_mask = (1 / normal_distribution(mu_mask, mu_mask, sigma_mask)) * scale_mask
    # cos_mask=normal_distribution(dist_matrix, mu_mask, sigma_mask)*k_mask
    #### sigmoid range
    # x0=20 #用于valley消失于多远
    # smooth=0.1
    cos_mask=1
    # cos_mask=sigmoid(-1*smooth*(dist_matrix-x0))
    return (cos**cos_mask)*dist_matrix


def sigmoid(x):
    return 1/(1+np.exp(-x))



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





if __name__ == '__main__':
    start_time = time.time()  # 计时起点
    x_len = 100
    y_len = 100
    coords = np.indices((x_len, y_len))
    coordinates = np.stack(coords, axis=-1)
    cauchy_field = coordinates.copy()

    sites=np.array(([30,50],[80,50],))
    cauchy_points=np.array(([[30,70],[60,50]],
                            [[50,40],[30,80]])) #Ns*Nc*dim
    # np.random.seed(0)
    # sites = np.random.randint(low=0, high=100, size=(40, 2))
    # cauchy_points = sites.copy()
    # cauchy_points = cauchy_points+cauchy_points *np.random.normal(loc=0, scale=1, size=cauchy_points.shape)

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

    # mask_field=coordinates
    # mask_field = d_euclidean(mask_field, cauchy_points)  # Ns*n*n
    # cauchy = cauchy_distribution(mask_field, gamma=10, scale=np.pi*10)
    # # soft=soft*(cauchy+1)
    beta = 5
    # rho = 1 - np.sum(soft ** beta, axis=0)
    rho = np.sum(soft ** beta, axis=0)
    rho=heaviside_projection(rho,eta=0.5, epoch=100)



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

