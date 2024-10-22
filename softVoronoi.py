import time
import numpy as np
from matplotlib import pyplot as plt


def batch_softmax(matrices):  # 形状 (1, 100, 100)
    exp_matrices = np.exp(-1 * matrices) #(2,100,100)
    sum_vals = np.sum(exp_matrices, axis=0, keepdims=True)  # 形状 (1, 100, 100)
    return exp_matrices / sum_vals

def relu(x,y=0.):
    return np.maximum(y, x)

def d_euclidean(x, xm):
    """
    euclidean distance
    :param x:
    :param xm:
    :return:
    """
    diff= x[np.newaxis, :, :, :] - xm[:, np.newaxis, np.newaxis, :] # Nc*n*n*dim
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1)) #Nc*n*n
    return dist_matrix

def d_mahalanobis(x, xm, Dm):
    """
    mahalanobis distance
    :param x:
    :param xm:
    :return:
    """
    diff= x[np.newaxis, :, :, :] - xm[:, np.newaxis, np.newaxis, :] # Nc*n*n*dim
    dot1=np.einsum('ijkl,ilm->ijkm', diff, Dm)
    dot2=np.einsum('ilm,ijkl->ijkl', Dm,diff)
    dist_matrix=np.sqrt(np.einsum("ijkl,ijkl->ijk",dot1,dot2)) # Nc*n*n*dim Nc*dim*dim
    return dist_matrix

def d_sigmoid(field, sites, **kwargs):

    def sigmoid_t(z,move=0):
        return 1 / (1 + np.exp(-z+move))
    dist=d_euclidean(field,sites) #Ns*n*n
    # dist=np.sum(dist_field[np.newaxis,:,:,:]-sites[:,np.newaxis,np.newaxis,:],axis=-1)#n*n*dim Ns*dim -> Ns*n*n*dim ->Ns*n*n
    dist_sig=dist*sigmoid_t(dist)*2 #Ns*n*n
    return dist_sig

def cauchy_mask(dist_field, point:np.array, mask_field):
    def cauchy_distribution(x,**kwargs):
        x0=kwargs['x0'] if 'x0' in kwargs.keys() else 0
        gamma=kwargs['gamma'] if 'gamma' in kwargs.keys() else 1
        scale=kwargs['scale'] if 'scale' in kwargs.keys() else 1
        cauchy=(1.*scale/np.pi)*(gamma/((x-x0)**2+gamma**2))
        return cauchy
    mask_field=d_euclidean(mask_field,point) # Ns*n*n
    cauchy=cauchy_distribution(mask_field,gamma=5,scale=10)
    assert cauchy.shape == dist_field.shape
    masked=dist_field[:,:,:]*( cauchy+ 1)[:,:,:] # Nc*n*n , Ns*n*n ,Nc==Ns-> Nc*n*n
    return masked # Nc*n*n


def voronoi_field(field,sites,**kwargs):
    if "Dm" in kwargs:
        dist=d_mahalanobis(field,sites,kwargs["Dm"])
    else:
        dist=d_euclidean(field,sites) # Nc,r,c
    if "cauchy_field" in kwargs and "cauchy_points" in kwargs:
        dist=cauchy_mask(dist,kwargs["cauchy_points"],kwargs["cauchy_field"])
    soft=batch_softmax(dist)
    beta=5
    rho=1-np.sum(soft**beta,axis=0)
    return rho

if __name__ == '__main__':

    start_time = time.time() #计时起点
    x_len=100
    y_len=100
    coords = np.indices((x_len, y_len))
    coordinates = np.stack(coords, axis=-1)
    cauchy_field = coordinates.copy()
    np.random.seed(0)

    # sites=np.array(([20,50],[80,50]))
    # cauchy_points=np.array(([40,20],[50,70]))

    sites=np.random.randint(low=-20, high=120, size=(20,2))
    cauchy_points=sites.copy()
    cauchy_points=cauchy_points+np.random.normal(loc=0, scale=10, size=cauchy_points.shape)


    Dm=np.tile(np.array(([1,0],[0,1])),(sites.shape[0],1,1)) #Nc*dim*dim
    Dm[0]=np.array(([1,0],[0,1]))

    # dist_field=voronoi_field(coordinates, sites, Dm=Dm, sigmoid_sites=sigmoid_sites, sigmoid_field=sigmoid_field)
    # field=voronoi_field(coordinates, sites, Dm=Dm)
    field=voronoi_field(coordinates, sites, Dm=Dm,cauchy_field=cauchy_field,cauchy_points=cauchy_points)


    print(f"代码运行时间：{time.time() - start_time:.6f} 秒")

    plt.imshow(field, cmap='viridis')  # 使用 'viridis' 颜色映射
    plt.colorbar(label='Pixel Value')  # 添加颜色条用于显示值的范围
    plt.title("Pixel Values Visualized with Colors")
    plt.scatter(sites[:,1], sites[:,0], marker='^', color='r')
    plt.scatter(cauchy_points[:,1], cauchy_points[:,0], marker='+', color='w')
    print(f"绘图消耗时间：{time.time() - start_time:.6f} 秒")
    plt.show()
