import jax
import tqdm
from scipy.ndimage import sobel
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
import jax.numpy as np
from jax import config
from scipy.spatial.distance import cdist

config.update("jax_enable_x64", True)
def extract_continuous_boundary(rho, threshold=0.5):
    # 阈值处理：将大于 threshold 的部分看作区域
    mask = (rho > threshold).astype(float)
    grad_x = sobel(mask, axis=0)  # x 方向梯度
    grad_y = sobel(mask, axis=1)  # y 方向梯度
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    boundary = grad_magnitude * (grad_magnitude > 0)
    return boundary



def blur_edges(matrix, blur_sigma=1.0):
    """
    对实体边缘进行模糊处理。
    Args:
        matrix: 输入二维矩阵，值在 0-1 范围内。
        blur_sigma (float): 高斯模糊的标准差，控制模糊强度。
    Returns:
        np.ndarray: 模糊处理后的矩阵。
    """
    # 确保矩阵值在 0-1 范围内
    matrix = np.clip(matrix, 0, 1)
    # 二值化，识别实体区域
    binary_matrix = (matrix > 0.5).astype(float)
    # 识别边界：实体 - 腐蚀后的实体
    structure = np.ones((3, 3))  # 定义 8 邻域结构
    boundary = binary_matrix - binary_erosion(binary_matrix, structure=structure)
    # 对边界进行高斯模糊
    blurred_boundary = gaussian_filter(boundary, sigma=blur_sigma)
    # 将模糊结果叠加到原始矩阵
    result = matrix + blurred_boundary
    result = np.clip(result, 0, 1)  # 保证矩阵值仍在 0-1 范围内
    return result

def map_func(source_min, source_max, target_min, target_max):
        """
        将 values 从 [source_min, source_max] 映射到 [target_min, target_max].
        """
        return lambda x :target_min + (x - source_min) * (target_max - target_min) / (source_max - source_min)

def inv_2d(matrix):
    a = matrix[:, 0, 0]
    b = matrix[:, 0, 1]
    c = matrix[:, 1, 0]
    d = matrix[:, 1, 1]
    determinants = a * d - b * c
    # determinants=np.linalg.det(matrix)
    # if np.any(np.isclose(determinants, 0)):
    #     raise ValueError("unable to compute inverse！")
    inverses = np.zeros_like(matrix,dtype=np.float64)
    inverses =inverses.at[:, 0, 0].set(d / determinants)  # d / det
    inverses =inverses.at[:, 0, 1].set(-b / determinants)  # -b / det
    inverses =inverses.at[:, 1, 0].set(-c / determinants ) # -c / det
    inverses =inverses.at[:, 1, 1].set(a / determinants)
    return inverses

def points_filter(points,threshold):

    # 计算所有点对之间的欧几里得距离
    distances = cdist(points, points)
    # 创建一个布尔矩阵，标记哪些点的距离小于阈值
    mask = np.triu(distances < threshold, 1)  # 保留上三角矩阵，不重复计算
    # 获取需要去除的点的索引
    to_remove = set()
    for i in range(mask.shape[0]):
        if any(mask[i]):  # 如果当前点与其他点太近
            to_remove.add(i)
    # 从原始点集中去除这些点
    points_filtered = np.delete(points, list(to_remove), axis=0)
    return points_filtered


def remove_nearby_points(max_stress_position, max_stress_direction, threshold):
    """
    根据 max_stress_position 中点的坐标去除离得太近的点，并保持 max_stress_direction 的一一对应关系。

    参数:
    max_stress_position (np.ndarray): 形状为 (n, 2)，点的位置坐标。
    max_stress_direction (np.ndarray): 形状为 (n, 2)，点对应的方向。
    threshold (float): 点之间的最小距离阈值，小于此距离的点会被认为是太近的并被去除。

    返回:
    max_stress_position_filtered (np.ndarray): 过滤后的 max_stress_position，保留满足距离阈值条件的点。
    max_stress_direction_filtered (np.ndarray): 过滤后的 max_stress_direction，对应去除的点已被移除。
    """
    # 计算所有点对之间的欧几里得距离
    distances = cdist(max_stress_position, max_stress_position)

    # 创建一个布尔矩阵，标记哪些点的距离小于阈值
    mask = np.triu(distances < threshold, 1)  # 只考虑上三角部分，避免重复检查

    # 获取需要去除的点的索引
    to_remove = set()
    for i in range(mask.shape[0]):
        if any(mask[i]):  # 如果当前点与其他点太近
            to_remove.add(i)

    # 将 to_remove 转换为列表并排序，这样我们就可以按顺序去除元素
    to_remove = np.array(sorted(to_remove, reverse=True))

    # 从 max_stress_position 和 max_stress_direction 中去除这些点
    max_stress_position_filtered = np.delete(max_stress_position, to_remove, axis=0)
    max_stress_direction_filtered = np.delete(max_stress_direction, to_remove, axis=0)

    return max_stress_position_filtered, max_stress_direction_filtered