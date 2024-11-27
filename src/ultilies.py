from scipy.ndimage import sobel
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
import jax.numpy as np

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