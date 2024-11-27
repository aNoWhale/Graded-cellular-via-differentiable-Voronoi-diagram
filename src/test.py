import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import sobel

from src.ultilies import blur_edges


def extract_continuous_boundary(rho, threshold=0.5):
    # 阈值处理：将大于 threshold 的部分看作区域
    mask = (rho > threshold).astype(float)
    grad_x = sobel(mask, axis=0)  # x 方向梯度
    grad_y = sobel(mask, axis=1)  # y 方向梯度
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    boundary = grad_magnitude * (grad_magnitude > 0)
    return boundary

matrix = np.zeros((100, 100), dtype=np.uint8)

# 圆心坐标和半径
center = (50, 50)  # 圆心在矩阵的中心
radius = 30

# 遍历矩阵，绘制圆
for i in range(100):
    for j in range(100):
        # 如果 (i, j) 在圆内，设置为 1
        if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2:
            matrix[i, j] = 1
# boundary = extract_continuous_boundary(matrix, threshold=0.5)
import ultilies as ut
boundary=blur_edges(matrix,blur_sigma=3.)
plt.imshow(boundary,cmap="viridis")
plt.show()
