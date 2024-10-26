import logging
from tqdm import tqdm
import time

# 设置 logging
logging.basicConfig(level=logging.INFO)

# 运行进度条
with tqdm(total=10, leave=True) as pbar:
    for i in range(10):
        time.sleep(0.5)  # 模拟一些工作
        logging.info(f"Processing item {i}")
        pbar.update(1)  # 更新进度条

# 进度条结束后，保持显示
print("任务完成！")
