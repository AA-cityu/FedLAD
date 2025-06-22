from typing import Tuple
import torch
import numpy as np


def parse_template(log_path: str, device: str = "cpu", **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Parse the template function for the general log dataset.
    Please customize this function according to the structure of the dataset.
    '''
    print(f"[INFO] Parsing logs from: {log_path}")
    # 示例返回结构，请根据实际情况替换
    x = np.random.rand(100, 10, 768)
    y = np.random.randint(0, 2, size=(100,))
    return x.astype(np.float32), y.astype(np.int32)
