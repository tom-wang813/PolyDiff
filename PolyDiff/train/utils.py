# PolyDiff/train/utils.py

import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    为 Python、NumPy、PyTorch (CPU & 所有 GPU) 设定随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
