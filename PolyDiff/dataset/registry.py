# PolyDiff/dataset/registry.py
"""
Thin re-export of registry utilities defined in `dataset.base`.

這樣用戶可選擇：
    from PolyDiff.dataset import build_dataset
或
    from PolyDiff.dataset.registry import build_dataset
均可。
"""
from PolyDiff.dataset.base import (
    build_dataset, register_dataset,
    build_collate, register_collate,
)

__all__ = [
    "build_dataset", "register_dataset",
    "build_collate", "register_collate",
]
