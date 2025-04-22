# PolyDiff/dataset/__init__.py
"""
PolyDiff ‑ dataset sub‑package

公開：
    * Registry helpers
    * 常用 Dataset / Collate 實作
    * build_dataloaders 便利函式
"""
from .registry import (
    build_dataset, register_dataset,
    build_collate, register_collate,
)
from .text import TextDiffusionDataset     # 使其自動註冊
from .builder import build_dataloaders, DataConfig

__all__ = [
    # registry
    "build_dataset", "register_dataset",
    "build_collate", "register_collate",
    # datasets
    "TextDiffusionDataset",
    # helpers
    "build_dataloaders", "DataConfig",
]
