# PolyDiff/dataset/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Callable, Any

from torch.utils.data import IterableDataset, Dataset


# --------------------------------------------------------------------------- #
# registry utilities
# --------------------------------------------------------------------------- #
_DATASET_REGISTRY: Dict[str, type] = {}
_COLLATE_REGISTRY: Dict[str, Callable] = {}


def register_dataset(name: str):
    def decorator(cls: type):
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def build_dataset(name: str, *args, **kwargs):
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Avail: {list(_DATASET_REGISTRY.keys())}")
    return _DATASET_REGISTRY[name](*args, **kwargs)


def register_collate(name: str):
    def decorator(fn: Callable):
        _COLLATE_REGISTRY[name] = fn
        return fn
    return decorator


def build_collate(name: str, *args, **kwargs) -> Callable:
    if name not in _COLLATE_REGISTRY:
        raise ValueError(f"Unknown collate '{name}'. "
                         f"Avail: {list(_COLLATE_REGISTRY.keys())}")
    return _COLLATE_REGISTRY[name](*args, **kwargs)


# --------------------------------------------------------------------------- #
# base Dataset (iterable & map)
# --------------------------------------------------------------------------- #
class BaseIterableDataset(IterableDataset, ABC):
    """
    Minimal iterable dataset: implement `_iter_data()` to yield raw samples.
    """

    def __iter__(self):
        yield from self._iter_data()

    # -------------- to be overriden by subclass -------------------------- #
    @abstractmethod
    def _iter_data(self):
        raise NotImplementedError


class BaseMapDataset(Dataset, ABC):
    """
    Map‑style dataset base; implement `__getitem__` & `__len__`.
    """

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError
