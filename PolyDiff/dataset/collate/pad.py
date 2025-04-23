# PolyDiff/dataset/collate/pad.py
from __future__ import annotations

from typing import List

import torch

from PolyDiff.dataset.base import register_collate


@register_collate("pad")
class PadCollator:
    """
    Simple left-aligned padding.

    >>> collate = PadCollator(pad_token=0)
    >>> batch   = collate(list_of_tensors)
    """

    def __init__(self, pad_token: int):
        self.pad_token = pad_token

    # ------------------------------------------------------------------ #
    def __call__(self, batch: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(x.size(0) for x in batch)
        out = torch.full((len(batch), max_len),
                         self.pad_token,
                         dtype=batch[0].dtype)
        for i, seq in enumerate(batch):
            out[i, : seq.size(0)] = seq
        return out
