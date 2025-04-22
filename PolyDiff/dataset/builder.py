# PolyDiff/dataset/builder.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from PolyDiff.dataset.registry import build_dataset, build_collate


# --------------------------------------------------------------------------- #
# simple dataclass config ‑ 可由 yaml 轉換而來
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    # --- dataset ------------------------------------------------------------
    dataset_name: str          # e.g. "text_diffusion"
    data_dir: str
    batch_size: int
    num_workers: int = 4
    split_train: str = "train"
    split_val: str   = "val"

    # --- collate ------------------------------------------------------------
    collate_name: str = "diffusion"          # or "pad"
    collate_kwargs: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    def loader_kwargs(self, device: str | torch.device) -> Dict[str, Any]:
        pin = str(device) != "cpu"
        return dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=pin,
            persistent_workers=self.num_workers > 0,
        )


# --------------------------------------------------------------------------- #
# public helper
# --------------------------------------------------------------------------- #
def build_dataloaders(
    cfg: DataConfig,
    tokenizer,
    *,
    device: str | torch.device = "cpu",
):
    """
    Return (train_loader, val_loader) built from config.
    """
    # -- dataset objects -----------------------------------------------------
    ds_train = build_dataset(
        cfg.dataset_name,
        data_dir=cfg.data_dir,
        split=cfg.split_train,
        tokenizer=tokenizer,
    )
    ds_val = build_dataset(
        cfg.dataset_name,
        data_dir=cfg.data_dir,
        split=cfg.split_val,
        tokenizer=tokenizer,
        shuffle_files=False,
    )

    # -- collate -------------------------------------------------------------
    collate_fn = build_collate(
        cfg.collate_name,
        **cfg.collate_kwargs,
        device=device,
    )

    # -- loaders -------------------------------------------------------------
    kwargs = cfg.loader_kwargs(device)
    train_dl = DataLoader(ds_train, shuffle=True,  collate_fn=collate_fn, **kwargs)
    val_dl   = DataLoader(ds_val,   shuffle=False, collate_fn=collate_fn, **kwargs)
    return train_dl, val_dl
