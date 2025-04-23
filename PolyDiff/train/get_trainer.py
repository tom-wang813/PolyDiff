# PolyDiff/train/get_trainer.py
from __future__ import annotations

from omegaconf import DictConfig
from hydra.utils import instantiate

from PolyDiff.train.trainer import Trainer


def get_trainer(cfg: DictConfig) -> Trainer:         # ← 只保留這個介面
    """利用 Hydra DictConfig 建立並回傳 Trainer。"""

    # 1) 建模、優化器、排程器、DataLoader
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    train_loader = instantiate(cfg.data.train_loader)
    val_loader   = instantiate(cfg.data.val_loader)

    # 2) Callbacks（cfg.callbacks 可以是 ListConfig）
    callbacks = [instantiate(cb) for cb in cfg.get("callbacks", [])]

    # 3) 組裝 Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fns=instantiate(cfg.get("loss_fns")) if "loss_fns" in cfg else None,
        device=cfg.trainer.device,
        max_epochs=cfg.trainer.max_epochs,
        grad_accum_steps=cfg.trainer.grad_accum_steps,
        clip_grad_norm=cfg.trainer.clip_grad_norm,
        callbacks=callbacks,
        resume=cfg.trainer.resume,
        validate_every_n_steps=cfg.trainer.validate_every_n_steps,
        seed=cfg.seed,                            # ↓ 第 3 節會加進 Trainer
    )
    return trainer
