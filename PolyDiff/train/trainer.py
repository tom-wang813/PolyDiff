# PolyDiff/train/trainer.py
from __future__ import annotations

import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Callable

from PolyDiff.configs import train_config
from PolyDiff.train.training_state import TrainingStateManager, CheckpointLoader
from PolyDiff.train.callbacks import Callback, TensorboardCallback
from PolyDiff.train.callbacks import ReproducibilityCallback  # 可选
from PolyDiff.train.earlystopping import EarlyStoppingCallback  # 可选

class Trainer:
    """
    只通过 callbacks（如 TensorboardCallback）来记录所有指标，
    不再输出 CSV。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        loss_fns: Dict[str, Callable] | None = None,
        device: str | torch.device = "cpu",
        max_epochs: int = train_config.EPOCHS,
        grad_accum_steps: int = 1,
        clip_grad_norm: float | None = None,
        callbacks: Optional[List[Callback]] = None,
        resume: bool = False,
        ckpt_step: int | None = None,
        ddp: bool = False,
        ddp_device_ids: List[int] | None = None,
    ) -> None:
        # — device & model & DDP wrap —
        self.device = torch.device(device)
        self.model = model.to(self.device)
        if ddp:
            assert dist.is_available() and dist.is_initialized(), \
                "DDP requires torch.distributed.init_process_group"
            self.model = DDP(self.model, device_ids=ddp_device_ids)

        # — core objects —
        self.optimizer, self.scheduler = optimizer, scheduler
        self.train_loader, self.val_loader = train_loader, val_loader
        self.loss_fns = loss_fns or {"loss": torch.nn.CrossEntropyLoss()}
        self.max_epochs = max_epochs
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad_norm = clip_grad_norm

        # — callbacks  (TensorboardCallback 至少要提供) —
        self.callbacks = callbacks or []
        if not any(isinstance(cb, TensorboardCallback) for cb in self.callbacks):
            # 如果用户没传，自动添加一个默认的
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.callbacks.insert(0, TensorboardCallback(log_dir=f"runs/train_{ts}"))

        # — training state & resume —
        self.state_mgr = TrainingStateManager(model, optimizer, scheduler)
        self.epoch, self.global_step = 0, 0
        if resume:
            meta = CheckpointLoader(model, optimizer, scheduler).load(step=ckpt_step)
            if meta is not None:
                self.epoch       = meta["epoch"] + 1
                self.global_step = meta["step"]  + 1

    def _on_step_end(self, phase: str, metrics: Dict[str, float]) -> None:
        """
        Training step / validation step 结束后，只触发 callbacks。
        """
        for cb in self.callbacks:
            cb.on_step_end(self, phase, metrics)

    def _on_epoch_end(self, summary: Dict[str, Any]) -> None:
        """
        Epoch 结束后，只触发 callbacks。
        """
        for cb in self.callbacks:
            cb.on_epoch_end(self, summary)

    def fit(self) -> None:
        # — train start —
        for cb in self.callbacks:
            cb.on_train_start(self)

        for self.epoch in range(self.epoch, self.max_epochs):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics   = self._run_epoch(self.val_loader,   train=False)

            # — scheduler step —
            try:
                self.scheduler.step(val_metrics["loss"])
            except TypeError:
                self.scheduler.step()

            # — epoch summary & callbacks —
            summary = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}":   v for k, v in val_metrics.items()},
                "epoch": self.epoch,
            }
            self._on_epoch_end(summary)

            # — checkpoint —
            self.state_mgr.update(self.global_step, epoch=self.epoch)

            # — early stopping (通过 EarlyStoppingCallback 设置 trainer.stop_training) —
            if getattr(self, "stop_training", False):
                break

        # — train end —
        for cb in self.callbacks:
            cb.on_train_end(self)

    def _run_epoch(self, loader: DataLoader, *, train: bool) -> Dict[str, float]:
        phase = "train" if train else "val"
        self.model.train(train)

        from PolyDiff.train.trainer import MetricTracker  # 保留原来的 MetricTracker
        tracker = MetricTracker()
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"{phase.title()} Epoch {self.epoch}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            inputs, targets = batch
            # 支持 MPS/GPU 非阻塞
            inputs = inputs.to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}

            # — forward & loss —
            outputs = self.model(inputs)
            loss_dict = {t: fn(outputs[t], targets[t]) for t, fn in self.loss_fns.items()}
            loss = sum(loss_dict.values()) / self.grad_accum_steps

            if train:
                loss.backward()
                if step % self.grad_accum_steps == 0:
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

            # — bookkeeping & callbacks per step —
            loss_cpu = {k: v.item() for k, v in loss_dict.items()}
            loss_cpu["loss"] = sum(loss_cpu.values())
            tracker.update(loss_cpu, n=inputs.size(0))
            pbar.set_postfix({"loss": f"{tracker.averages()['loss']:.4f}"})

            # 触发 step_end callback
            self._on_step_end(phase, loss_cpu)

        return tracker.averages()
