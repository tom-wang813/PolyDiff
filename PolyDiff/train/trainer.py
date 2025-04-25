from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from PolyDiff.train.callbacks import (
    Callback,
    TensorboardCallback,
    ReproducibilityCallback,
    ModelCheckpointCallback,
    EarlyStoppingCallback,
)

__all__ = ["Trainer", "MetricTracker"]

class MetricTracker:
    """
    Keeps a weighted running mean of scalar metrics.
    """
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for k, v in values.items():
            self.totals[k] = self.totals.get(k, 0.0) + v * n
            self.counts[k] = self.counts.get(k, 0) + n

    def averages(self) -> Dict[str, float]:
        return {k: self.totals[k] / max(self.counts[k], 1) for k in self.totals}

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()


class Trainer:
    """
    Orchestrates training, validation, checkpointing, logging, and reproducibility.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        exp_dir: str,
        *,
        loss_fns: Optional[Dict[str, Callable]] = None,
        device: str | torch.device = "cpu",
        max_epochs: int = 10,
        grad_accum_steps: int = 1,
        clip_grad_norm: float | None = None,
        callbacks: Optional[List[Callback]] = None,
        resume: bool = False,
        seed: int = 42,
        validate_every_n_steps: Optional[int] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)

        # — core objects —
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fns = loss_fns or {"loss": torch.nn.CrossEntropyLoss()}
        self.max_epochs = max_epochs
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad_norm = clip_grad_norm
        self.validate_every_n_steps = validate_every_n_steps

        # — callbacks: reproducibility, checkpoint, tensorboard , earlystopping—
        self.callbacks: List[Callback] = callbacks or []
        if not any(isinstance(cb, ReproducibilityCallback) for cb in self.callbacks):
            self.callbacks.insert(0, ReproducibilityCallback(seed=seed))
        ckpt_cb = ModelCheckpointCallback(model, optimizer, scheduler, resume=resume, ckpt_dir=exp_dir)
        self.callbacks.insert(1, ckpt_cb)
        if not any(isinstance(cb, TensorboardCallback) for cb in self.callbacks):
            self.callbacks.append(TensorboardCallback(log_dir=exp_dir))

        if not any(isinstance(cb, EarlyStoppingCallback) for cb in self.callbacks):
            self.callbacks.append(
                EarlyStoppingCallback(
                    monitor="val_loss",
                    patience=50,
                    mode="min",
                    verbose=True,
                )
            )

        # — initialize epoch & step —
        self.epoch: int = 0
        self.global_step: int = 0

    def _maybe_validate_step(self) -> None:
        """
        If step-based validation is configured, run validation and trigger callbacks.
        """
        if (
            self.validate_every_n_steps
            and self.global_step > 0
            and self.global_step % self.validate_every_n_steps == 0
        ):
            val_metrics = self._run_epoch(self.val_loader, train=False)
            summary = {f"val_{k}": v for k, v in val_metrics.items()}
            summary.update({"epoch": self.epoch, "global_step": self.global_step})
            for cb in self.callbacks:
                cb.on_epoch_end(self, summary)

    def _on_step_end(self, phase: str, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_step_end(self, phase, metrics)
        if phase == "train":
            self._maybe_validate_step()

    def _on_epoch_end(self, summary: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(self, summary)

    def fit(self) -> None:
        # — train start —
        for cb in self.callbacks:
            cb.on_train_start(self)

        for self.epoch in range(self.epoch, self.max_epochs):
            # training epoch
            train_metrics = self._run_epoch(self.train_loader, train=True)

            # — scheduler step on training loss —
            if "loss" in train_metrics:
                if isinstance(self.scheduler, _LRScheduler):
                    self.scheduler.step(train_metrics["loss"])
                else:
                    self.scheduler.step()

            # validation epoch (or skipped if step-based)
            if self.validate_every_n_steps is None:
                val_metrics = self._run_epoch(self.val_loader, train=False)
                summary = {f"train_{k}": v for k, v in train_metrics.items()}
                summary.update({f"val_{k}": v for k, v in val_metrics.items()})
                summary["epoch"] = self.epoch
            else:
                summary = {f"train_{k}": v for k, v in train_metrics.items()}
                summary["epoch"] = self.epoch

            # epoch end callbacks
            self._on_epoch_end(summary)

            # stop if flagged by callbacks
            if getattr(self, "stop_training", False):
                break

        # — train end —
        for cb in self.callbacks:
            cb.on_train_end(self)

    def _run_epoch(self, loader: DataLoader, *, train: bool) -> Dict[str, float]:
        phase = "train" if train else "val"
        self.model.train(train)

        tracker = MetricTracker()
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"{phase.title()} Epoch {self.epoch}", leave=False)
    
        for step, batch in enumerate(pbar, start=1):
            if isinstance(batch, dict):
                inputs = batch["input"]
                targets = batch["labels"]
            else:
                inputs = batch[0]
                targets = batch[1]
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device, non_blocking=True)
            if isinstance(targets, dict):
                targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
            else:
                targets = targets.to(self.device, non_blocking=True)

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

            loss_cpu = {k: v.item() for k, v in loss_dict.items()}
            loss_cpu["loss"] = sum(loss_cpu.values())
            tracker.update(loss_cpu, n=inputs.size(0))
            pbar.set_postfix({"loss": f"{tracker.averages()['loss']:.4f}"})

            # step end callbacks (including optional validation)
            self._on_step_end(phase, loss_cpu)

        return tracker.averages()
