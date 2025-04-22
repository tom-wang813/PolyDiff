# PolyDiff/train/trainer.py
from __future__ import annotations

import datetime, pathlib
from collections.abc import Callable
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from PolyDiff.configs import train_config
from PolyDiff.train.logger import CSVLogger
from PolyDiff.train import TrainingStateManager, CheckpointLoader, CSVLogger


# --------------------------------------------------------------------------- #
# util classes
# --------------------------------------------------------------------------- #
class MetricTracker:
    """
    Keeps a (weighted) running mean of arbitrary scalar metrics.
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


# --------------------------------------------------------------------------- #
# main Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """
    Generic training loop wrapper.

    Parameters
    ----------
    resume : bool
        若為 True，會嘗試從最後一個 / 指定 step 的 checkpoint 繼續。
    ckpt_step : int | None
        若指定數字，將強制載入對應 step；否則自動尋找最新檔案。
    early_stopper : Callable[[Dict[str, float]], bool] | None
        可插拔 early‑stopping 物件；傳入 metrics dict，回傳 bool 表示是否中止。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        loss_fns: Dict[str, Callable] | None = None,
        device: str | torch.device = "mps",
        max_epochs: int = train_config.EPOCHS,
        grad_accum_steps: int = 1,
        clip_grad_norm: float | None = None,
        log_fn: Callable[[Dict[str, Any]], None] | None = None,
        resume: bool = False,
        ckpt_step: int | None = None,
        early_stopper: Callable[[Dict[str, float]], bool] | None = None,
    ) -> None:
        # --- core objects ----------------------------------------------------
        self.model = model.to(device)
        self.optimizer, self.scheduler = optimizer, scheduler
        self.train_loader, self.val_loader = train_loader, val_loader
        self.device = torch.device(device)
        self.loss_fns = loss_fns or {"loss": nn.CrossEntropyLoss()}
        # --- hyper‑params -----------------------------------------------------
        self.max_epochs = max_epochs
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad_norm = clip_grad_norm
        self.early_stopper = early_stopper
        # --- logger -----------------------------------------------------------
        if log_fn is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = pathlib.Path("runs") / f"train_{ts}.csv"
            self.log_fn = CSVLogger(str(default_path))
            print(f"[Logger] metrics → {default_path}")
            self._own_logger = True       # 方便結束時關檔
        else:
            self.log_fn = log_fn
            self._own_logger = False
        # --- state / checkpoint ----------------------------------------------
        self.state_mgr = TrainingStateManager(self.model, self.optimizer, self.scheduler)
        self.epoch, self.global_step = 0, 0
        if resume:
            meta = CheckpointLoader(self.model, self.optimizer, self.scheduler).load(step=ckpt_step)
            if meta is not None:
                self.epoch        = meta["epoch"]  + 1
                self.global_step  = meta["step"]   + 1
                print(f"[Resume] Start from epoch {self.epoch} / step {self.global_step}")
        # ---------------------------------------------------------------------

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _log_step(self, phase: str, step_metrics: Dict[str, float]) -> None:
        """
        Write one‑step metrics to logger (if provided).
        """
        if self.log_fn is None:
            return
        row = {
            "phase": phase,
            "global_step": self.global_step,
            **step_metrics,
        }
        self.log_fn(row)

    # --------------------------------------------------------------------- #
    # main public API
    # --------------------------------------------------------------------- #
    def fit(self) -> None:
        for self.epoch in range(self.epoch, self.max_epochs):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics   = self._run_epoch(self.val_loader,   train=False)

            # ---- scheduler step --------------------------------------------
            try:
                self.scheduler.step(val_metrics["loss"])
            except TypeError:
                self.scheduler.step()

            # ---- epoch‑level logging ---------------------------------------
            epoch_summary = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}":   v for k, v in val_metrics.items()},
                "epoch": self.epoch,
            }
            if self.log_fn is None:
                print(epoch_summary)
            else:
                self.log_fn(epoch_summary)

            # ---- checkpointing ---------------------------------------------
            self.state_mgr.update(self.global_step, epoch=self.epoch)

            # ---- early‑stopping --------------------------------------------
            if self.early_stopper is not None:
                if self.early_stopper({"val_loss": val_metrics["loss"]}):
                    print("Early stopping triggered.")
                    break

        if self._own_logger and hasattr(self.log_fn, "close"):
            self.log_fn.close()

    # --------------------------------------------------------------------- #
    # epoch runner
    # --------------------------------------------------------------------- #
    def _run_epoch(self, loader: DataLoader, *, train: bool) -> Dict[str, float]:
        phase = "train" if train else "val"
        self.model.train(mode=train)

        tracker = MetricTracker()
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pbar = tqdm(loader, desc=f"{phase.title()} Epoch {self.epoch}", leave=False)
            for step, batch in enumerate(pbar, start=1):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                # ---- forward ------------------------------------------------
                outputs = self.model(inputs)

                # ---- loss (multi‑task) -------------------------------------
                loss_dict = {
                    task: fn(outputs[task], targets[task])
                    for task, fn in self.loss_fns.items()
                }
                loss = sum(loss_dict.values()) / self.grad_accum_steps

                if train:
                    loss.backward()
                    if step % self.grad_accum_steps == 0:
                        if self.clip_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.clip_grad_norm
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1

                # ---- bookkeeping ------------------------------------------
                loss_cpu = {k: v.item() for k, v in loss_dict.items()}
                loss_cpu["loss"] = sum(loss_cpu.values())
                tracker.update(loss_cpu, n=inputs.size(0))
                pbar.set_postfix({"loss": f"{tracker.averages()['loss']:.4f}"})

                # ---- per‑step logging --------------------------------------
                self._log_step(phase, loss_cpu)

        return tracker.averages()


# --------------------------------------------------------------------------- #
# legacy shim – import Trainer from new location
# --------------------------------------------------------------------------- #
__all__ = ["MetricTracker", "Trainer"]
