# PolyDiff/train/callbacks.py

from typing import Dict, Any, Optional, Union
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler

from PolyDiff.train.utils import seed_everything
from PolyDiff.train.checkpoint import ModelCheckpointManager

__all__ = [
    "Callback",
    "ReproducibilityCallback",
    "TensorboardCallback",
    "ModelCheckpointCallback",
    "EarlyStoppingCallback",
]

class Callback:
    """
    Base class for Trainer callbacks.
    """

    def on_train_start(self, trainer: Any) -> None:
        """训练开始时调用一次"""
        pass

    def on_step_end(
        self,
        trainer: Any,
        phase: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        每个 step 结束后调用。

        Parameters
        ----------
        phase : "train" or "val"
        metrics : 当 step 结束时的 loss dict, e.g. {"loss": 1.23, "task1": 0.4}
        """
        pass

    def on_epoch_end(self, trainer: Any, metrics: Dict[str, Any]) -> None:
        """
        每个 epoch 结束后调用。

        metrics 包含 epoch 级别的 summary dict, e.g.
        {
          "train_loss": 0.12,
          "val_loss": 0.15,
          "epoch": 3
        }
        """
        pass

    def on_train_end(self, trainer: Any) -> None:
        """整个训练循环结束时调用一次"""
        pass


class ReproducibilityCallback(Callback):
    """
    在训练开始前全局设定随机种子，保证实验可复现。
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def on_train_start(self, trainer: Any) -> None:
        seed_everything(self.seed)
        print(f"[Reproducibility] set seed = {self.seed}")


class TensorboardCallback(Callback):
    """
    用 TensorBoard 记录 step & epoch 指标。
    """

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def on_train_start(self, trainer: Any) -> None:
        # 可以写一些超参到 TB
        for k, v in vars(trainer).items():
            # 举例：记录 lr、batch_size
            if k in ("max_epochs", "grad_accum_steps"):
                self.writer.add_text(f"config/{k}", str(v), 0)

    def on_step_end(
        self,
        trainer: Any,
        phase: str,
        metrics: Dict[str, float],
    ) -> None:
        for name, value in metrics.items():
            # e.g. tag = "train/loss" 或 "val/loss"
            tag = f"{phase}/{name}"
            self.writer.add_scalar(tag, value, trainer.global_step)

    def on_epoch_end(self, trainer: Any, metrics: Dict[str, Any]) -> None:
        for name, value in metrics.items():
            # e.g. tag = "epoch/train_loss" 或 "epoch/val_loss"
            if name == "epoch":
                continue
            tag = f"epoch/{name}"
            self.writer.add_scalar(tag, value, trainer.epoch)

    def on_train_end(self, trainer: Any) -> None:
        self.writer.close()


class ModelCheckpointCallback(Callback):
    """
    Restore at the start of training, save at the end of each epoch.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        resume: bool = False,
        ckpt_dir: Union[str, Path] = "checkpoints",
    ) -> None:
        # Initialize manager with custom checkpoint directory
        self.ckpt_mgr = ModelCheckpointManager(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_dir=Path(ckpt_dir),
        )
        self.resume = resume
        self.loaded_meta: Optional[Dict[str, int]] = None

    def on_train_start(self, trainer: Any) -> None:
        if self.resume:
            meta = self.ckpt_mgr.load()
            if meta is not None:
                trainer.epoch = meta["epoch"] + 1
                trainer.global_step = meta["step"] + 1
                print(
                    f"[CheckpointCallback] Resume from epoch={trainer.epoch} "
                    f"step={trainer.global_step}"
                )

    def on_epoch_end(self, trainer: Any, metrics: Dict[str, Any]) -> None:
        # The "metrics" dict typically contains an "epoch" field
        epoch = metrics.get("epoch", trainer.epoch)
        self.ckpt_mgr.save(step=trainer.global_step, epoch=epoch)




class EarlyStoppingCallback(Callback):
    """
    General Early-Stopping callback supporting min/max mode, patience, and delta.

    Example:
        EarlyStoppingCallback(
            monitor="val_loss",
            mode="min",
            delta=0.01,
            patience=3,
            verbose=True
        )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
    ) -> None:
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.monitor = monitor
        self.mode = mode
        self.delta = delta
        self.patience = patience
        self.verbose = verbose

        self.best: float | None = None
        self.counter = 0

    def on_epoch_end(self, trainer: Any, metrics: Dict[str, Any]) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            return

        improved = (
            (self.mode == "min" and (self.best is None or current < self.best - self.delta))
            or (self.mode == "max" and (self.best is None or current > self.best + self.delta))
        )

        if improved:
            self.best = current
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] {self.monitor} improved to {current:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("[EarlyStopping] Triggered, will stop training.")
                # 通过给 trainer 设置标记来通知
                trainer.stop_training = True
