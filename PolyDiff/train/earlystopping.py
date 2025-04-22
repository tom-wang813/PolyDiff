# PolyDiff/train/early_stopping.py

from typing import Dict, Any
from PolyDiff.train.callbacks import Callback


class EarlyStoppingCallback(Callback):
    """
    通用 Early‐Stopping，支持 min/max 模式、patience、delta。

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
        patience: int = 5,
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
