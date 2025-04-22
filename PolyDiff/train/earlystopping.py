from __future__ import annotations

from typing import Dict
from PolyDiff.configs import train_config



class EarlyStopping:
    """Early stop training when a monitored metric stops improving."""

    def __init__(
        self,
        patience: int = train_config.EARLY_STOPPING_PATIENCE,
        delta: float = train_config.EARLY_STOPPING_DELTA,
        monitor: str = "val_loss",
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self.verbose = verbose

    def __call__(self, metrics: Dict[str, float]) -> bool:
        current = metrics[self.monitor]
        improve = (
            (self.mode == "min" and (self.best is None or current < self.best - self.delta))
            or (self.mode == "max" and (self.best is None or current > self.best + self.delta))
        )

        if improve:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
        return self.counter >= self.patience
