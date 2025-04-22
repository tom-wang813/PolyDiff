# PolyDiff/train/callbacks.py

from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
from PolyDiff.train.utils import seed_everything


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
