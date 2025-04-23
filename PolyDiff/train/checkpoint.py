import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["ModelCheckpointManager"]

_ckpt_re = re.compile(r"checkpoint_step_(\d+)\.pth$")


class ModelCheckpointManager:
    """
    Unified management of snapshots for the model, optimizer, and scheduler.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        ckpt_dir: Union[str, Path] = "checkpoints",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # 指定或建立 checkpoint 資料夾
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _make_path(self, step: int) -> Path:
        """Compose checkpoint file path from step."""
        return self.ckpt_dir / f"checkpoint_step_{step}.pth"

    def save(self, step: int, epoch: int) -> None:
        """
        Save model, optimizer, scheduler states and metadata.
        """
        payload: Dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            # 若有其他層級資料，可在此加入
        }
        path = self._make_path(step)
        torch.save(payload, path)
        print(f"[Checkpoint] Saved to {path}")

    def latest_step(self) -> Optional[int]:
        """
        Return the maximum checkpoint step found, or None if no checkpoints.
        """
        steps = [
            int(m.group(1))
            for p in self.ckpt_dir.iterdir()
            if (m := _ckpt_re.match(p.name))
        ]
        return max(steps) if steps else None

    def load(self, step: Optional[int] = None) -> Optional[Dict[str, int]]:
        """
        Load checkpoint for given step (or latest if None), restore states,
        and return metadata (epoch, step).
        """
        if step is None:
            step = self.latest_step()
        if step is None:
            print("[Checkpoint] No checkpoint found, starting fresh.")
            return None

        path = self._make_path(step)
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"[Checkpoint] Loaded from {path}")
        return {"epoch": ckpt["epoch"], "step": ckpt["step"]}
