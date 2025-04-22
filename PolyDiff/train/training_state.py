# PolyDiff/train/training_state.py
from __future__ import annotations

import os, re
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from PolyDiff.configs import train_config

__all__ = ["TrainingStateManager", "CheckpointLoader"]

_ckpt_regex = re.compile(r"checkpoint_step_(\d+)\.pth$")


class TrainingStateManager:
    """
    Handles **saving** checkpoints / gradients.
    """

    def __init__(self, model, optimizer, scheduler, initial_step: int = 0) -> None:
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.step       = initial_step

        # dir paths ----------------------------------------------------------
        self.checkpoint_dir      = Path(train_config.CHECKPOINT_DIR)
        self.gradient_save_dir   = Path(train_config.GRADIENT_SAVE_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_save_dir.mkdir(parents=True, exist_ok=True)

        # intervals ----------------------------------------------------------
        self.save_interval            = train_config.SAVE_INTERVAL
        self.gradient_save_interval   = train_config.GRADIENT_SAVE_INTERVAL
        self.max_grad_files           = train_config.MAX_GRAD_FILES
        self.grad_files: deque[str]   = deque(maxlen=self.max_grad_files)

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _filename(self, kind: str, step: int) -> str:
        return f"{kind}_step_{step}.pth"

    def _save(self, obj: Any, kind: str) -> None:
        path = self.checkpoint_dir / self._filename(kind, self.step)
        torch.save(obj, path)
        print(f"[TrainingState] Saved {kind} @ step {self.step} → {path}")

    # --------------------------------------------------------------------- #
    # public save API
    # --------------------------------------------------------------------- #
    def save_checkpoint(self, *, epoch: int) -> None:
        payload: Dict[str, Any] = {
            "model_state_dict":      self.model.state_dict(),
            "optimizer_state_dict":  self.optimizer.state_dict(),
            "scheduler_state_dict":  self.scheduler.state_dict(),
            # meta -----------------------------------------------------------
            "epoch":          epoch,
            "step":           self.step,
            "learning_rate":  self.optimizer.param_groups[0]["lr"],
            "batch_size":     train_config.BATCH_SIZE,
            "epochs":         train_config.EPOCHS,
        }
        self._save(payload, "checkpoint")

    def save_gradients(self) -> None:
        grads = {
            n: p.grad.clone()
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }
        path = self.gradient_save_dir / self._filename("gradients", self.step)
        torch.save(grads, path)

    def update(self, step: int, *, epoch: int) -> None:
        """
        Call **once per training step**. Decides when to actually write files.
        """
        self.step = step
        if self.step % self.save_interval == 0:
            self.save_checkpoint(epoch=epoch)
        if self.step % self.gradient_save_interval == 0:
            self.save_gradients()


# --------------------------------------------------------------------------- #
# loading / resuming
# --------------------------------------------------------------------------- #
class CheckpointLoader:
    """
    Restores **model / optimizer / scheduler** *in place*.

    Returns meta dict on success, None if no ckpt found.
    """

    def __init__(self, model, optimizer, scheduler) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ckpt_dir = Path(train_config.CHECKPOINT_DIR)

    # ------------------------------------------------------------------ #
    def _latest_step(self) -> Optional[int]:
        steps = [
            int(m.group(1))
            for p in self.ckpt_dir.glob("checkpoint_step_*.pth")
            if (m := _ckpt_regex.match(p.name))
        ]
        return max(steps) if steps else None

    # ------------------------------------------------------------------ #
    def load(self, *, step: int | None = None) -> Optional[Dict[str, Any]]:
        if step is None:
            step = self._latest_step()
        if step is None:
            print("[CheckpointLoader] No checkpoints found – start fresh.")
            return None

        ckpt_path = self.ckpt_dir / f"checkpoint_step_{step}.pth"
        if not ckpt_path.exists():
            print(f"[CheckpointLoader] {ckpt_path} not found – start fresh.")
            return None

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        print(f"[CheckpointLoader] Restored from {ckpt_path}")
        return {
            "epoch": ckpt["epoch"],
            "step":  ckpt["step"],
            "learning_rate": ckpt["learning_rate"],
            "batch_size":    ckpt["batch_size"],
        }


