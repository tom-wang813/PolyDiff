# schedule_factory.py (updated for Hydra integration)
from __future__ import annotations

from omegaconf import DictConfig
from PolyDiff.diffusion.diffusion_forward import LinearSchedule, CosineSchedule, SigmoidSchedule

__all__ = ["get_schedule"]

_SCHEDULES = {
    "linear": LinearSchedule,
    "cosine": CosineSchedule,
    "sigmoid": SigmoidSchedule,
}

def get_schedule(cfg: DictConfig, *, device: str = "cpu"):
    """
    Create a beta schedule based on the Hydra config.

    Args:
        cfg: OmegaConf DictConfig with a 'diffusion' field containing:
            - beta_schedule: str
            - max_timesteps: int
            - beta_start: float
            - beta_end: float
        device: torch device string

    Returns:
        An instance of the requested schedule class.
    """
    sched_cfg = cfg.diffusion
    name = sched_cfg.beta_schedule.lower()
    num_steps = sched_cfg.max_timesteps
    beta_start = sched_cfg.beta_start
    beta_end = sched_cfg.beta_end

    if name not in _SCHEDULES:
        raise ValueError(
            f"Unsupported beta_schedule '{sched_cfg.beta_schedule}'. Available: {', '.join(_SCHEDULES)}"
        )

    return _SCHEDULES[name](
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    )
