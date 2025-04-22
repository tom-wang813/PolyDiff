from __future__ import annotations

from PolyDiff.configs import diffusion_config
from PolyDiff.diffusion.diffusion_forward import LinearSchedule, CosineSchedule, SigmoidSchedule

__all__ = ["get_schedule"]


_SCHEDULES = {
    "linear": LinearSchedule,
    "cosine": CosineSchedule,
    "sigmoid": SigmoidSchedule,
}


def get_schedule(*, device: str = "cpu"):
    name = diffusion_config.BETA_SCHEDULE.lower()
    num_steps = diffusion_config.MAX_TIMESTEPS

    if name not in _SCHEDULES:
        raise ValueError(
            f"Unsupported BETA_SCHEDULE '{diffusion_config.BETA_SCHEDULE}'. "
            f"Available: {', '.join(_SCHEDULES)}"
        )

    return _SCHEDULES[name](num_steps=num_steps, device=device)
