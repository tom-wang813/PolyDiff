from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
__all__ = [
    'BaseSchedule',
    'LinearSchedule',
    'CosineSchedule',
    'SigmoidSchedule',
    'DiffusionModel',
    'AbsorbingDiffusion'
]
# Diffusion forward process classes

class BaseSchedule(ABC):
    """
    Abstract base for noise schedules. Generates beta and alpha_cumprod tensors.
    """
    def __init__(self, num_steps: int, device: str = 'cpu'):
        self.num_steps = num_steps
        self.device = device
        self.beta: torch.Tensor = torch.zeros(num_steps + 1, device=self.device)
        self.alpha_cumprod: torch.Tensor = torch.ones(num_steps + 1, device=self.device)
        self.generate()

    @abstractmethod
    def generate(self) -> None:
        """Populate self.beta and self.alpha_cumprod"""
        pass


class LinearSchedule:
    def __init__(
        self,
        num_steps: int,
        beta_start: float,
        beta_end: float,
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        # Linearly spaced betas
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)

    def __call__(self):
        return self.betas


class CosineSchedule:
    def __init__(
        self,
        num_steps: int,
        beta_start: float,
        beta_end: float,
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        # Implement cosine schedule (example)
        steps = torch.arange(num_steps, dtype=torch.float32, device=device)
        alphas_cumprod = torch.cos((steps / num_steps + 0.008) / 1.008 * torch.pi / 2) ** 2
        self.betas = torch.clamp(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], beta_start, beta_end)

    def __call__(self):
        return self.betas


class SigmoidSchedule:
    def __init__(
        self,
        num_steps: int,
        beta_start: float,
        beta_end: float,
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        # Sigmoid-based schedule
        steps = torch.linspace(-6, 6, num_steps, device=device)
        sigmoid = torch.sigmoid(steps)
        self.betas = beta_start + (beta_end - beta_start) * (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())

    def __call__(self):
        return self.betas



class DiffusionModel(ABC):
    """Abstract base for different diffusion behaviors."""
    @abstractmethod
    def diffuse(
        self,
        x: torch.Tensor,
        t: Union[int, torch.Tensor],
        condition_mask: Optional[torch.Tensor] = None,
        rnd: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return a noised version of x after t steps."""
        ...

    __call__ = diffuse


class AbsorbingDiffusion(DiffusionModel):
    """
    Absorbing diffusion: once masked, tokens stay masked.
    Uses a noise schedule for masking probabilities.

    - schedule.beta: tensor of shape (T+1,), β_0 unused, β_t ∈ [0,1)
    - mask_token: integer ID to replace masked tokens
    - pad_token: integer ID which should never be masked
    """
    def __init__(
        self,
        num_steps: int,
        mask_token: int,
        schedule: BaseSchedule,
        device: str = 'cpu',
        pad_token: Optional[int] = None
    ):
        self.num_steps = num_steps
        self.mask_token = mask_token
        self.schedule = schedule
        self.device = device
        self.pad_token = pad_token

    def get_eligible_mask(
        self,
        x: torch.Tensor,
        condition_mask: Optional[torch.Tensor]
    ) -> torch.BoolTensor:
        eligible = torch.ones_like(x, dtype=torch.bool, device=self.device)
        if self.pad_token is not None:
            eligible &= (x != self.pad_token)
        if condition_mask is not None:
            eligible &= ~condition_mask  # 保持条件位置不被更改
        eligible &= (x != self.mask_token)  # 已经 masked 的也不再重复
        return eligible

    def compute_cumulative_prob(
        self,
        eligible_ratio: torch.Tensor,
        t_tensor: torch.Tensor
    ) -> torch.Tensor:
        # 计算到每个时间步被 mask 的累计概率
        t_max = int(t_tensor.max().item())
        betas = self.schedule.beta[1:t_max + 1].to(self.device)
        steps = torch.arange(t_max, device=self.device).unsqueeze(0)
        valid = steps < t_tensor.unsqueeze(1)
        factors = 1 - betas.unsqueeze(0) * eligible_ratio.unsqueeze(1)
        factors = torch.where(valid, factors, torch.ones_like(factors))
        prod = torch.prod(factors, dim=1)
        return 1 - prod

    def diffuse(
        self,
        x: torch.Tensor,
        t: Union[int, torch.Tensor],
        condition_mask: Optional[torch.Tensor] = None,
        rnd: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 支持 int 或 Tensor 的 t 输入
        if isinstance(t, int):
            t_tensor = torch.full((x.size(0),), t, dtype=torch.long, device=self.device)
        else:
            t_tensor = t.to(self.device)

        # t=0 时无噪声
        if (t_tensor == 0).all():
            return x

        # 计算 eligible 比例
        eligible_mask = self.get_eligible_mask(x, condition_mask)
        eff_ratio = eligible_mask.float().mean(dim=1)

        # 累计 mask 概率
        cum_prob = self.compute_cumulative_prob(eff_ratio, t_tensor)

        # 随机采样决定哪些 token 被 mask
        if rnd is None:
            rnd = torch.rand_like(x, dtype=torch.float, device=self.device)
        update = eligible_mask & (rnd < cum_prob.unsqueeze(1))
        x_noised = x.clone()
        x_noised[update] = self.mask_token
        return x_noised