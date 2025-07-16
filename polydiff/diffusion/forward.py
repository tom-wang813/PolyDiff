from typing import Optional, Tuple, Union

import torch

from .schedules import DiffusionSchedule


class DiffusionForward:
    """Base class for forward diffusion process.

    This class implements the forward diffusion process for different types of data,
    including continuous (e.g., images, embeddings) and discrete (e.g., text, SMILES)
    data.
    """

    def __init__(
        self,
        schedule: DiffusionSchedule,
        device: Union[str, torch.device] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """Initialize the forward diffusion process.

        Args:
            schedule: The noise schedule to use for the diffusion process
            device: The device to perform computations on
        """
        self.schedule = schedule
        self.device = device

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: Union[torch.Tensor, int],
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process: q(x_t | x_0).

        Adds noise to the input according to the noise schedule.

        Args:
            x_start: The initial data (B x ...)
            t: Timesteps (B,) or single timestep
            noise: Optional pre-generated noise. If None, random noise will be used

        Returns:
            Tuple containing:
                - The noised data x_t
                - The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Get diffusion parameters for timestep t
        alpha_t, sqrt_alpha_t, sqrt_one_minus_alpha_t = self.schedule.get_parameters(t)

        # Reshape parameters to match input dimensions
        sqrt_alpha_t = sqrt_alpha_t.view(-1, *([1] * (len(x_start.shape) - 1)))
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(
            -1, *([1] * (len(x_start.shape) - 1))
        )

        # Forward process: x_t = √α_t * x_0 + √(1-α_t) * ε
        x_t = sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise

        return x_t, noise

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: Union[torch.Tensor, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: The initial data (B x ...)
            x_t: The noised data at timestep t
            t: Timesteps (B,) or single timestep

        Returns:
            Tuple containing:
                - Posterior mean
                - Posterior variance
        """
        alpha_t, _, _ = self.schedule.get_parameters(t)
        # Handle device compatibility for indexing
        if isinstance(t, torch.Tensor):
            device = t.device
            t_cpu = t.cpu()
            # Handle the t-1 indexing carefully
            t_minus_1 = torch.clamp(t_cpu - 1, min=0)
            alpha_tm1 = self.schedule.alphas_cumprod[t_cpu].to(device)
            # For t=0, use ones
            alpha_tm1 = torch.where(t == 0, torch.ones_like(alpha_t), alpha_tm1)
            beta_t = self.schedule.betas[t_cpu].to(device)
        else:
            alpha_tm1 = (
                self.schedule.alphas_cumprod[t - 1]
                if t > 0
                else torch.ones_like(alpha_t)
            )
            beta_t = self.schedule.betas[t]

        # Compute posterior mean coefficient for x_0 and x_t
        # Reshape parameters to match data dimensions for proper broadcasting
        # alpha_t, alpha_tm1, beta_t have shape (B,), need to reshape to (B, 1, ...) to match x_start/x_t
        alpha_t_expanded = alpha_t
        alpha_tm1_expanded = alpha_tm1
        beta_t_expanded = beta_t

        # Expand dimensions of alpha_t, alpha_tm1, beta_t to match x_start
        # This handles both scalar and 1D (batch) timesteps
        while alpha_t_expanded.dim() < x_start.dim():
            alpha_t_expanded = alpha_t_expanded.unsqueeze(-1)
            alpha_tm1_expanded = alpha_tm1_expanded.unsqueeze(-1)
            beta_t_expanded = beta_t_expanded.unsqueeze(-1)

        posterior_variance = (
            (1 - alpha_tm1_expanded) / (1 - alpha_t_expanded) * beta_t_expanded
        ).expand_as(x_start)
        posterior_mean = (
            torch.sqrt(alpha_tm1_expanded)
            * beta_t_expanded
            / (1 - alpha_t_expanded)
            * x_start
            + torch.sqrt(alpha_t_expanded)
            * (1 - alpha_tm1_expanded)
            / (1 - alpha_t_expanded)
            * x_t
        )

        return posterior_mean, posterior_variance


class ExcitedStateDiffusion:
    """
    Markov-style mask diffusion：基于全局mask noise实现 q(x_t | x_0)
    一旦mask就保持mask，不會逆轉，支持任意timestep采样。
    """

    def __init__(
        self,
        schedule: DiffusionSchedule,
        vocab_size: int,
        mask_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        device: Union[str, torch.device] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.schedule = schedule
        self.vocab_size = vocab_size
        self.mask_token_id = (
            mask_token_id if mask_token_id is not None else vocab_size - 1
        )
        self.pad_token_id = pad_token_id
        self.device = device

    def forward_mask_process(
        self,
        x_start: torch.Tensor,  # 原始token id (B, L)
        t: Union[int, torch.Tensor],
        mask_noise: Optional[torch.Tensor] = None,  # 固定的 mask noise (B, L)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用固定的mask noise，根據當前t計算哪些token被mask（不可逆）

        Returns:
            masked_tokens: (B, L)
            mask_positions: (B, L) float
        """
        if len(x_start.shape) == 3:
            token_ids = torch.argmax(x_start, dim=-1)
        elif len(x_start.shape) == 2:
            token_ids = x_start
        else:
            raise ValueError(f"输入维度不正确: {x_start.shape}")

        B, L = token_ids.shape

        # 構造 timestep tensor
        if isinstance(t, int):
            t = torch.full((B,), t, device=self.device, dtype=torch.long)
        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.expand(B).to(self.device)
        else:
            t = t.to(self.device)

        # 取得 mask 概率 gamma_t = 1 - alpha_t
        alpha_t, _, _ = self.schedule.get_parameters(t)
        gamma_t = (1.0 - alpha_t).unsqueeze(1).expand(B, L)

        # 如果沒有給定 mask_noise，就生成新的
        if mask_noise is None:
            mask_noise = torch.rand(B, L, device=self.device)

        # 決定哪些位置需要mask
        final_mask = mask_noise < gamma_t

        # 不mask pad token
        if self.pad_token_id is not None:
            final_mask = final_mask & (token_ids != self.pad_token_id)

        # 應用mask
        x_t = token_ids.clone()

        x_t[final_mask] = self.mask_token_id

        return x_t, final_mask.float()

    def get_mask_probabilities(self, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_t, _, _ = self.schedule.get_parameters(timesteps)
        return 1.0 - alpha_t

    def progressive_masking(
        self, x_start: torch.Tensor, timesteps: list, fixed_seed: Optional[int] = None
    ) -> list:
        """
        使用固定mask noise，觀察不同timestep下的mask變化過程
        """
        results = []

        if fixed_seed is not None:
            torch.manual_seed(fixed_seed)

        B, L = x_start.shape
        mask_noise = torch.rand(B, L, device=self.device)

        for t in timesteps:
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long)
            x_t, mask_pos = self.forward_mask_process(
                x_start, t_tensor, mask_noise=mask_noise
            )
            mask_prob = self.get_mask_probabilities(t_tensor)

            results.append(
                {
                    "timestep": t,
                    "masked_tokens": x_t.cpu(),
                    "mask_positions": mask_pos.cpu(),
                    "mask_probability": mask_prob.item(),
                    "num_masked": mask_pos.sum(dim=1).tolist(),
                }
            )

        return results
