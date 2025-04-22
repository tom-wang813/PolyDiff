# inference/sampler.py
from typing import Optional

import torch

from PolyDiff.configs import model_config


from __future__ import annotations

"""PolyDiff – discrete diffusion sampling utilities

A simple **mask‑based backward sampler** for text / SMILES diffusion.
This is *not* PLMS / DDIM; rather a straightforward iterative procedure
sufficient for first experiments. Feel free to swap with an advanced
sampler later.
"""

from typing import List, Optional

import torch
from torch import nn
from tqdm.auto import trange

from PolyDiff.configs import model_config, diffusion_config
from PolyDiff.diffusion.diffusion_forward import BaseSchedule

__all__ = ["DiscreteSampler"]


class DiscreteSampler:
    def __init__(
        self,
        model: nn.Module,  # DiffusionBertModel
        schedule: BaseSchedule,
        mask_token_id: int = model_config.MASK_TOKEN_ID,
        pad_token_id: int = model_config.PAD_TOKEN_ID,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.schedule = schedule
        self.T = schedule.num_steps
        self.mask_token = mask_token_id
        self.pad_token = pad_token_id
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _init_sequence(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Start from all [MASK] (except [PAD] if desired)."""
        x = torch.full((batch_size, seq_len), self.mask_token, dtype=torch.long, device=self.device)
        return x

    # ------------------------------------------------------------------
    # main public api
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate discrete sequences by reversing the absorbing process."""

        x = self._init_sequence(batch_size, seq_len)

        for t_inv in trange(self.T - 1, -1, -1, desc="Sampling", leave=False):
            t = torch.full((batch_size,), t_inv, dtype=torch.long, device=self.device)
            logits = self.model(x, t)  # (B, L, vocab)
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = logits.topk(top_k, dim=-1)
                probs = torch.full_like(logits, float("-inf"))
                probs.scatter_(-1, topk_idx, topk_vals)
                logits = probs
            probs = torch.softmax(logits, dim=-1)
            # sample tokens only at MASK positions
            mask_positions = x == self.mask_token
            if mask_positions.any():
                sampled = torch.multinomial(probs[mask_positions], num_samples=1).squeeze(-1)
                x[mask_positions] = sampled

            # Optional: re‑mask some tokens according to schedule.beta to mimic stochasticity (not implemented here)

        return x.cpu()


def sample_model(
    model: torch.nn.Module,
    schedule,
    num_samples: int,
    seq_len: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    使用 DiscreteSampler 做純推理，回傳 shape=(num_samples, seq_len) 的長整數 tensor。
    """
    sampler = DiscreteSampler(
        model=model,
        schedule=schedule,
        mask_token_id=model_config.MASK_TOKEN_ID,
        pad_token_id=model_config.PAD_TOKEN_ID,
        device=device,
    )
    return sampler.sample(
        batch_size=num_samples,
        seq_len=seq_len,
        temperature=temperature,
        top_k=top_k,
    )
