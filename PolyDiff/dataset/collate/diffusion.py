# PolyDiff/dataset/collate/diffusion.py
from __future__ import annotations

from typing import List

import torch

from PolyDiff.dataset.base import register_collate
from PolyDiff.dataset.collate.pad import PadCollator
from PolyDiff.diffusion import AbsorbingDiffusion
from PolyDiff.configs import diffusion_config


@register_collate("diffusion")
class DiffusionCollator:
    """
    Compose **padding** + **forward‑diffusion** in one callable.

    Returns
    -------
    xt : Tensor[B, L]
    targets : dict
        { "labels": xt_minus1 ( -100 masked), "t": timestep }
    """

    def __init__(
        self,
        *,
        schedule,           
        mask_token: int,
        pad_token: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.pad = PadCollator(pad_token)
        self.q = AbsorbingDiffusion(
            num_steps=schedule.num_steps,
            mask_token=mask_token,
            schedule=schedule,
            device=device,
            pad_token=pad_token,
        )
        self.T_MAX = schedule.num_steps
        self.pad_token = pad_token
        self.device = torch.device(device)

    # ------------------------------------------------------------------ #
    def __call__(self, batch: List[torch.Tensor]):
        # 1) pad to rectangular [B, L]
        x0 = self.pad(batch).to(self.device)              # (B, L)

        # 2) sample random t ∈ [1, T‑1]
        t = torch.randint(
            1, self.T_MAX, (x0.size(0),),
            dtype=torch.long, device=self.device
        )

        # 3) forward diffusion
        xt        = self.q(x0, t)
        xt_minus1 = self.q(x0, t - 1)

        # 4) build labels (ignore PAD with -100 for CE)
        labels = xt_minus1.clone()
        labels[labels == self.pad_token] = -100

        targets = {"labels": labels, "t": t}
        return xt, targets
