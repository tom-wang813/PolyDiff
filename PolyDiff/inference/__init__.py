"""
Inference module for PolyDiff.

This package contains classes and functions for performing inference with the trained diffusion models.
"""

__all__ = ["PolymerDiffusionInference"]

import logging
import os
from typing import Any, Dict, List, Optional, Union, cast

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig

from ..diffusion.schedules import (CosineSchedule, DiffusionSchedule,
                                   LinearSchedule)
from ..model.bert_diffusion_model import BertDiffusionModel
from ..tasks.diffusion_task import DiffusionTask

logger = logging.getLogger(__name__)


class PolymerDiffusionInference:
    """
    Inference class for polymer diffusion model generation.
    Loads a trained DiffusionTask checkpoint and performs molecule generation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        schedule_type: Optional[str] = None,
        diffusion_steps: Optional[int] = None,
    ):
        """
        Initialize the inference class by loading a trained DiffusionTask checkpoint.

        Args:
            checkpoint_path (str): Path to the trained DiffusionTask checkpoint.
            device (str): Device to run inference on (
            'auto', 'cuda', 'cpu', 'mps').
            schedule_type (str, optional): Override the schedule type from checkpoint.
            diffusion_steps (int, optional): Override the number of diffusion steps.
        """
        self.device = self._get_device(device)
        self.checkpoint_path = checkpoint_path

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

        # Load the DiffusionTask from checkpoint
        # map_location ensures it loads onto the correct device
        try:
            self.task = DiffusionTask.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path, map_location=self.device
            )
            self.task.eval()  # Set model to evaluation mode
            self.task.freeze()  # Freeze parameters
        except Exception as e:
            raise RuntimeError(f"Failed to load DiffusionTask from checkpoint: {e}")

        self.model = self.task.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            cast(dict, self.task.hparams)["tokenizer_name"]
        )

        # Determine diffusion steps and schedule
        self.diffusion_steps = (
            diffusion_steps
            if diffusion_steps is not None
            else cast(dict, self.task.hparams)["diffusion_config"]["num_timesteps"]
        )
        self.schedule_type = (
            schedule_type
            if schedule_type is not None
            else cast(dict, self.task.hparams)["diffusion_config"]["schedule_type"]
        )

        self.schedule: DiffusionSchedule
        if self.schedule_type == "linear":
            self.schedule = LinearSchedule(num_timesteps=self.diffusion_steps)
        elif self.schedule_type == "cosine":
            self.schedule = CosineSchedule(num_timesteps=self.diffusion_steps)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Ensure the diffusion process in the task is on the correct device
        self.task.diffusion_process.device = self.device

        logger.info(f"Inference setup complete. Model loaded on {self.device}")
        logger.info(f"Model config: {self.model.config}")
        logger.info(f"Diffusion steps: {self.diffusion_steps}")
        logger.info(f"Schedule type: {self.schedule_type}")

    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch, "has_mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def generate_molecules(
        self,
        num_samples: int = 1,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Generate new molecules using the diffusion model.

        Args:
            num_samples (int): Number of molecules to generate.
            max_length (int, optional): Maximum sequence length for generated molecules.
                                        If None, uses model's max_position_embeddings.
            temperature (float): Sampling temperature (higher = more random).
            seed (int, optional): Random seed for reproducibility.

        Returns:
            List[str]: Generated SMILES strings.
        """
        if seed is not None:
            torch.manual_seed(seed)
            # np.random.seed(seed) # numpy not used for sampling directly

        # Use model's max_position_embeddings if max_length is not specified
        if max_length is None:
            max_length = self.model.config.max_position_embeddings
        else:
            max_length = min(max_length, self.model.config.max_position_embeddings)

        with torch.no_grad():
            # Start with pure noise (mask tokens)
            # Initialize with mask tokens, then gradually denoise
            input_ids = torch.full(
                (num_samples, max_length),
                self.tokenizer.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.ones(
                num_samples, max_length, device=self.device, dtype=torch.long
            )

            # Reverse diffusion process
            for t in tqdm(
                range(self.diffusion_steps - 1, -1, -1), desc="Denoising molecules"
            ):
                # Current timestep
                timesteps = torch.full(
                    (num_samples,), t, device=self.device, dtype=torch.long
                )

                # Get timestep embeddings
                timestep_embeds = self.task.timestep_embedding(timesteps)

                # Model prediction (predict original x_0 from x_t)
                model_output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    timestep_embeds=timestep_embeds,
                )
                predicted_logits = self.task.output_projection(
                    model_output.last_hidden_state
                )

                # Apply temperature scaling
                if temperature != 1.0:
                    predicted_logits = predicted_logits / temperature

                # Sample from the predicted distribution
                probs = F.softmax(predicted_logits, dim=-1)
                # Sample tokens from the predicted probability distribution
                sampled_tokens = torch.multinomial(
                    probs.view(-1, probs.size(-1)), 1
                ).view(num_samples, max_length)

                # For the next step, we use the sampled tokens as the new input_ids
                # This is a simple sampling strategy. More advanced strategies (e.g., DDIM)
                # would involve predicting x_0 and then sampling x_{t-1}.
                input_ids = sampled_tokens

            # Decode final tokens to SMILES
            generated_smiles = []
            for i in range(num_samples):
                tokens = input_ids[i].cpu().numpy()
                smiles = self.tokenizer.decode(tokens, skip_special_tokens=True)
                generated_smiles.append(smiles.strip())

        return generated_smiles

    def generate_molecules_batch(
        self,
        num_batches: int,
        samples_per_batch: int = 1,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Generates multiple batches of new molecules using the diffusion model.

        Args:
            num_batches (int): The number of batches of molecules to generate.
            samples_per_batch (int): Number of molecules to generate per batch.
            max_length (int, optional): Maximum sequence length for generated molecules.
                                        If None, uses model's max_position_embeddings.
            temperature (float): Sampling temperature (higher = more random).
            seed (int, optional): Random seed for reproducibility.

        Returns:
            List[str]: A flattened list of all generated SMILES strings from all batches.
        """
        all_generated_smiles = []
        for i in tqdm(range(num_batches), desc="Generating batches"): # noqa: E501
            # Use a unique seed for each batch if a base seed is provided
            batch_seed = seed + i if seed is not None else None
            generated_smiles = self.generate_molecules(
                num_samples=samples_per_batch,
                max_length=max_length,
                temperature=temperature,
                seed=batch_seed,
            )
            all_generated_smiles.extend(generated_smiles)
        return all_generated_smiles