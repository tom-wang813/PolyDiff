import logging
from typing import cast, Optional, Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertConfig

from ..diffusion.forward import ExcitedStateDiffusion
from ..diffusion.schedules import CosineSchedule
from ..model.bert_diffusion_model import BertDiffusionModel

logger = logging.getLogger(__name__)


class DiffusionTask(pl.LightningModule):
    """
    A PyTorch Lightning module for training a BERT-based diffusion model.

    This task encapsulates the model, the diffusion process, and the training logic.
    """

    def __init__(
        self,
        model_config: dict,
        diffusion_config: dict,
        optimizer_config: dict,
        tokenizer_name: str,
        inference_class: Any = None, # Added for custom development
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store the inference class for later use
        self.inference_class = inference_class

        # Initialize Model
        config = cast(BertConfig, BertConfig.from_dict(model_config))
        self.model = BertDiffusionModel(config)
        self.vocab_size = config.vocab_size

        # Initialize Tokenizer to get special token IDs
        # This is a temporary tokenizer just to get the IDs, as the main
        # tokenizer is in DataModule
        temp_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = temp_tokenizer.mask_token_id
        self.pad_token_id = temp_tokenizer.pad_token_id

        # Initialize Diffusion Schedule
        schedule_type = diffusion_config.get("schedule_type", "cosine")
        if schedule_type == "cosine":
            self.schedule = CosineSchedule(
                num_timesteps=diffusion_config.get("num_timesteps", 1000),
                s=diffusion_config.get("s", 0.008),  # Use default s if not provided
            )
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")

        # Initialize ExcitedStateDiffusion process
        self.diffusion_process = ExcitedStateDiffusion(
            schedule=self.schedule,
            vocab_size=self.vocab_size,
            mask_token_id=self.mask_token_id,
            pad_token_id=self.pad_token_id,
            device=self.device,  # Device will be set by Lightning Trainer
        )

        # Timestep embedding layer
        self.timestep_embedding = torch.nn.Embedding(
            self.schedule.num_timesteps, config.hidden_size
        )

        # Output projection layer
        self.output_projection = torch.nn.Linear(config.hidden_size, config.vocab_size)

    def setup(self, stage: str):
        """
        Called before training, validation, testing, or predicting.
        Ensures the diffusion process is on the correct device.
        """
        self.diffusion_process.device = self.device

    def _apply_masking(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Applies masking to the input_ids based on the timestep t using the diffusion process."""
        # The forward_mask_process returns (masked_tokens, mask_positions)
        # We only need the masked_tokens (xt)
        xt, _ = self.diffusion_process.forward_mask_process(input_ids, t)
        return xt

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        batch_size = input_ids.size(0)

        # 1. Sample a random timestep for each example in the batch
        t = torch.randint(
            0, self.schedule.num_timesteps, (batch_size,), device=self.device
        ).long()

        # 2. Apply forward diffusion (noise the data)
        # This is where the logic from the old dataset's _apply_masking will go.
        # For now, we'll use a placeholder.
        xt = self._apply_masking(input_ids, t)

        # 3. Get timestep embeddings
        timestep_embeds = self.timestep_embedding(t)

        # 4. Forward pass through the model
        model_output = self.model(
            input_ids=xt, attention_mask=attention_mask, timestep_embeds=timestep_embeds
        )
        hidden_state = model_output.last_hidden_state

        # 5. Project to vocabulary size
        logits = self.output_projection(hidden_state)

        # 6. Calculate loss
        # The target should be the original, un-noised input_ids
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), input_ids.view(-1))

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        batch_size = input_ids.size(0)

        t = torch.randint(
            0, self.schedule.num_timesteps, (batch_size,), device=self.device
        ).long()
        xt = self._apply_masking(input_ids, t)
        timestep_embeds = self.timestep_embedding(t)

        model_output = self.model(
            input_ids=xt, attention_mask=attention_mask, timestep_embeds=timestep_embeds
        )
        hidden_state = model_output.last_hidden_state
        logits = self.output_projection(hidden_state)

        loss = F.cross_entropy(logits.view(-1, self.vocab_size), input_ids.view(-1))

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        # Using AdamW optimizer as it's standard for transformers
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.optimizer_config.get("lr", 1e-4),
            weight_decay=self.hparams.optimizer_config.get("weight_decay", 0.01),
        )
        return optimizer

    def log_generated_samples(
        self,
        num_samples: int = 5,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        log_every_n_epochs: int = 1,
    ):
        """
        Generates and logs sample molecules during training.

        Args:
            num_samples (int): Number of molecules to generate.
            max_length (Optional[int]): Maximum sequence length for generated molecules.
            temperature (float): Sampling temperature.
            seed (Optional[int]): Random seed for reproducibility.
            log_every_n_epochs (int): Log samples every N epochs.
        """
        if self.trainer.current_epoch % log_every_n_epochs == 0:
            # Use the provided inference class
            _inference_class = self.inference_class

            logger.info(f"Generating {num_samples} sample molecules at epoch {self.trainer.current_epoch}...")

            # Create a dummy checkpoint path for inference (not actually used for loading)
            dummy_checkpoint_path = "dummy_path.ckpt"

            # Temporarily save the current model as a checkpoint
            self.trainer.save_checkpoint(dummy_checkpoint_path)

            inference_engine = _inference_class(
                checkpoint_path=dummy_checkpoint_path,
                device=str(self.device),
                diffusion_steps=self.schedule.num_timesteps,
            )

            generated_smiles = inference_engine.generate_molecules(
                num_samples=num_samples,
                max_length=max_length,
                temperature=temperature,
                seed=seed,
            )

            # Log samples to logger
            if self.trainer.logger:
                # Use experiment.log_text for MLflow or similar loggers
                if hasattr(self.trainer.logger, 'experiment'):
                    try:
                        self.trainer.logger.experiment.log_text(
                            "\n".join(generated_smiles),
                            f"generated_smiles_epoch_{self.trainer.current_epoch}.txt"
                        )
                    except Exception:
                        # Fallback to regular logging
                        self.log("generated_samples_count", len(generated_smiles))
                else:
                    self.log("generated_samples_count", len(generated_smiles))
            logger.info(f"Generated samples: {generated_smiles}")

            # Clean up dummy checkpoint
            import os
            if os.path.exists(dummy_checkpoint_path):
                os.remove(dummy_checkpoint_path)
