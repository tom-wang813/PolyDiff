#!/usr/bin/env python3
"""
Training Script for BERT-based Discrete Diffusion Model
======================================================

A high-level training script that uses PyTorch Lightning to train
a BERT-based discrete diffusion model for polymer SMILES generation.

Usage:
    # Using new hierarchical config system
    python scripts/train.py --config configs/experiments/polymer_generation.yaml
    python scripts/train.py --config configs/experiments/quick_test.yaml
    
    # Legacy support
    python scripts/train.py --config experiments/model-1/configs/model-1.yaml
"""

import mlflow
import mlflow.pytorch
import argparse
from typing import Union
import logging
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig, ListConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from polydiff.data.datamodule import SmilesDataModule
from polydiff.tasks.diffusion_task import DiffusionTask
from polydiff.inference import PolymerDiffusionInference

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str) -> Union[DictConfig, ListConfig]:
    """Load configuration with support for hierarchical configs."""
    config_path_obj = Path(config_path)
    
    # Check if this is a new hierarchical config
    if "configs/" in str(config_path_obj) and config_path_obj.exists():
        logging.info(f"Loading hierarchical configuration from {config_path_obj}")
        config = OmegaConf.load(config_path_obj)
        
        # Handle defaults (inheritance)
        if "defaults" in config:
            base_configs = []
            for default in config.defaults:
                if "/" in default:
                    # Model variant or nested config
                    default_path = config_path_obj.parent.parent / f"{default}.yaml"
                else:
                    # Base config
                    default_path = config_path_obj.parent.parent / f"{default}.yaml"
                
                if default_path.exists():
                    base_config = OmegaConf.load(default_path)
                    base_configs.append(base_config)
                else:
                    logging.warning(f"Default config not found: {default_path}")
            
            # Merge configs (later configs override earlier ones)
            if base_configs:
                merged_config = base_configs[0]
                for base_config in base_configs[1:]:
                    merged_config = OmegaConf.merge(merged_config, base_config)
                
                # Current config overrides all defaults
                config = OmegaConf.merge(merged_config, config)
                
                # Remove defaults key from final config
                if "defaults" in config and isinstance(config, DictConfig):
                    del config["defaults"]
        
        return config
    else:
        # Legacy config loading
        logging.info(f"Loading legacy configuration from {config_path_obj}")
        return OmegaConf.load(config_path_obj)


def main(config_path: str) -> None:
    """Main training function.

    Args:
        config_path: Path to the YAML configuration file.
    """
    logging.info(f"Starting training with config: {config_path}")
    config = load_config(config_path)

    # --- Setup DataModule ---
    data_config = config.data
    dm = SmilesDataModule(
        data_path=data_config.data_path,
        tokenizer_name=data_config.tokenizer_name,
        train_split=data_config.train_split,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        max_length=data_config.max_length,
        val_subset_ratio=data_config.val_subset_ratio,
    )

    # --- Setup Diffusion Task (LightningModule) ---
    model_config = config.model
    diffusion_config = config.diffusion
    optimizer_config = config.optimizer

    task = DiffusionTask(
        model_config=model_config,
        diffusion_config=diffusion_config,
        optimizer_config=optimizer_config,
        tokenizer_name=data_config.tokenizer_name,
        inference_class=PolymerDiffusionInference, # Pass the inference class
    )

    # --- Setup Callbacks and Logger ---
    callbacks: list[pl.Callback] = []
    if config.checkpointing.enable:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.checkpointing.dirpath,
            filename=config.checkpointing.filename,
            monitor=config.checkpointing.monitor,
            mode=config.checkpointing.mode,
            save_top_k=config.checkpointing.save_top_k,
            save_last=config.checkpointing.save_last,
        )
        callbacks.append(checkpoint_callback)

    if config.logging.enable:
        tb_logger: TensorBoardLogger | bool = TensorBoardLogger(
            save_dir=config.logging.save_dir,
            name=config.logging.name,
            version=config.logging.version,
        )
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        logger = tb_logger
    else:
        logger = False  # Disable logger if not enabled in config

    # --- Setup Trainer ---
    trainer_config = config.trainer
    pl_trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        max_epochs=trainer_config.max_epochs,
        logger=logger,
        callbacks=callbacks,
        # Add other trainer arguments from config as needed
        # e.g., gradient_clip_val, accumulate_grad_batches, precision
    )

    # --- Start Training ---
    logging.info("Starting training...")
    with mlflow.start_run():
        # Log hyperparameters
        config_dict = OmegaConf.to_container(config, resolve=True)
        if isinstance(config_dict, dict):
            # Ensure all keys are strings
            params_dict = {str(k): v for k, v in config_dict.items()}
            mlflow.log_params(params_dict)

        pl_trainer.fit(task, datamodule=dm)

        # Log generated samples periodically
        if hasattr(config, "generation") and config.generation.enable:
            task.log_generated_samples(
                num_samples=config.generation.num_samples,
                max_length=config.generation.max_length,
                temperature=config.generation.temperature,
                seed=config.generation.seed,
                log_every_n_epochs=config.generation.log_every_n_epochs,
            )

        # Log metrics from trainer
        metrics = pl_trainer.callback_metrics
        if metrics:
            # Convert tensor values to float and ensure all values are floats
            float_metrics = {}
            for k, v in metrics.items():
                if hasattr(v, 'item'):
                    float_metrics[str(k)] = float(v.item())
                elif isinstance(v, (int, float)):
                    float_metrics[str(k)] = float(v)
            if float_metrics:
                mlflow.log_metrics(float_metrics)

        # Log the model
        mlflow.pytorch.log_model(task, "model")

    logging.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BERT Diffusion Model Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()
    main(args.config)
