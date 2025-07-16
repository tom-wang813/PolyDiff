# Tasks Module (`polydiff/tasks`)

The `polydiff/tasks` module defines the PyTorch Lightning `LightningModule` for training the BERT-based discrete diffusion model. This module encapsulates the model, the diffusion process, and the entire training and validation logic, making it easy to train and manage experiments.

## Key Components

### `DiffusionTask`

`DiffusionTask` is the central `LightningModule` that orchestrates the training of the diffusion model. It integrates the `BertDiffusionModel`, the `ExcitedStateDiffusion` process, and the chosen `DiffusionSchedule`.

-   **Purpose:** To provide a self-contained training loop for the diffusion model, handling forward passes, loss calculation, optimization, and logging.
-   **Initialization (`__init__`):**
    -   `model_config` (dict): Configuration parameters for the `BertDiffusionModel` (e.g., `vocab_size`, `hidden_size`).
    -   `diffusion_config` (dict): Configuration for the diffusion process (e.g., `num_timesteps`, `schedule_type`).
    -   `optimizer_config` (dict): Configuration for the optimizer (e.g., `lr`, `weight_decay`).
    -   `tokenizer_name` (str): The name of the tokenizer used, primarily to get special token IDs.
-   **Internal Attributes:**
    -   `self.model`: An instance of `BertDiffusionModel`.
    -   `self.vocab_size`: Vocabulary size from the model configuration.
    -   `self.mask_token_id`, `self.pad_token_id`: Special token IDs from the tokenizer.
    -   `self.schedule`: The initialized diffusion schedule (e.g., `CosineSchedule`).
    -   `self.diffusion_process`: An instance of `ExcitedStateDiffusion`.
    -   `self.timestep_embedding`: An embedding layer for timesteps.
    -   `self.output_projection`: A linear layer to project model output to vocabulary size.
-   **Methods:**
    -   `setup(stage)`: Called before training, validation, testing, or predicting. Ensures the diffusion process is on the correct device.
    -   `_apply_masking(input_ids, t)`: Internal helper to apply the forward masking process to input IDs based on the timestep `t`.
    -   `training_step(batch, batch_idx)`: Defines the logic for a single training step, including sampling timesteps, applying forward diffusion, model prediction, and loss calculation.
    -   `validation_step(batch, batch_idx)`: Defines the logic for a single validation step, similar to `training_step` but without backpropagation.
    -   `configure_optimizers()`: Configures and returns the optimizer (e.g., AdamW) for training.

## Development Guide

### Customizing Training Logic

-   **Loss Function:** To use a different loss function, modify the `training_step` and `validation_step` methods to calculate and log the new loss.
-   **Sampling Timesteps:** The current implementation samples random timesteps. Alternative sampling strategies (e.g., importance sampling) can be implemented by modifying the timestep sampling logic in `training_step`.
-   **Model Output Processing:** If the `BertDiffusionModel`'s output format changes, or if additional processing is needed before calculating `predicted_logits`, modify the relevant parts of `training_step` and `validation_step`.

### Integrating New Diffusion Processes or Schedules

-   If you implement a new `DiffusionSchedule` or a new diffusion process (e.g., a different `DiffusionForward` variant), you would update the `__init__` method of `DiffusionTask` to instantiate and use the new components based on the `diffusion_config`.

### Optimizer Configuration

-   To use a different optimizer or customize learning rate schedules, modify the `configure_optimizers` method. PyTorch Lightning provides various built-in learning rate schedulers that can be integrated here.

## Testing

-   **Unit Tests:** Refer to `tests/tasks/test_diffusion_task.py` for examples of how to test the `DiffusionTask`. It's crucial to mock external dependencies like `BertDiffusionModel` and `AutoTokenizer` to ensure isolated and fast tests.
-   **Mocking Strategies:** Pay close attention to how `MagicMock` is used to simulate the behavior of the model, tokenizer, and other components. This allows testing the training and validation steps without needing a full end-to-end run.
-   **Coverage:** When making changes, run `pytest --cov=polydiff.tasks --cov-report=term-missing` to ensure your changes are adequately covered by tests.
