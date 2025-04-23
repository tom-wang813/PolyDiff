# PolyDiff/configs/train_config.py
from pathlib import Path

# PolyDiff project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


MODEL_STATE_DIR = PROJECT_ROOT / "model_state"
CHECKPOINT_DIR = MODEL_STATE_DIR / "checkpoint"
GRADIENT_SAVE_DIR = MODEL_STATE_DIR / "gradient_saves"

# Ensure directories exist (can also be handled in TrainingStateManager)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
GRADIENT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Saving interval settings
SAVE_INTERVAL = 1000  # Interval for saving checkpoints (in steps)
SNAPSHOT_INTERVAL = 500  # Interval for saving snapshots (in steps)
GRADIENT_SAVE_INTERVAL = 500  # Interval for saving gradients (in steps)

# Training hyperparameters
LEARNING_RATE = 1e-3  # Learning rate
BATCH_SIZE = 32  # Batch size
EPOCHS = 100  # Total number of epochs

# Maximum number of gradient files
MAX_GRAD_FILES = 3

# Early stopping settings
EARLY_STOPPING_PATIENCE = 5  # Number of epochs with no improvement before stopping
EARLY_STOPPING_DELTA = 0.01  # Minimum improvement to reset patience
