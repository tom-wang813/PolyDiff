from pathlib import Path

# 找到 PolyDiff/ 目錄
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_STATE_DIR     = PROJECT_ROOT / "model_state"
CHECKPOINT_DIR      = MODEL_STATE_DIR / "checkpoint"
GRADIENT_SAVE_DIR   = MODEL_STATE_DIR / "gradient_saves"

# 確保資料夾存在（也可以放到 TrainingStateManager 裡統一建立）
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
GRADIENT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Saving interval settings
SAVE_INTERVAL = 1000  # Checkpoint saving interval (in steps)
SNAPSHOT_INTERVAL = 500  # Snapshot saving interval (in steps)
GRADIENT_SAVE_INTERVAL = 500  # Gradient saving interval (in steps)

# Training hyperparameters
LEARNING_RATE = 1e-3  # Learning rate
BATCH_SIZE = 32  # Batch size
EPOCHS = 100  # Total epochs

# Maximum number of gradient files
MAX_GRAD_FILES = 3

# Early stopping settings
EARLY_STOPPING_PATIENCE = 5  # Number of epochs with no improvement after which training will be stopped
EARLY_STOPPING_DELTA = 0.01  # Minimum change to qualify as an improvement
