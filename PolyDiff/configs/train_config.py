# Path settings
CHECKPOINT_DIR = "/home/wangziwei/graphdiffusion/PolyDiff/model_state/checkpoint"  # Directory for saving checkpoints
GRADIENT_SAVE_DIR = "/home/wangziwei/graphdiffusion/PolyDiff/model_state/gradient_saves"  # Directory for saving gradients

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
