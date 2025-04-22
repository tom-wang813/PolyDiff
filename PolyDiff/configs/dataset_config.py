from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent  # PolyDiff/

# Data Paths
DATA_DIR: Path = PROJECT_ROOT / "data"
TRAIN_SPLIT: str = "train"  # matches train0.txt, train_01.txt …
VAL_SPLIT: str = "val"      # matches val.txt, val0.txt …

# Loader hyper‑parameters
BATCH_SIZE: int = 32
NUM_WORKERS: int = 8  # adjust to CPU cores; 0 for debugging
SHUFFLE_FILES: bool = True  # True → randomise file order each epoch

# Sequences longer than this will be truncated by the tokenizer.
MAX_SEQ_LENGTH: int = 512
