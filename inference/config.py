# inference/config.py
from __future__ import annotations

from dataclasses import dataclass
import yaml


@dataclass
class InferenceConfig:
    """
    推理流程的超參數與路徑。
    會從 YAML 讀入。
    """
    checkpoint_path: str       # 要載入的 checkpoint 檔案
    device: str = "cpu"        # "cpu" / "cuda:0" / "mps"
    num_samples: int = 10      # 一次要 sample 幾筆
    seq_len: int | None = None # 序列長度，None 則用 model_config.MAX_SEQ_LENGTH
    temperature: float = 1.0
    top_k: int | None = None   # top-k sampling，如不指定則不做截斷
    output_path: str = "samples.json"  # 輸出結果路徑

    @classmethod
    def from_yaml(cls, path: str) -> InferenceConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
