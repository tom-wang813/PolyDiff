# inference/utils.py
import torch
from PolyDiff.model import DiffusionBertModel
from PolyDiff.configs import model_config


def load_model(checkpoint_path: str, device: str = "cpu") -> DiffusionBertModel:
    """
    1. 實例化模型
    2. torch.load checkpoint["model_state_dict"]
    3. model.to(device).eval()
    """
    model = DiffusionBertModel()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model
