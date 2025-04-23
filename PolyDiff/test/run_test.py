import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, TensorDataset
from PolyDiff.train.get_trainer import get_trainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*?")

# helper: 生成輸出 dict 的 DataLoader
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int, in_features: int, out_features: int):
        self.x = torch.randn(n_samples, in_features)
        self.y = torch.randint(0, out_features, (n_samples,), dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        # 返回 dict，以符合 Trainer.expect 預期
        return {"input": self.x[idx], "labels": {"target": self.y[idx]}}


def make_dummy_loader(batch_size: int, in_features: int, out_features: int, n_samples: int = 100):
    dataset = DummyDataset(n_samples, in_features, out_features)
    return DataLoader(dataset, batch_size=batch_size)

class TestModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(TestModel, self).__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return {"target": x}
    
@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    # 由 cfg.data 自動 instantiate train/val loader
    trainer = get_trainer(cfg)
    # 執行訓練
    print(f"trainer.loss_fns is {trainer.loss_fns}")
    metrics = trainer.fit()

if __name__ == "__main__":
    main()
