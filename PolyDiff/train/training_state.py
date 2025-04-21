import torch
import os
from collections import deque
from PolyDiff.configs import train_config  # 直接從train_config中導入

class TrainingStateManager:
    def __init__(self, model, optimizer, scheduler, initial_step=0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step = initial_step

        # 儲存路徑設置
        self.checkpoint_dir = train_config.CHECKPOINT_DIR
        self.gradient_save_dir = train_config.GRADIENT_SAVE_DIR

        # 儲存間隔設置
        self.save_interval = train_config.SAVE_INTERVAL
        self.gradient_save_interval = train_config.GRADIENT_SAVE_INTERVAL
        self.max_grad_files = train_config.MAX_GRAD_FILES
        self.grad_files = deque(maxlen=self.max_grad_files)

        # 創建保存目錄
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.gradient_save_dir, exist_ok=True)

    def _get_filename(self, file_type, step):
        """生成文件名，根據文件類型（checkpoint、gradient）和步數生成唯一的文件名"""
        return f"{file_type}_step_{step}.pth"

    def _get_file_path(self, file_type, step, directory):
        """根據文件類型和步數生成完整路徑"""
        filename = self._get_filename(file_type, step)
        return os.path.join(directory, filename)

    def _save(self, data, file_type, directory):
        """通用保存方法，根據文件類型和目錄保存數據"""
        path = self._get_file_path(file_type, self.step, directory)
        torch.save(data, path)
        print(f"Saved {file_type} at step {self.step} to {path}")

    def save_checkpoint(self):
        """保存模型、優化器狀態及超參數"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'learning_rate': self.optimizer.param_groups[0]['lr'],  # 從優化器獲取學習率
            'batch_size': train_config.BATCH_SIZE,  # 可以從config中獲取其他超參數
            'epochs': train_config.EPOCHS
        }
        self._save(checkpoint, "checkpoint", self.checkpoint_dir)

    def save_gradients(self):
        """保存最近的梯度文件"""
        grad_data = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_data[name] = param.grad.clone()
        self._save(grad_data, "gradients", self.gradient_save_dir)

    def update(self, step):
        """更新訓練步數並根據設定的間隔保存檢查點和梯度"""
        self.step = step

        # 確保每個間隔保存一次相應的文件
        if self.step % self.save_interval == 0:
            self.save_checkpoint()
        if self.step % self.gradient_save_interval == 0:
            self.save_gradients()

class CheckpointLoader:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def load_checkpoint(self, step):
        """從保存的檢查點中加載模型、優化器狀態"""
        checkpoint_path = os.path.join(train_config.CHECKPOINT_DIR, f"checkpoint_step_{step}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.step = checkpoint['step']
            self.learning_rate = checkpoint['learning_rate']
            self.batch_size = checkpoint['batch_size']
            self.epochs = checkpoint['epochs']
            print(f"Loaded checkpoint from {checkpoint_path}, resuming from step {step}")
            return self.step
        else:
            print(f"No checkpoint found at step {step}, starting from scratch.")
            return step

if __name__ == "__main__":
    # 測試代碼區域，這裡創建模型，優化器，調度器，並進行簡單的檢查
    model = torch.nn.Linear(10, 10)  # 假設用一個簡單的線性層作為示例模型
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    # 創建 TrainingStateManager 實例
    manager = TrainingStateManager(model, optimizer, scheduler)

    # 測試保存檢查點
    manager.step = 1000  # 假設訓練到步數1000
    manager.save_checkpoint()
    manager.save_gradients()

    # 測試加載檢查點
    loader = CheckpointLoader(model, optimizer, scheduler)
    loader.load_checkpoint(1000)