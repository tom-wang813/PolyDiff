import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from PolyDiff.train.training_state import TrainingStateManager  # Assuming you have this class in functions
from PolyDiff.configs import train_config

class EarlyStopping:
    def __init__(self):
        self.patience = train_config.EARLY_STOPPING_PATIENCE
        self.delta = train_config.EARLY_STOPPING_DELTA
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

class Trainer:
    def __init__(self, model, optimizer: Optimizer, scheduler, dataloader: DataLoader, loss_fn: dict, 
                 checkpoint_dir, device, max_epochs=100, patience=10):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.max_epochs = max_epochs
        self.epoch = 0

        # Early stopping setup
        self.early_stopping = EarlyStopping(patience=patience)

        # Initialize the training state manager
        self.state_manager = TrainingStateManager(checkpoint_dir)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss for each task (multi-task)
            loss = 0
            for task_name, task_loss_fn in self.loss_fn.items():
                task_loss = task_loss_fn(outputs[task_name], targets[task_name])
                loss += task_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate total loss
            total_loss += loss.item()

        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch {self.epoch}: Training Loss: {avg_loss}")
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = 0
                for task_name, task_loss_fn in self.loss_fn.items():
                    task_loss = task_loss_fn(outputs[task_name], targets[task_name])
                    loss += task_loss
                
                total_loss += loss.item()

        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch {self.epoch}: Validation Loss: {avg_loss}")
        return avg_loss

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def train(self):
        for self.epoch in range(self.epoch, self.max_epochs):
            # Training step
            train_loss = self.train_one_epoch()

            # Validation step
            val_loss = self.validate()

            # Step the scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Check early stopping condition
            if self.early_stopping(val_loss):
                print(f"Early stopping at epoch {self.epoch}")
                break

            # Save checkpoint periodically
            if self.epoch % 10 == 0:
                self.save_checkpoint()

