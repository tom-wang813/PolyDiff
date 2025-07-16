import torch
import pytest
from unittest.mock import MagicMock, patch
from polydiff.tasks.diffusion_task import DiffusionTask
from transformers import BertConfig
import pytorch_lightning as pl

# Mock BertDiffusionModel and AutoTokenizer for isolated testing
class MockBertDiffusionModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.last_hidden_state = torch.randn(2, 5, config.hidden_size)

    def forward(self, input_ids, attention_mask, timestep_embeds):
        # Simulate model output
        return MagicMock(last_hidden_state=self.last_hidden_state)

class MockAutoTokenizer:
    def __init__(self, tokenizer_name):
        self.mask_token_id = 101
        self.pad_token_id = 0

    @classmethod
    def from_pretrained(cls, tokenizer_name):
        instance = cls(tokenizer_name)
        return instance

@pytest.fixture
def mock_diffusion_task_configs():
    model_config = {"vocab_size": 1000, "hidden_size": 768}
    diffusion_config = {"schedule_type": "cosine", "num_timesteps": 100}
    optimizer_config = {"lr": 1e-4, "weight_decay": 0.01}
    tokenizer_name = "mock_tokenizer"
    return model_config, diffusion_config, optimizer_config, tokenizer_name

@patch('polydiff.tasks.diffusion_task.BertDiffusionModel', new=MockBertDiffusionModel)
@patch('polydiff.tasks.diffusion_task.AutoTokenizer', new=MockAutoTokenizer)
def test_diffusion_task_initialization(mock_diffusion_task_configs):
    model_config, diffusion_config, optimizer_config, tokenizer_name = mock_diffusion_task_configs
    task = DiffusionTask(model_config, diffusion_config, optimizer_config, tokenizer_name, inference_class=MagicMock())

    assert isinstance(task.model, MockBertDiffusionModel)
    assert task.vocab_size == model_config["vocab_size"]
    assert task.mask_token_id == 101
    assert task.pad_token_id == 0
    assert task.schedule.num_timesteps == diffusion_config["num_timesteps"]
    assert isinstance(task.timestep_embedding, torch.nn.Embedding)
    assert isinstance(task.output_projection, torch.nn.Linear)

@patch('polydiff.tasks.diffusion_task.BertDiffusionModel', new=MockBertDiffusionModel)
@patch('polydiff.tasks.diffusion_task.AutoTokenizer', new=MockAutoTokenizer)
def test_diffusion_task_training_step(mock_diffusion_task_configs):
    model_config, diffusion_config, optimizer_config, tokenizer_name = mock_diffusion_task_configs
    task = DiffusionTask(model_config, diffusion_config, optimizer_config, tokenizer_name, inference_class=MagicMock())
    task.trainer = MagicMock() # Mock trainer for logging

    batch = {
        "input_ids": torch.randint(0, model_config["vocab_size"], (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }

    loss = task.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0 # Loss should be non-negative

@patch('polydiff.tasks.diffusion_task.BertDiffusionModel', new=MockBertDiffusionModel)
@patch('polydiff.tasks.diffusion_task.AutoTokenizer', new=MockAutoTokenizer)
def test_diffusion_task_validation_step(mock_diffusion_task_configs):
    model_config, diffusion_config, optimizer_config, tokenizer_name = mock_diffusion_task_configs
    task = DiffusionTask(model_config, diffusion_config, optimizer_config, tokenizer_name, inference_class=MagicMock())
    task.trainer = MagicMock() # Mock trainer for logging

    batch = {
        "input_ids": torch.randint(0, model_config["vocab_size"], (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }

    loss = task.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0 # Loss should be non-negative

@patch('polydiff.tasks.diffusion_task.BertDiffusionModel', new=MockBertDiffusionModel)
@patch('polydiff.tasks.diffusion_task.AutoTokenizer', new=MockAutoTokenizer)
def test_diffusion_task_configure_optimizers(mock_diffusion_task_configs):
    model_config, diffusion_config, optimizer_config, tokenizer_name = mock_diffusion_task_configs
    task = DiffusionTask(model_config, diffusion_config, optimizer_config, tokenizer_name, inference_class=MagicMock())
    optimizer = task.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == optimizer_config['lr']
    assert optimizer.defaults['weight_decay'] == optimizer_config['weight_decay']

@patch('polydiff.inference.PolymerDiffusionInference')
@patch('polydiff.tasks.diffusion_task.logger')
def test_log_generated_samples(mock_logger, MockPolymerDiffusionInference, mock_diffusion_task_configs):
    model_config, diffusion_config, optimizer_config, tokenizer_name = mock_diffusion_task_configs
    task = DiffusionTask(model_config, diffusion_config, optimizer_config, tokenizer_name, inference_class=MockPolymerDiffusionInference)
    
    # Mock trainer and its logger
    mock_trainer = MagicMock()
    mock_trainer.current_epoch = 0
    mock_trainer.logger = MagicMock()
    mock_trainer.save_checkpoint = MagicMock()
    task.trainer = mock_trainer
    task.device = torch.device("cpu") # Set device for testing

    # Mock the inference engine instance
    mock_inference_instance = MockPolymerDiffusionInference.return_value
    mock_inference_instance.generate_molecules.return_value = ["mock_smiles_1", "mock_smiles_2"]

    # Test logging at epoch 0
    task.log_generated_samples(num_samples=2, log_every_n_epochs=1)
    mock_logger.info.assert_any_call("Generating 2 sample molecules at epoch 0...")
    mock_trainer.save_checkpoint.assert_called_once_with("dummy_path.ckpt")
    MockPolymerDiffusionInference.assert_called_once_with(
        checkpoint_path="dummy_path.ckpt",
        device="cpu",
        diffusion_steps=task.schedule.num_timesteps,
    )
    mock_inference_instance.generate_molecules.assert_called_once_with(
        num_samples=2,
        max_length=None,
        temperature=1.0,
        seed=None,
    )
    mock_trainer.logger.log_text.assert_called_once_with(
        key="generated_smiles",
        text="mock_smiles_1\nmock_smiles_2",
        step=0,
    )
    mock_logger.info.assert_any_call("Generated samples: ['mock_smiles_1', 'mock_smiles_2']")
    # Check if dummy checkpoint is removed
    import os
    assert not os.path.exists("dummy_path.ckpt")

    # Test not logging if not the correct epoch
    mock_trainer.current_epoch = 1
    mock_trainer.save_checkpoint.reset_mock()
    MockPolymerDiffusionInference.reset_mock()
    mock_inference_instance.generate_molecules.reset_mock()
    mock_trainer.logger.log_text.reset_mock()
    mock_logger.info.reset_mock()

    task.log_generated_samples(num_samples=2, log_every_n_epochs=2)
    mock_logger.info.assert_not_called()
    mock_trainer.save_checkpoint.assert_not_called()
    MockPolymerDiffusionInference.assert_not_called()
    mock_inference_instance.generate_molecules.assert_not_called()
    mock_trainer.logger.log_text.assert_not_called()
