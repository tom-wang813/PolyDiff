import torch
import pytest
from unittest.mock import MagicMock, patch
from polydiff.inference import PolymerDiffusionInference
from polydiff.tasks.diffusion_task import DiffusionTask
from transformers import AutoTokenizer, BertConfig

@pytest.fixture
def polymer_diffusion_inference_instance():
    mock_task = MagicMock()
    mock_task.model = MagicMock()
    mock_task.model.config = MagicMock(max_position_embeddings=512)
    mock_task.timestep_embedding = MagicMock(return_value=torch.randn(1, 768)) # hidden_size
    mock_task.output_projection = MagicMock(return_value=torch.randn(1, 1000)) # vocab_size
    mock_task.diffusion_process = MagicMock()
    mock_task.hparams = MagicMock()
    mock_task.hparams.__getitem__.side_effect = lambda key: {
        "tokenizer_name": "mock_tokenizer",
        "diffusion_config": {"num_timesteps": 100, "schedule_type": "cosine"}
    }[key]
    mock_task.eval = MagicMock()
    mock_task.freeze = MagicMock()

    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.mask_token_id = 101
    mock_tokenizer_instance.pad_token_id = 0
    mock_tokenizer_instance.model_max_length = 512
    mock_tokenizer_instance.decode.return_value = "mock_smiles"
    mock_tokenizer_instance.return_value = MagicMock(input_ids=torch.randint(0, 1000, (1, 512)), attention_mask=torch.ones(1, 512))

    with patch('polydiff.inference.inference.DiffusionTask.load_from_checkpoint', return_value=mock_task), \
         patch('polydiff.tasks.diffusion_task.AutoTokenizer.from_pretrained', return_value=mock_tokenizer_instance), \
         patch('polydiff.inference.inference.AutoTokenizer.from_pretrained', return_value=mock_tokenizer_instance), \
         patch('os.path.exists', return_value=True):
        instance = PolymerDiffusionInference(checkpoint_path="mock_path")
        return instance


def test_polymer_diffusion_inference_initialization(polymer_diffusion_inference_instance):
    instance = polymer_diffusion_inference_instance
    assert isinstance(instance.model, MagicMock)
    assert isinstance(instance.tokenizer, MagicMock)
    assert instance.diffusion_steps == 100
    assert instance.schedule_type == "cosine"

@pytest.mark.parametrize("num_samples, max_length", [(1, 10), (2, 20), (1, None)])
def test_polymer_diffusion_inference_generate_molecules(polymer_diffusion_inference_instance, num_samples, max_length):
    instance = polymer_diffusion_inference_instance
    
    # Dynamically set the shape of last_hidden_state based on max_length
    effective_max_length = max_length if max_length is not None else instance.tokenizer.model_max_length
    instance.model.return_value = MagicMock(last_hidden_state=torch.randn(num_samples, effective_max_length, 768))
    instance.task.output_projection = MagicMock(return_value=torch.randn(num_samples, effective_max_length, 1000))

    generated_smiles = instance.generate_molecules(num_samples=num_samples, max_length=max_length)
    assert isinstance(generated_smiles, list)
    assert len(generated_smiles) == num_samples
    assert all(s == "mock_smiles" for s in generated_smiles)

    # Test with seed
    generated_smiles_seeded = instance.generate_molecules(num_samples=num_samples, max_length=max_length, seed=42)
    assert isinstance(generated_smiles_seeded, list)

    # Test with temperature
    generated_smiles_temp = instance.generate_molecules(num_samples=num_samples, max_length=max_length, temperature=0.5)
    assert isinstance(generated_smiles_temp, list)

@pytest.mark.parametrize("num_batches, samples_per_batch, max_length", [(2, 5, 10), (3, 1, None)])
def test_generate_molecules_batch(polymer_diffusion_inference_instance, num_batches, samples_per_batch, max_length):
    instance = polymer_diffusion_inference_instance
    
    # Mock the generate_molecules method to control its output
    instance.generate_molecules = MagicMock(return_value=["mock_smiles"] * samples_per_batch)

    generated_smiles = instance.generate_molecules_batch(
        num_batches=num_batches,
        samples_per_batch=samples_per_batch,
        max_length=max_length,
        temperature=1.0,
        seed=42,
    )

    assert isinstance(generated_smiles, list)
    assert len(generated_smiles) == num_batches * samples_per_batch
    assert all(s == "mock_smiles" for s in generated_smiles)

    # Verify that generate_molecules was called the correct number of times with correct arguments
    assert instance.generate_molecules.call_count == num_batches
    for call_args, call_kwargs in instance.generate_molecules.call_args_list:
        assert call_kwargs["num_samples"] == samples_per_batch
        assert call_kwargs["max_length"] == max_length
        assert call_kwargs["temperature"] == 1.0
        assert "seed" in call_kwargs # Seed should be passed, even if it's None or incremented
