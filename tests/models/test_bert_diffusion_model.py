import pytest
import torch
from torch import nn
from transformers import BertConfig

from polydiff.model.bert_diffusion_model import BertDiffusionModel


@pytest.fixture(scope="module")
def model_config():
    """Provides a standard BERT configuration for testing."""
    return BertConfig(
        vocab_size=30,  # Small vocab for testing
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )


@pytest.fixture(scope="module")
def diffusion_model(model_config):
    """Initializes the BertDiffusionModel from the config."""
    return BertDiffusionModel(model_config)


class TestBertDiffusionModel:
    def test_initialization(self, diffusion_model, model_config):
        assert diffusion_model.config.vocab_size == model_config.vocab_size
        assert diffusion_model.config.hidden_size == model_config.hidden_size
        assert isinstance(diffusion_model, BertDiffusionModel)

    def test_forward_pass(self, diffusion_model, model_config):
        batch_size = 4
        seq_length = 64
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_length))

        # Test forward pass without timestep embeddings
        outputs = diffusion_model(input_ids=input_ids)
        assert outputs.last_hidden_state.shape == (
            batch_size,
            seq_length,
            model_config.hidden_size,
        )

    def test_forward_with_timestep_embeddings(self, diffusion_model, model_config):
        batch_size = 4
        seq_length = 64
        hidden_size = model_config.hidden_size
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_length))
        timestep_embeds = torch.randn(batch_size, hidden_size)

        # Get output without timestep embeddings
        base_output = diffusion_model(input_ids=input_ids).last_hidden_state

        # Get output with timestep embeddings
        conditioned_output = diffusion_model(
            input_ids=input_ids, timestep_embeds=timestep_embeds
        ).last_hidden_state

        assert conditioned_output.shape == (batch_size, seq_length, hidden_size)
        # Check that the timestep embeddings actually changed the output
        assert not torch.allclose(base_output, conditioned_output)

    def test_return_dict_behavior(self, diffusion_model):
        batch_size = 2
        seq_length = 16
        input_ids = torch.randint(
            0, diffusion_model.config.vocab_size, (batch_size, seq_length)
        )

        # Test with return_dict=True (default)
        outputs_dict = diffusion_model(input_ids=input_ids, return_dict=True)
        assert hasattr(outputs_dict, "last_hidden_state")

        # Test with return_dict=False
        outputs_tuple = diffusion_model(input_ids=input_ids, return_dict=False)
        assert isinstance(outputs_tuple, tuple)
        assert outputs_tuple[0].shape == (
            batch_size,
            seq_length,
            diffusion_model.config.hidden_size,
        )

    def test_get_embedding_layer(self, diffusion_model):
        embedding_layer = diffusion_model.get_embedding_layer()
        assert isinstance(embedding_layer, nn.Module)
        # Further checks can be added, e.g., type of embedding layer, its parameters
        assert hasattr(embedding_layer, 'word_embeddings')
        assert hasattr(embedding_layer, 'position_embeddings')
        assert hasattr(embedding_layer, 'token_type_embeddings')
