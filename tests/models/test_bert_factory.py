import pytest
from unittest.mock import MagicMock, patch
from polydiff.model.bert_factory import build_medium_bert_diffusion_model
from transformers import AutoTokenizer, BertConfig

@patch('polydiff.model.bert_factory.AutoTokenizer')
@patch('polydiff.model.bert_factory.BertDiffusionModel')
def test_build_medium_bert_diffusion_model(MockBertDiffusionModel, MockAutoTokenizer):
    # Mock tokenizer instance
    mock_tokenizer_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_tokenizer_instance.__len__.return_value = 1000  # Simulate vocab_size

    # Call the function under test
    model, tokenizer = build_medium_bert_diffusion_model()

    # Assertions for tokenizer
    MockAutoTokenizer.from_pretrained.assert_called_once_with(
        "seyonec/ChemBERTa-zinc-base-v1", use_fast=True
    )
    assert tokenizer == mock_tokenizer_instance

    # Assertions for model
    MockBertDiffusionModel.assert_called_once()
    args, kwargs = MockBertDiffusionModel.call_args
    assert kwargs["vocab_size"] == 1000
    assert kwargs["hidden_size"] == 512
    assert kwargs["num_hidden_layers"] == 6
    assert kwargs["num_attention_heads"] == 8
    assert kwargs["intermediate_size"] == 2048
    assert kwargs["max_position_embeddings"] == 512
    assert kwargs["diffusion_steps"] == 100
    assert model == MockBertDiffusionModel.return_value

@patch('polydiff.model.bert_factory.AutoTokenizer')
@patch('polydiff.model.bert_factory.BertDiffusionModel')
def test_build_medium_bert_diffusion_model_with_custom_args(MockBertDiffusionModel, MockAutoTokenizer):
    # Mock tokenizer instance
    mock_tokenizer_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_tokenizer_instance.__len__.return_value = 2000  # Simulate custom vocab_size

    # Call the function with custom arguments
    model, tokenizer = build_medium_bert_diffusion_model(tokenizer=mock_tokenizer_instance, diffusion_steps=500)

    # Assertions for tokenizer
    MockAutoTokenizer.from_pretrained.assert_not_called() # Should not be called if tokenizer is provided
    assert tokenizer == mock_tokenizer_instance

    # Assertions for model
    MockBertDiffusionModel.assert_called_once()
    args, kwargs = MockBertDiffusionModel.call_args
    assert kwargs["vocab_size"] == 2000
    assert kwargs["hidden_size"] == 512
    assert kwargs["num_hidden_layers"] == 6
    assert kwargs["num_attention_heads"] == 8
    assert kwargs["intermediate_size"] == 2048
    assert kwargs["max_position_embeddings"] == 512
    assert kwargs["diffusion_steps"] == 500
    assert model == MockBertDiffusionModel.return_value
