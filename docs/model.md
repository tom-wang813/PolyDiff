# Model Module (`polydiff/model`)

The `polydiff/model` module defines the neural network architectures used in the PolyDiff project, primarily focusing on the BERT-based discrete diffusion model. It leverages the Hugging Face `transformers` library for its core BERT implementation.

## Key Components

### `BertDiffusionModel`

`BertDiffusionModel` is a custom BERT-based model adapted for discrete diffusion tasks. It inherits from Hugging Face's `BertPreTrainedModel`, allowing it to utilize pre-trained BERT weights and configurations while extending its functionality for diffusion.

-   **Purpose:** To serve as the denoiser in the diffusion process, predicting the original (un-noised) input from a noised input at a given timestep.
-   **Initialization (`__init__`):**
    -   `config` (BertConfig): A configuration object (from `transformers`) that defines the BERT model's architecture (e.g., `vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `max_position_embeddings`).
-   **Methods:**
    -   `forward(...)`: The forward pass of the model. It takes standard BERT inputs (`input_ids`, `attention_mask`, etc.) and an additional `timestep_embeds`. The `timestep_embeds` are added to the BERT's sequence output, allowing the model to be conditioned on the diffusion timestep. It returns a `BaseModelOutputWithPoolingAndCrossAttentions` object or a tuple of tensors, depending on the `return_dict` argument.

### `build_medium_bert_diffusion_model` (Factory Function)

This is a utility function that provides a quick way to construct a medium-sized `BertDiffusionModel` with predefined architectural parameters. It also handles the initialization of the tokenizer.

-   **Purpose:** To simplify the creation of a standard model instance for common use cases, ensuring consistent model configurations.
-   **Parameters:**
    -   `tokenizer` (Optional): An existing tokenizer instance. If `None`, it will load `"seyonec/ChemBERTa-zinc-base-v1"`.
    -   `diffusion_steps` (int, default: 100): The number of diffusion steps, passed to the model's configuration.
-   **Model Configuration (Hardcoded in function):**
    -   `hidden_size`: 512
    -   `num_hidden_layers`: 6
    -   `num_attention_heads`: 8
    -   `intermediate_size`: 2048
    -   `max_position_embeddings`: 512
-   **Returns:** A tuple containing the initialized `BertDiffusionModel` instance and the tokenizer.

## Development Guide

### Customizing Model Architecture

-   **`BertDiffusionModel`:** To modify the model's architecture (e.g., add new layers, change attention mechanisms), you would directly edit the `BertDiffusionModel` class. Ensure that any new layers are compatible with the existing BERT structure and that the `forward` pass correctly integrates them.
-   **New Model Sizes/Configurations:** To create different sizes of BERT diffusion models, you can either:
    -   Modify `build_medium_bert_diffusion_model` to accept more parameters or create new `build_` functions (e.g., `build_large_bert_diffusion_model`).
    -   Directly instantiate `BertDiffusionModel` with a custom `BertConfig` object.

### Integrating Other Pre-trained Models

-   If you wish to use a different pre-trained model (e.g., RoBERTa, GPT-2) as the backbone, you would need to create a new model class inheriting from the respective `PreTrainedModel` (e.g., `RobertaPreTrainedModel`) and adapt the diffusion logic accordingly.

### Timestep Embeddings

-   The current implementation adds `timestep_embeds` directly to the `sequence_output`. Alternative methods for incorporating timestep information (e.g., concatenating, using FiLM layers) could be explored by modifying the `forward` method.

## Testing

-   **Unit Tests:** Refer to `tests/models/test_bert_diffusion_model.py` for testing the `BertDiffusionModel`'s forward pass and basic functionality. For `build_medium_bert_diffusion_model`, see `tests/models/test_bert_factory.py` which demonstrates how to mock `AutoTokenizer` and `BertDiffusionModel` to verify correct instantiation and parameter passing.
-   **Coverage:** When making changes, run `pytest --cov=polydiff.model --cov-report=term-missing` to ensure your changes are adequately covered by tests.
