# Inference Module (`polydiff/inference`)

The `polydiff/inference` module provides the functionality to load trained diffusion models and generate new molecules (SMILES strings). It abstracts away the complexities of model loading, device management, and the reverse diffusion (denoising) process.

## Key Components

### `PolymerDiffusionInference`

`PolymerDiffusionInference` is the main class for performing inference. It loads a trained `DiffusionTask` checkpoint and provides a method to generate new SMILES strings.

-   **Purpose:** To offer a convenient and robust interface for generating molecules using a trained PolyDiff model.
-   **Initialization (`__init__`):**
    -   `checkpoint_path` (str): The file path to the trained `DiffusionTask` checkpoint (`.ckpt` file).
    -   `device` (str, default: "auto"): The computing device to use for inference (e.g., "cuda", "cpu", "mps"). "auto" attempts to select the best available device.
    -   `schedule_type` (Optional[str], optional): Allows overriding the diffusion schedule type specified in the checkpoint (e.g., "linear", "cosine").
    -   `diffusion_steps` (Optional[int], optional): Allows overriding the number of diffusion steps for generation.
-   **Internal Attributes:**
    -   `self.task`: The loaded `DiffusionTask` instance.
    -   `self.model`: The underlying BERT-based diffusion model.
    -   `self.tokenizer`: The tokenizer used by the model.
    -   `self.diffusion_steps`: The number of steps used for the reverse diffusion process.
    -   `self.schedule_type`: The type of diffusion schedule used.
    -   `self.schedule`: The initialized diffusion schedule object.
-   **Methods:**
    -   `_get_device(device)`: Internal helper to determine the appropriate `torch.device` based on the input string and system availability.
    -   `generate_molecules(num_samples=1, max_length=None, temperature=1.0, seed=None)`:
        -   **Purpose:** Generates a specified number of new SMILES strings by running the reverse diffusion process.
        -   **Args:**
            -   `num_samples` (int): The number of molecules to generate.
            -   `max_length` (Optional[int]): The maximum sequence length for generated molecules. If `None`, it defaults to the model's `max_position_embeddings`.
            -   `temperature` (float): Sampling temperature. Higher values lead to more random (diverse) outputs, while lower values lead to more conservative (less diverse) outputs.
            -   `seed` (Optional[int]): A random seed for reproducibility of the generation process.
        -   **Returns:** A `List[str]` containing the generated SMILES strings.

## Development Guide

### Customizing Inference Logic

-   **Sampling Strategy:** The `generate_molecules` method currently implements a basic sampling strategy (sampling tokens from the predicted distribution at each step). To implement more advanced sampling techniques (e.g., DDIM, ancestral sampling with different noise predictions), you would modify the loop within `generate_molecules`.
-   **Batching for Generation:** For generating a large number of molecules, ensure that the batching logic within `generate_molecules` is efficient and utilizes the available hardware (e.g., by processing multiple samples in parallel).

### Model Loading and Compatibility

-   **Checkpoint Compatibility:** If the underlying `DiffusionTask` or `BertDiffusionModel` architecture changes, ensure that `PolymerDiffusionInference`'s `load_from_checkpoint` logic remains compatible or is updated accordingly.
-   **Device Management:** The `_get_device` method handles device selection. If new hardware types or specific device configurations need to be supported, this method should be extended.

### Testing

-   **Unit Tests:** Refer to `tests/inference/test_inference.py` for examples of how to test the `PolymerDiffusionInference` class. It's crucial to mock external dependencies like `DiffusionTask` and `AutoTokenizer` to ensure isolated and fast tests.
-   **Mocking Strategies:** Pay attention to how `MagicMock` is used to simulate the behavior of `DiffusionTask`'s attributes (`model`, `hparams`, `tokenizer`, `diffusion_process`) and their return values. This is key for testing the inference logic without needing a fully trained model or network access.
-   **Coverage:** When making changes, run `pytest --cov=polydiff.inference --cov-report=term-missing` to ensure your changes are adequately covered by tests.
