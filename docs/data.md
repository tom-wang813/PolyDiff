# Data Module (`polydiff/data`)

The `polydiff/data` module is responsible for handling all data-related operations within the PolyDiff project. This includes loading, preprocessing, splitting, and validating SMILES string datasets, as well as providing PyTorch `DataLoader` instances for training and validation.

## Key Components

### `SmilesDataset`

`SmilesDataset` is a standard PyTorch `Dataset` that tokenizes a list of SMILES strings. It integrates with Hugging Face `AutoTokenizer` to convert raw SMILES into numerical input IDs and attention masks suitable for BERT-based models.

-   **Purpose:** To provide an efficient way to load and tokenize SMILES data for model consumption.
-   **Initialization (`__init__`):**
    -   `smiles_list` (List[str]): A list of raw SMILES strings.
    -   `tokenizer`: An initialized Hugging Face tokenizer (e.g., `AutoTokenizer` instance).
    -   `max_length` (int): The maximum sequence length for tokenization. Sequences longer than this will be truncated, and shorter ones will be padded.
-   **Methods:**
    -   `__len__()`: Returns the total number of SMILES strings in the dataset.
    -   `__getitem__(idx)`: Retrieves the tokenized encoding for a SMILES string at the given index. Returns a dictionary with `input_ids`, `attention_mask`, and the original `smiles` string.

### `SmilesDataModule`

`SmilesDataModule` is a PyTorch Lightning `LightningDataModule` that encapsulates the full data lifecycle. It handles data loading from a specified path, splitting into training and validation sets, and creating `DataLoader` instances.

-   **Purpose:** To abstract away data preparation complexities, making the training loop cleaner and more reproducible, especially in distributed training environments.
-   **Initialization (`__init__`):**
    -   `data_path` (str): Path to the text file containing SMILES strings (one per line).
    -   `tokenizer_name` (str): Name of the pre-trained tokenizer to use (e.g., `"seyonec/ChemBERTa-zinc-base-v1"`).
    -   `train_split` (float, default: 0.8): Proportion of data to allocate for the training set.
    -   `batch_size` (int, default: 32): Batch size for the `DataLoader`s.
    -   `num_workers` (int, default: 4): Number of subprocesses to use for data loading.
    -   `max_length` (int, default: 512): Maximum sequence length for tokenization.
    -   `val_subset_ratio` (float, default: 1.0): Ratio of the validation set to use as a subset. Useful for quick validation runs.
-   **Methods:**
    -   `prepare_data()`: Downloads and caches the tokenizer if not already present. This is called only once per node in distributed training.
    -   `setup(stage)`: Loads the SMILES data from `data_path`, performs validation using RDKit (skipping invalid SMILES), splits the data into training and validation sets, and initializes `SmilesDataset` instances. This is called on every GPU in a distributed setup.
    -   `train_dataloader()`: Returns a `DataLoader` for the training set.
    -   `val_dataloader()`: Returns a `DataLoader` for the validation set.

## Development Guide

### Extending Data Loading

-   **New Dataset Formats:** If you need to support new data formats (e.g., CSV, JSON), you would modify the `setup` method in `SmilesDataModule` to parse the new format into a `smiles_list`.
-   **Custom Tokenization:** If a different tokenization strategy is required, you might need to create a new `Dataset` class or modify `SmilesDataset` to accept a custom tokenization function instead of a direct `tokenizer` object.

### Enhancing Data Validation

-   **Additional Validation Rules:** You can add more validation rules within the `setup` method of `SmilesDataModule`. For example, checking for specific atom types, molecular weight ranges, or other chemical properties.
-   **Error Handling:** Improve error logging or introduce custom error types for different data validation failures.

### Distributed Training Considerations

-   `prepare_data` is designed for single-process operations (e.g., downloading). `setup` is for operations that need to be performed on each process (e.g., loading data into memory).
-   Ensure that any data transformations or augmentations are handled consistently across all workers if `num_workers > 0`.

## Example Usage

Refer to `scripts/train.py` for an example of how `SmilesDataModule` is initialized and used within a PyTorch Lightning training pipeline.
