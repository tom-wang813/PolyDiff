# Scripts Documentation

This directory contains all the executable scripts for training, inference, evaluation, and configuration management.

## Quick Reference

### Training
```bash
# Quick test (equivalent to old model-1 config)
python scripts/train.py --config configs/experiments/quick_test.yaml

# Polymer generation experiment
python scripts/train.py --config configs/experiments/polymer_generation.yaml

# General SMILES generation
python scripts/train.py --config configs/experiments/smiles_generation.yaml
```

### Inference
```bash
# Generate molecules
python scripts/inference.py \
    --checkpoint experiments/model-1/checkpoints/best-epoch=00-val_loss=0.51.ckpt \
    --num_samples 10 \
    --output results.json \
    --validate

# Evaluate model performance
python scripts/evaluate.py \
    --checkpoint experiments/model-1/checkpoints/best-epoch=00-val_loss=0.51.ckpt \
    --num_samples 100 \
    --reference_data data/sample_smiles.txt
```

### Configuration Management
```bash
# List all configurations
python scripts/config_tool.py list

# Preview a configuration
python scripts/config_tool.py preview configs/experiments/polymer_generation.yaml

# Validate configuration
python scripts/config_tool.py validate configs/experiments/quick_test.yaml

# Compare configurations
python scripts/config_tool.py diff config1.yaml config2.yaml
```

## Project Structure

```
polydiffusion/
├── configs/                     # 🆕 Hierarchical configuration system
│   ├── base.yaml               # Base configuration
│   ├── model_variants/         # Model architecture variants
│   │   ├── small.yaml         # Small model for testing
│   │   ├── medium.yaml        # Balanced model
│   │   └── large.yaml         # Full-size model
│   └── experiments/            # Experiment configurations
│       ├── polymer_generation.yaml
│       ├── smiles_generation.yaml
│       └── quick_test.yaml
├── scripts/
│   ├── train.py               # Training script (updated)
│   ├── inference.py           # 🆕 Inference script
│   ├── evaluate.py            # 🆕 Evaluation script
│   └── config_tool.py         # 🆕 Configuration management
├── polydiff/                  # Source code
│   ├── data/
│   ├── diffusion/
│   ├── inference/
│   ├── model/
│   └── tasks/
├── experiments/               # Experiment outputs
│   └── model-1/
│       ├── checkpoints/
│       └── logs/
├── data/                      # Training data
├── docs/                      # Documentation
│   └── config_guide.md        # 🆕 Configuration guide
└── tests/                     # Test files
```

## Features

### 🎯 **Enhanced Configuration System**
- **Hierarchical configs**: Inherit from base configs and model variants
- **Better organization**: Separate configs by purpose (base, variants, experiments)
- **Configuration tools**: List, preview, validate, and compare configs
- **Legacy support**: Existing configs still work

### 🚀 **Dedicated Inference Pipeline**
- **Flexible generation**: Control temperature, seed, batch size
- **Built-in validation**: SMILES validity check with RDKit
- **Diversity metrics**: Calculate molecular diversity
- **Multiple output formats**: Console display or JSON files

### 📊 **Comprehensive Evaluation**
- **Standard metrics**: Validity, uniqueness, diversity, novelty
- **Molecular properties**: MW, LogP, TPSA, ring count, etc.
- **Performance tracking**: Generation speed and efficiency
- **Reference comparison**: Compare against training data

### 🔧 **Improved Developer Experience**
- **Better CLI**: Clear help messages and error handling
- **Validation**: Catch configuration errors early
- **Documentation**: Comprehensive guides and examples
- **Debugging tools**: Config preview and comparison

## Documentation

- [Configuration Guide](docs/config_guide.md) - Complete guide to the new config system
- [Data Documentation](docs/data.md)
- [Model Documentation](docs/model.md)
- [Tasks Documentation](docs/tasks.md)
- [Diffusion Documentation](docs/diffusion.md)
- [Inference Documentation](docs/inference.md)

## Installation

```bash
# Install in development mode
pip install -e .

# Or install dependencies manually
pip install torch pytorch-lightning transformers omegaconf rdkit
```

## Legacy Support

The new system maintains full backward compatibility. All existing scripts and configurations continue to work:

```bash
# Legacy training (still works)
python scripts/train.py --config experiments/model-1/configs/model-1.yaml
``` By leveraging the power of BERT for molecular representation and a diffusion process for generation, PolyDiff aims to provide a robust and flexible platform for exploring chemical space.

## Key Features

-   **Discrete Diffusion Models:** Core implementation of discrete diffusion processes tailored for molecular generation.
-   **BERT Architecture Integration:** Utilizes BERT for powerful sequence representation and generation capabilities.
-   **Flexible Data Processing:** A robust data pipeline for handling SMILES strings, including validation and splitting.
-   **Comprehensive Evaluation Metrics:** Tools and utilities for evaluating the performance of generated molecules.
-   **Automated Code Quality:** Enforces high code standards with `mypy` (type checking), `Black` (code formatting), and `isort` (import sorting).
-   **Extensive Testing:** Comprehensive unit and integration tests ensure the reliability and correctness of the codebase.
-   **Structured Configuration Management:** Utilizes `OmegaConf` for flexible and reproducible experiment configuration.
-   **Experiment Tracking & Model Versioning:** Integrates `MLflow` for tracking experiments, logging metrics, parameters, and managing model versions.
-   **Continuous Integration (CI):** Automated checks via GitHub Actions ensure code quality and test coverage on every push and pull request.

## Installation

To set up the development environment, we highly recommend using Conda to manage your virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/polydiff.git
    cd polydiff
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create -n polydiff python=3.10
    conda activate polydiff
    ```

3.  **Install the project in editable mode:**
    ```bash
    pip install -e .
    ```

4.  **Install development and experiment tracking dependencies:**
    ```bash
    pip install mypy black isort pytest pytest-cov omegaconf mlflow
    ```

## Development Workflow

### Code Quality Checks

We maintain high code quality standards using automated tools. It's recommended to run these checks before committing your changes.

-   **Run MyPy (type checking):** Ensures type consistency throughout the codebase.
    ```bash
    mypy --config-file mypy.ini .
    ```

-   **Run Black (code formatting - check only):** Verifies adherence to code style guidelines.
    ```bash
    black . --check --exclude "(.git|__pycache__|.pytest_cache|experiments|polydiff.egg-info|notebook)"
    ```

-   **Run isort (import sorting - check only):** Checks if imports are correctly sorted and grouped.
    ```bash
    isort . --check-only --skip-glob "__pycache__,.pytest_cache,experiments,polydiff.egg-info,notebook"
    ```

-   **To automatically format code and sort imports (use with caution, ideally before committing):**
    ```bash
    black . --exclude "(.git|__pycache__|.pytest_cache|experiments|polydiff.egg-info|notebook)"
    isort . --skip-glob "__pycache__,.pytest_cache,experiments,polydiff.egg-info|notebook"
    ```

### Running Tests

We use `pytest` for running tests and `pytest-cov` for measuring code coverage. Aim for high test coverage to ensure code reliability.

-   **Run all tests with coverage report:**
    ```bash
    pytest --cov=polydiff --cov-report=term-missing
    ```

-   **Run specific tests (e.g., for data module):**
    ```bash
    pytest tests/data/test_datamodule.py
    ```

### Configuration Management with OmegaConf

Project configurations are managed using `OmegaConf`, allowing for flexible and hierarchical settings. Configuration files are located in the `experiments/` directory.

-   **Loading Configuration:** Configurations are loaded from YAML files (e.g., `experiments/model-1/configs/model-1.yaml`).
-   **Accessing Parameters:** Parameters can be accessed using dot notation (e.g., `config.data.batch_size`).
-   **Overriding Parameters:** `OmegaConf` supports easy overriding of parameters via command-line arguments or other configuration sources, facilitating hyperparameter tuning and experimentation.

### Experiment Tracking and Model Versioning with MLflow

`MLflow` is integrated to track machine learning experiments, log parameters, metrics, and artifacts (like trained models). This enables reproducibility and efficient model management.

-   **Starting an MLflow Run:** Training scripts automatically initiate an MLflow run, logging all relevant details.
-   **Viewing Runs:** To view your MLflow runs, navigate to the project root and run:
    ```bash
    mlflow ui
    ```
    Then open your web browser to `http://localhost:5000` (or the address indicated by MLflow).
-   **Model Logging:** Trained models are automatically logged to MLflow, allowing for versioning and easy retrieval for inference or further analysis.

## Usage

### Training a Model

To train a new model, use the `train.py` script with a configuration file. Ensure your Conda environment is activated.

```bash
conda activate polydiff
python scripts/train.py --config experiments/model-1/configs/model-1.yaml
```

Adjust the configuration file (`experiments/model-1/configs/model-1.yaml`) to suit your needs, including data paths, model parameters, and training settings. You can also override parameters directly from the command line (e.g., `python scripts/train.py --config experiments/model-1/configs/model-1.yaml trainer.max_epochs=5`).

### Generating Molecules (Inference)

After training, you can use the `simple_inference.py` or `demo_inference.py` scripts for molecule generation. These scripts demonstrate how to load a trained model from a checkpoint and generate new SMILES strings.

```bash
conda activate polydiff
python simple_inference.py
# or
python demo_inference.py
```

Refer to these scripts for examples on how to load a trained model and generate new SMILES strings.

## Continuous Integration

This project uses GitHub Actions for Continuous Integration. Whenever code is pushed to the `main` branch or a pull request is opened, the following checks are automatically performed:

-   Python environment setup
-   Dependency installation
-   MyPy type checking
-   Black code formatting check
-   isort import sorting check
-   Unit and integration tests with code coverage reporting (uploaded to Codecov)

This ensures that all contributions adhere to the project's code quality standards.

## Contributing

We welcome contributions! Please ensure your code adheres to the established code quality standards and includes appropriate tests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details. (Note: A LICENSE file is assumed to exist. If not, please create one.)
