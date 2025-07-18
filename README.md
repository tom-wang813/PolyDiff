# 🧬 PolyDiffusion

<div align="center">

**Advanced Molecular Generation using BERT-based Discrete Diffusion Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[🚀 Quick Start](#-quick-start) •
[📖 Documentation](#-documentation) •
[🎯 Features](#-features) •
[🔬 Examples](#-examples) •
[🤝 Contributing](#-contributing)

</div>

---

## 🌟 Overview

PolyDiffusion is a cutting-edge molecular generation framework that combines the power of **BERT transformers** with **discrete diffusion models** to generate novel polymers and small molecules. By leveraging state-of-the-art NLP techniques adapted for chemical space exploration, PolyDiffusion enables researchers to discover new molecular structures with desired properties.

### 🎯 Why PolyDiffusion?

| **Challenge** | **Our Solution** |
|---------------|------------------|
| 🧪 **Chemical Diversity** | Generate diverse, valid molecular structures using advanced diffusion processes |
| 🔧 **Flexibility** | Hierarchical configuration system for easy experimentation |
| ⚡ **Performance** | GPU-accelerated training with PyTorch Lightning |
| 📊 **Evaluation** | Comprehensive metrics including validity, uniqueness, and novelty |
| 🔬 **Research-Ready** | Built-in experiment tracking and reproducible configurations |

### 🏗️ Architecture

```
 SMILES Input     BERT Encoder      Diffusion Process     Denoising      Generated Molecules
     │                 │                   │                 │                    │
┌────▼────┐      ┌─────▼─────┐      ┌──────▼──────┐   ┌─────▼─────┐        ┌─────▼─────┐
│ "CC(=O)O"│ ──→  │ [CLS] C C │ ──→  │ Add Noise   │──→│ Learned   │   ──→  │ Novel     │
│  ...     │      │ [SEP] ... │      │ t=0→T       │   │ Denoising │        │ SMILES    │
└─────────┘      └───────────┘      └─────────────┘   └───────────┘        └───────────┘
```

---

## 🚀 Quick Start

### 📦 Installation

Choose one of the following installation methods:

<details>
<summary><b>🐍 Option 1: Conda (Recommended)</b></summary>

```bash
# Clone the repository
git clone https://github.com/tom-wang813/PolyDiff.git
cd PolyDiff

# Create environment from file
conda env create -f environment_minimal.yml
conda activate polydiff

# Install the package
pip install -e .
```

</details>

<details>
<summary><b>🔧 Option 2: pip</b></summary>

```bash
# Clone the repository
git clone https://github.com/tom-wang813/PolyDiff.git
cd PolyDiff

# Create virtual environment
python -m venv polydiff_env
source polydiff_env/bin/activate  # On Windows: polydiff_env\Scripts\activate

# Install dependencies
pip install -e .
pip install torch pytorch-lightning transformers omegaconf rdkit
```

</details>

### ⚡ 30-Second Demo

```bash
# 1. Quick test training
python scripts/train.py --config configs/experiments/quick_test.yaml

# 2. Generate molecules
python scripts/inference.py \
    --checkpoint experiments/quick_test/checkpoints/test-epoch=00-val_loss=1.63.ckpt \
    --num_samples 10 \
    --validate

# 3. Evaluate performance
python scripts/evaluate.py \
    --checkpoint experiments/quick_test/checkpoints/test-epoch=00-val_loss=1.63.ckpt \
    --num_samples 100
```

---

## 🎯 Features

### 🧠 **Advanced ML Architecture**
- **🤖 BERT-based Encoding**: Leverages pre-trained ChemBERTa for molecular understanding
- **🌊 Discrete Diffusion**: State-of-the-art diffusion processes for molecular generation
- **⚡ GPU Acceleration**: PyTorch Lightning for efficient multi-GPU training
- **🎛️ Flexible Schedules**: Cosine, linear, and custom noise schedules

### 🔧 **Developer-Friendly**
- **📁 Hierarchical Configs**: Organized, reusable configuration system
- **🛠️ Config Management**: Built-in tools for config validation and comparison
- **📊 Experiment Tracking**: MLflow integration for reproducible research
- **🧪 Testing Suite**: Comprehensive test coverage with pytest

### 📈 **Comprehensive Evaluation**
- **✅ Validity**: RDKit-based molecular validation
- **🔄 Diversity**: Tanimoto similarity-based diversity metrics
- **🆕 Novelty**: Comparison against training datasets
- **⚖️ Properties**: Molecular weight, LogP, TPSA analysis

### 🎨 **User Experience**
- **🖥️ Rich CLI**: Beautiful command-line interface with progress bars
- **📝 Detailed Logging**: Comprehensive logging with TensorBoard
- **🔍 Debugging Tools**: Config preview, validation, and diff tools
- **📚 Documentation**: Extensive guides and examples

---

## 🔬 Examples

### 🧪 Training Custom Models

<details>
<summary><b>Polymer Generation</b></summary>

```bash
# Train a model specifically for polymer generation
python scripts/train.py --config configs/experiments/polymer_generation.yaml

# Generated config automatically includes:
# - Medium-sized BERT model
# - Polymer-specific data loading
# - Optimized hyperparameters
# - Automatic checkpointing
```

</details>

<details>
<summary><b>Quick Experimentation</b></summary>

```bash
# Fast iteration for development
python scripts/train.py --config configs/experiments/quick_test.yaml

# Features:
# - Small model (fast training)
# - Minimal epochs
# - Quick validation
```

</details>

### 🎯 Molecular Generation

<details>
<summary><b>Basic Generation</b></summary>

```bash
python scripts/inference.py \
    --checkpoint path/to/model.ckpt \
    --num_samples 50 \
    --temperature 0.8
```

</details>

<details>
<summary><b>Advanced Generation with Analysis</b></summary>

```bash
python scripts/inference.py \
    --checkpoint path/to/model.ckpt \
    --num_samples 1000 \
    --temperature 0.7 \
    --validate \
    --diversity \
    --output results.json
```

**Output includes:**
- Generated SMILES strings
- Validity statistics
- Diversity metrics
- Molecular property analysis

</details>

### 📊 Model Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint experiments/model-1/checkpoints/best-epoch=00-val_loss=0.51.ckpt \
    --num_samples 100 \
    --reference_data data/sample_smiles.txt \
    --output evaluation_report.json
```

**Evaluation Metrics:**
- **Validity Rate**: Percentage of chemically valid molecules
- **Uniqueness**: Ratio of unique molecules generated
- **Diversity**: Average Tanimoto distance between molecules
- **Novelty**: Percentage of molecules not in training set
- **Property Distribution**: Statistical analysis of molecular properties

---

## 📖 Documentation

| **Guide** | **Description** |
|-----------|-----------------|
| [📋 Configuration Guide](docs/config_guide.md) | Complete guide to the hierarchical config system |
| [🗂️ Scripts Documentation](scripts/README.md) | Detailed script usage and examples |
| [📊 Data Documentation](docs/data.md) | Data formats and preprocessing |
| [🧠 Model Documentation](docs/model.md) | Model architecture and customization |
| [🔬 Inference Documentation](docs/inference.md) | Generation and evaluation guides |

---

## 🏗️ Project Structure

```
polydiffusion/
├── 📁 configs/                  # 🆕 Hierarchical configuration system
│   ├── 📄 base.yaml            # Base configuration template
│   ├── 📁 model_variants/      # Model size variants
│   │   ├── 📄 small.yaml      # Small model (quick testing)
│   │   ├── 📄 medium.yaml     # Balanced model
│   │   └── 📄 large.yaml      # Full-size model
│   └── 📁 experiments/         # Experiment configurations
│       ├── 📄 polymer_generation.yaml
│       ├── 📄 smiles_generation.yaml
│       └── 📄 quick_test.yaml
├── 📁 scripts/                 # Executable scripts
│   ├── 🐍 train.py            # Training script
│   ├── 🐍 inference.py        # 🆕 Inference script
│   ├── 🐍 evaluate.py         # 🆕 Evaluation script
│   └── 🐍 config_tool.py      # 🆕 Configuration management
├── 📁 polydiff/               # Core source code
│   ├── 📁 data/               # Data processing modules
│   ├── 📁 diffusion/          # Diffusion model implementation
│   ├── 📁 inference/          # Inference utilities
│   ├── 📁 model/              # BERT model components
│   └── 📁 tasks/              # PyTorch Lightning tasks
├── 📁 experiments/            # Training outputs
├── 📁 data/                   # Training datasets
├── 📁 docs/                   # Documentation
└── 📁 tests/                  # Test suite
```

---

## 🤝 Contributing

We welcome contributions! This project uses **research-friendly CI/CD** that focuses on functionality over strict code formatting.

### 🔄 **CI/CD Philosophy**
- ✅ **Functionality First**: Tests must pass, but code style warnings won't break builds
- 🎯 **Research-Focused**: MyPy, Black, and isort run as advisory checks only
- 🚀 **Developer-Friendly**: Push without worrying about perfect formatting
- 📊 **Coverage Optional**: Test coverage reporting available but not mandatory

### 🧪 Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run checks locally (all optional)
mypy --config-file mypy.ini .     # Type checking (relaxed)
black . --check                   # Code formatting (advisory)
isort . --check-only              # Import sorting (advisory)
pytest --cov=polydiff            # Tests (required)
```
---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ChemBERTa**: Pre-trained molecular representations
- **PyTorch Lightning**: Efficient deep learning framework
- **RDKit**: Chemical informatics toolkit
- **Hugging Face**: Transformer implementations

---
