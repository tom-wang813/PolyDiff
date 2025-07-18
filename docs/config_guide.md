# Configuration Management Guide

## Overview

PolyDiffusion now uses a hierarchical configuration system that provides better organization, reusability, and maintainability of experiment configurations.

## Configuration Structure

```
configs/
├── base.yaml                    # Base configuration with defaults
├── model_variants/              # Model architecture variants
│   ├── small.yaml              # Small model for quick testing
│   ├── medium.yaml             # Balanced performance model
│   └── large.yaml              # Full-size model
└── experiments/                 # Experiment-specific configurations
    ├── polymer_generation.yaml # Polymer SMILES generation
    ├── smiles_generation.yaml  # General SMILES generation
    └── quick_test.yaml         # Quick testing configuration
```

## Using Configurations

### New Hierarchical Configs

```bash
# Quick testing (equivalent to old model-1 config)
python scripts/train.py --config configs/experiments/quick_test.yaml

# Polymer generation experiment
python scripts/train.py --config configs/experiments/polymer_generation.yaml

# General SMILES generation
python scripts/train.py --config configs/experiments/smiles_generation.yaml
```

### Legacy Support

The system still supports legacy configurations:

```bash
# Legacy config (still works)
python scripts/train.py --config experiments/model-1/configs/model-1.yaml
```

## Configuration Inheritance

Experiment configs can inherit from base configs and model variants:

```yaml
# Example: configs/experiments/my_experiment.yaml
defaults:
  - base                    # Inherit from base.yaml
  - model_variants/medium   # Use medium model variant

# Override specific settings
data:
  data_path: "data/my_data.txt"
  batch_size: 16

trainer:
  max_epochs: 200
```

## Configuration Management Tools

### List Available Configurations

```bash
python scripts/config_tool.py list
```

### Preview Configuration

See the final merged configuration:

```bash
python scripts/config_tool.py preview configs/experiments/polymer_generation.yaml
```

### Validate Configuration

Check if a configuration is valid:

```bash
python scripts/config_tool.py validate configs/experiments/polymer_generation.yaml
```

### Compare Configurations

See differences between configurations:

```bash
python scripts/config_tool.py diff configs/experiments/polymer_generation.yaml configs/experiments/smiles_generation.yaml
```

## Inference and Evaluation

### Generate Molecules

```bash
# Basic generation
python scripts/inference.py --checkpoint experiments/model-1/checkpoints/best-epoch=00-val_loss=0.51.ckpt --num_samples 10

# Advanced generation with validation
python scripts/inference.py \\
    --checkpoint experiments/model-1/checkpoints/best-epoch=00-val_loss=0.51.ckpt \\
    --num_samples 100 \\
    --temperature 0.8 \\
    --output results.json \\
    --validate \\
    --diversity
```

### Evaluate Model Performance

```bash
python scripts/evaluate.py \\
    --checkpoint experiments/model-1/checkpoints/best-epoch=00-val_loss=0.51.ckpt \\
    --num_samples 1000 \\
    --reference_data data/smiles.txt \\
    --output evaluation_results.json
```

## Creating Custom Configurations

### 1. Model Variant

Create a new model variant in `configs/model_variants/`:

```yaml
# configs/model_variants/tiny.yaml
model:
  hidden_size: 128
  num_hidden_layers: 4
  num_attention_heads: 4
  intermediate_size: 512

data:
  batch_size: 64
```

### 2. Experiment Configuration

Create a new experiment that inherits from base and model variant:

```yaml
# configs/experiments/my_experiment.yaml
defaults:
  - base
  - model_variants/tiny

data:
  data_path: "data/my_dataset.txt"

checkpointing:
  dirpath: "experiments/my_experiment/checkpoints"

logging:
  save_dir: "experiments/my_experiment/logs"
  name: "my_experiment"
```

## Best Practices

1. **Use inheritance**: Always inherit from `base.yaml` and appropriate model variants
2. **Organize by purpose**: Group related experiments in the `experiments/` folder
3. **Validate configs**: Use the config tool to validate before training
4. **Document changes**: Add comments explaining configuration choices
5. **Version control**: Include all config files in git for reproducibility

## Migration from Legacy Configs

Legacy configs are still supported, but for new experiments:

1. Use the new hierarchical system
2. Start with `base.yaml` and a model variant
3. Override only what you need to change
4. Use the config tool to verify your configuration

## Troubleshooting

### Config not found error
```bash
# Check if file exists
python scripts/config_tool.py list

# Preview to see what's actually loaded
python scripts/config_tool.py preview your_config.yaml
```

### Inheritance not working
```bash
# Validate the config
python scripts/config_tool.py validate your_config.yaml

# Check that default configs exist
ls configs/base.yaml
ls configs/model_variants/
```

### Unexpected values
```bash
# Compare with base config
python scripts/config_tool.py diff configs/base.yaml your_config.yaml
```
