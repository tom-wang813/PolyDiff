#!/usr/bin/env python3
"""
Configuration Management Tool for PolyDiffusion
==============================================

A utility script to help manage, validate, and preview configurations.

Usage:
    # List all available configurations
    python scripts/config_tool.py list
    
    # Preview a configuration (show merged result)
    python scripts/config_tool.py preview configs/experiments/polymer_generation.yaml
    
    # Validate a configuration
    python scripts/config_tool.py validate configs/experiments/polymer_generation.yaml
    
    # Show configuration differences
    python scripts/config_tool.py diff configs/experiments/polymer_generation.yaml configs/experiments/smiles_generation.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import List, Union

from omegaconf import OmegaConf, DictConfig, ListConfig
import yaml


def list_configs(configs_dir: Path = Path("configs")) -> None:
    """List all available configuration files."""
    print("Available Configurations:")
    print("=" * 50)
    
    # Convert to absolute path to avoid issues
    configs_dir = configs_dir.resolve()
    
    # Base configs
    base_files = list(configs_dir.glob("*.yaml"))
    if base_files:
        print("\\nBase Configurations:")
        for config_file in sorted(base_files):
            rel_path = config_file.relative_to(Path.cwd())
            print(f"  📄 {rel_path}")
    
    # Model variants
    variants_dir = configs_dir / "model_variants"
    if variants_dir.exists():
        variant_files = list(variants_dir.glob("*.yaml"))
        if variant_files:
            print("\\nModel Variants:")
            for config_file in sorted(variant_files):
                rel_path = config_file.relative_to(Path.cwd())
                print(f"  🔧 {rel_path}")
    
    # Experiments
    experiments_dir = configs_dir / "experiments"
    if experiments_dir.exists():
        experiment_files = list(experiments_dir.glob("*.yaml"))
        if experiment_files:
            print("\\nExperiment Configurations:")
            for config_file in sorted(experiment_files):
                rel_path = config_file.relative_to(Path.cwd())
                print(f"  🧪 {rel_path}")
    
    # Legacy configs
    legacy_configs = []
    experiments_path = Path("experiments")
    if experiments_path.exists():
        for legacy_config in experiments_path.glob("*/configs/*.yaml"):
            legacy_configs.append(legacy_config.resolve())
    
    if legacy_configs:
        print("\\nLegacy Configurations:")
        for config_file in sorted(legacy_configs):
            try:
                rel_path = config_file.relative_to(Path.cwd())
                print(f"  📋 {rel_path}")
            except ValueError:
                # Fallback to just the filename if relative path fails
                print(f"  📋 {config_file}")


def load_hierarchical_config(config_path: Path) -> Union[DictConfig, ListConfig]:
    """Load a hierarchical configuration with defaults resolution."""
    config = OmegaConf.load(config_path)
    
    # Handle defaults (inheritance)
    if "defaults" in config:
        base_configs = []
        for default in config.defaults:
            if "/" in default:
                # Model variant or nested config
                default_path = config_path.parent.parent / f"{default}.yaml"
            else:
                # Base config
                default_path = config_path.parent.parent / f"{default}.yaml"
            
            if default_path.exists():
                base_config = OmegaConf.load(default_path)
                base_configs.append(base_config)
            else:
                print(f"Warning: Default config not found: {default_path}")
        
        # Merge configs (later configs override earlier ones)
        if base_configs:
            merged_config = base_configs[0]
            for base_config in base_configs[1:]:
                merged_config = OmegaConf.merge(merged_config, base_config)
            
            # Current config overrides all defaults
            config = OmegaConf.merge(merged_config, config)
            
            # Remove defaults key from final config
            if "defaults" in config and isinstance(config, DictConfig):
                del config["defaults"]
    
    return config


def preview_config(config_path: str) -> None:
    """Preview the final merged configuration."""
    config_path_obj = Path(config_path)
    
    if not config_path_obj.exists():
        print(f"Error: Configuration file not found: {config_path_obj}")
        return
    
    print(f"Configuration Preview: {config_path_obj}")
    print("=" * 50)
    
    try:
        if "configs/" in str(config_path_obj):
            # New hierarchical config
            config = load_hierarchical_config(config_path_obj)
        else:
            # Legacy config
            config = OmegaConf.load(config_path_obj)
        
        # Convert to YAML for pretty printing
        yaml_str = OmegaConf.to_yaml(config)
        print(yaml_str)
        
    except Exception as e:
        print(f"Error loading configuration: {e}")


def validate_config(config_path: str) -> bool:
    """Validate a configuration file."""
    config_path_obj = Path(config_path)
    
    print(f"Validating: {config_path_obj}")
    print("=" * 30)
    
    if not config_path_obj.exists():
        print("❌ File not found")
        return False
    
    try:
        if "configs/" in str(config_path_obj):
            config = load_hierarchical_config(config_path_obj)
        else:
            config = OmegaConf.load(config_path_obj)
        
        # Basic validation checks
        required_sections = ["model", "data", "trainer"]
        missing_sections = []
        
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing required sections: {missing_sections}")
            return False
        
        # Check specific required fields
        validation_errors = []
        
        # Data validation
        if "data_path" not in config.data:
            validation_errors.append("data.data_path is required")
        
        # Model validation
        required_model_fields = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
        for field in required_model_fields:
            if field not in config.model:
                validation_errors.append(f"model.{field} is required")
        
        if validation_errors:
            print("❌ Validation errors:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
        
        print("✅ Configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False


def diff_configs(config1_path: str, config2_path: str) -> None:
    """Show differences between two configurations."""
    try:
        path1 = Path(config1_path)
        path2 = Path(config2_path)
        
        if "configs/" in str(path1):
            config1 = load_hierarchical_config(path1)
        else:
            config1 = OmegaConf.load(path1)
            
        if "configs/" in str(path2):
            config2 = load_hierarchical_config(path2)
        else:
            config2 = OmegaConf.load(path2)
        
        print(f"Configuration Differences:")
        print(f"  {config1_path}")
        print(f"  vs")
        print(f"  {config2_path}")
        print("=" * 50)
        
        # Convert to dicts for comparison
        dict1 = OmegaConf.to_container(config1, resolve=True)
        dict2 = OmegaConf.to_container(config2, resolve=True)
        
        def compare_dicts(d1, d2, path=""):
            differences = []
            all_keys = set(d1.keys()) | set(d2.keys())
            
            for key in sorted(all_keys):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences.append(f"+ {current_path}: {d2[key]} (only in config2)")
                elif key not in d2:
                    differences.append(f"- {current_path}: {d1[key]} (only in config1)")
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    differences.extend(compare_dicts(d1[key], d2[key], current_path))
                elif d1[key] != d2[key]:
                    differences.append(f"~ {current_path}: {d1[key]} -> {d2[key]}")
            
            return differences
        
        differences = compare_dicts(dict1, dict2)
        
        if differences:
            for diff in differences:
                print(diff)
        else:
            print("No differences found")
            
    except Exception as e:
        print(f"Error comparing configurations: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="PolyDiffusion Configuration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all available configurations")
    
    # Preview command
    preview_parser = subparsers.add_parser("preview", help="Preview a configuration")
    preview_parser.add_argument("config", help="Path to configuration file")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a configuration")
    validate_parser.add_argument("config", help="Path to configuration file")
    
    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Compare two configurations")
    diff_parser.add_argument("config1", help="First configuration file")
    diff_parser.add_argument("config2", help="Second configuration file")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_configs()
    elif args.command == "preview":
        preview_config(args.config)
    elif args.command == "validate":
        success = validate_config(args.config)
        sys.exit(0 if success else 1)
    elif args.command == "diff":
        diff_configs(args.config1, args.config2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
