#!/usr/bin/env python3
"""
Installation Verification Script for PolyDiffusion
=================================================

This script checks if all required dependencies are correctly installed.

Usage:
    python verify_installation.py
"""

import sys
from importlib import import_module

# Required packages for basic functionality
REQUIRED_PACKAGES = [
    ("torch", "PyTorch"),
    ("pytorch_lightning", "PyTorch Lightning"),
    ("transformers", "Hugging Face Transformers"),
    ("omegaconf", "OmegaConf"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("yaml", "PyYAML"),
]

# Optional packages for enhanced functionality
OPTIONAL_PACKAGES = [
    ("rdkit", "RDKit (for molecular validation)"),
    ("mlflow", "MLflow (for experiment tracking)"),
    ("matplotlib", "Matplotlib (for visualization)"),
    ("jupyter", "Jupyter (for notebooks)"),
    ("rich", "Rich (for beautiful CLI)"),
]

# Development packages
DEV_PACKAGES = [
    ("pytest", "Pytest (for testing)"),
    ("black", "Black (for code formatting)"),
    ("mypy", "MyPy (for type checking)"),
    ("isort", "isort (for import sorting)"),
]


def check_package(package_name: str, display_name: str) -> bool:
    """Check if a package is installed and importable."""
    try:
        import_module(package_name)
        return True
    except ImportError:
        return False


def main():
    """Main verification function."""
    print("🧬 PolyDiffusion Installation Verification")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ ERROR: Python 3.8+ is required")
        sys.exit(1)
    elif python_version < (3, 10):
        print("⚠️  WARNING: Python 3.10+ is recommended")
    else:
        print("✅ Python version is compatible")
    
    print()
    
    # Check required packages
    print("📦 Checking Required Packages:")
    print("-" * 30)
    
    all_required_ok = True
    for package, display_name in REQUIRED_PACKAGES:
        if check_package(package, display_name):
            print(f"✅ {display_name}")
        else:
            print(f"❌ {display_name} - MISSING")
            all_required_ok = False
    
    print()
    
    # Check optional packages
    print("🔧 Checking Optional Packages:")
    print("-" * 30)
    
    optional_count = 0
    for package, display_name in OPTIONAL_PACKAGES:
        if check_package(package, display_name):
            print(f"✅ {display_name}")
            optional_count += 1
        else:
            print(f"⚪ {display_name} - Optional")
    
    print()
    
    # Check development packages
    print("🛠️  Checking Development Packages:")
    print("-" * 30)
    
    dev_count = 0
    for package, display_name in DEV_PACKAGES:
        if check_package(package, display_name):
            print(f"✅ {display_name}")
            dev_count += 1
        else:
            print(f"⚪ {display_name} - Development only")
    
    print()
    
    # Summary
    print("📊 Installation Summary:")
    print("-" * 25)
    
    if all_required_ok:
        print("✅ All required packages are installed")
        print("🚀 PolyDiffusion is ready to use!")
        
        if optional_count >= 3:
            print("🌟 Most optional features are available")
        
        if dev_count >= 2:
            print("🔧 Development environment is set up")
        
        print("\\n🎯 Quick Start:")
        print("   python scripts/config_tool.py list")
        print("   python scripts/train.py --config configs/experiments/quick_test.yaml")
        
    else:
        print("❌ Some required packages are missing")
        print("\\n📦 Installation commands:")
        print("   conda env create -f environment_minimal.yml")
        print("   # OR")
        print("   pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
