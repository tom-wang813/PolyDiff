#!/usr/bin/env python3
"""
Polymer Diffusion Model Demo
===========================

This script demonstrates the key features of the polymer diffusion inference system.
"""

import os
import sys

from polydiff.inference import PolymerDiffusionInference


def main():
    print("🧪 Polymer Diffusion Model Inference Demo")
    print("=" * 50)

    # Initialize model
    model_path = (
        "/Users/wang-work/polydiffusion/experiments/model-1/checkpoints/"
        "epoch=2-step=19572-val_loss=0.6891.ckpt"
    )

    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint not found: {model_path}")
        return

    print("📂 Loading model...")
    inference = PolymerDiffusionInference(
        model_path=model_path,
        device="auto",
        schedule_type="linear",
    )

    print("✅ Model loaded successfully!")
    print(f"   Device: {inference.device}")
    print(f"   Diffusion steps: {inference.diffusion_steps}")
    print(f"   Vocab size: {inference.model.bert_config.vocab_size}")
    print(f"   Max length: {inference.model.bert_config.max_position_embeddings}")

    # Test 1: Basic Generation
    print("\n🎲 Test 1: Basic Molecule Generation")
    print("-" * 30)

    molecules = inference.generate_molecules(
        num_samples=3, max_length=64, temperature=1.0, seed=42
    )

    print("Generated molecules:")
    for i, mol in enumerate(molecules):
        print(f"  {i+1}. {mol[:80]}{'...' if len(mol) > 80 else ''}")

    # Test 2: Temperature Effects
    print("\n🌡️ Test 2: Temperature Effects")
    print("-" * 30)

    for temp in [0.5, 1.0, 1.5]:
        print(f"\nTemperature = {temp}:")
        temp_molecules = inference.generate_molecules(
            num_samples=2, max_length=32, temperature=temp, seed=42
        )
        for i, mol in enumerate(temp_molecules):
            print(f"  {mol[:60]}{'...' if len(mol) > 60 else ''}")

    # Test 3: Sampling Methods
    print("\n🎯 Test 3: Advanced Sampling")
    print("-" * 30)

    # Top-k sampling
    print("Top-k sampling (k=50):")
    topk_molecules = inference.generate_molecules(
        num_samples=2, max_length=32, top_k=50, seed=42
    )
    for mol in topk_molecules:
        print(f"  {mol[:60]}{'...' if len(mol) > 60 else ''}")

    # Top-p sampling
    print("\nTop-p sampling (p=0.9):")
    topp_molecules = inference.generate_molecules(
        num_samples=2, max_length=32, top_p=0.9, seed=42
    )
    for mol in topp_molecules:
        print(f"  {mol[:60]}{'...' if len(mol) > 60 else ''}")

    # Test 4: Molecule Similarity
    print("\n🔍 Test 4: Molecule Similarity")
    print("-" * 30)

    test_molecules = [
        "CCO",  # Ethanol
        "CCC",  # Propane
        "CCCC",  # Butane
        "CCO",  # Ethanol (duplicate)
    ]

    print("Computing pairwise similarities:")
    for i, mol1 in enumerate(test_molecules):
        for j, mol2 in enumerate(test_molecules):
            if i <= j:  # Only compute upper triangle
                try:
                    sim = inference.compute_molecule_similarity(mol1, mol2)
                    print(f"  {mol1} vs {mol2}: {sim:.3f}")
                except Exception as e:
                    print(f"  {mol1} vs {mol2}: Error - {str(e)[:50]}")

    # Test 5: Validation with RDKit (if available)
    print("\n🔬 Test 5: Molecule Validation")
    print("-" * 30)

    try:
        # Make sure to import rdkit correctly
        from rdkit import Chem

        # MolFromSmiles is a function in the Chem module
        # Generate a batch for validation
        validation_molecules = inference.generate_molecules(
            num_samples=10, max_length=32, temperature=1.0, seed=123
        )
        valid_count = 0
        for i, mol in enumerate(validation_molecules):
            rdkit_mol = Chem.MolFromSmiles(mol)
            is_valid = rdkit_mol is not None
            valid_count += is_valid
            status = "✅" if is_valid else "❌"
            print(f"  {i+1:2d}. {status} {mol[:50]}{'...' if len(mol) > 50 else ''}")

        validity_rate = valid_count / len(validation_molecules) * 100
        print("\n📊 Validation Results:")
        print(
            f"   Valid molecules: {valid_count}/{len(validation_molecules)} "
            f"({validity_rate:.1f}%)"
        )

    except ImportError:
        print("⚠️ RDKit not available for validation")
        print("   To install: conda install -c rdkit rdkit")

    # Performance Summary
    print("\n📈 Demo Summary")
    print("-" * 30)
    print("✅ Successfully demonstrated:")
    print("   • Model loading and initialization")
    print("   • Basic molecule generation")
    print("   • Temperature-controlled sampling")
    print("   • Advanced sampling methods (top-k, top-p)")
    print("   • Molecule similarity computation")
    print("   • Chemical validation (if RDKit available)")

    print("\n🎯 Next Steps:")
    print("   • Use Jupyter notebooks for interactive exploration")
    print("   • Experiment with different sampling parameters")
    print("   • Analyze molecular properties and diversity")
    print("   • Fine-tune model for better chemical validity")


if __name__ == "__main__":
    main()
