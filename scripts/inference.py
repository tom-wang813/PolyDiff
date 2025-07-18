#!/usr/bin/env python3
"""
Inference Script for BERT-based Discrete Diffusion Model
======================================================

A script to generate molecules using a trained PolyDiff model.

Usage:
    python scripts/inference.py --checkpoint path/to/model.ckpt --num_samples 10
    python scripts/inference.py --checkpoint experiments/model-1/checkpoints/best-epoch=00-val_loss=0.51.ckpt --num_samples 20 --output results.json --validate
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

from polydiff.inference import PolymerDiffusionInference

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def validate_smiles(smiles_list: List[str]) -> List[dict]:
    """Validate generated SMILES using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        logging.warning("RDKit not available. Skipping SMILES validation.")
        return [{"smiles": smiles, "valid": None} for smiles in smiles_list]
    
    results = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            results.append({
                "smiles": smiles,
                "valid": True,
                "mol_weight": Descriptors.ExactMolWt(mol),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_rings": Descriptors.RingCount(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol)
            })
        else:
            results.append({
                "smiles": smiles,
                "valid": False
            })
    
    return results


def calculate_diversity(smiles_list: List[str]) -> float:
    """Calculate Tanimoto diversity of generated molecules."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        from rdkit import DataStructs
        import numpy as np
    except ImportError:
        logging.warning("RDKit not available. Skipping diversity calculation.")
        return None
    
    # Generate Morgan fingerprints
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
            fps.append(fp)
    
    if len(fps) < 2:
        return None
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
    
    # Diversity is 1 - average similarity
    avg_similarity = np.mean(similarities)
    diversity = 1.0 - avg_similarity
    
    return diversity


def main():
    parser = argparse.ArgumentParser(
        description="Generate molecules using PolyDiff",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10, 
        help="Number of molecules to generate"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=None, 
        help="Maximum sequence length (default: use model config)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0, 
        help="Sampling temperature"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Output file path (JSON format)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="Device to use for inference (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--validate", 
        action="store_true", 
        help="Validate generated SMILES with RDKit"
    )
    parser.add_argument(
        "--diversity", 
        action="store_true", 
        help="Calculate diversity metrics"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for generation"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Print detailed generation information"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Initialize inference engine
    logging.info(f"Loading model from {args.checkpoint}")
    try:
        inference_engine = PolymerDiffusionInference(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return
    
    # Generate molecules
    logging.info(f"Generating {args.num_samples} molecules...")
    try:
        generated_smiles = inference_engine.generate_molecules(
            num_samples=args.num_samples,
            max_length=args.max_length,
            temperature=args.temperature,
            seed=args.seed,
            batch_size=args.batch_size
        )
        logging.info(f"Successfully generated {len(generated_smiles)} molecules")
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return
    
    # Prepare results
    results = {
        "generated_smiles": generated_smiles,
        "generation_params": {
            "num_samples": args.num_samples,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "checkpoint": str(args.checkpoint)
        }
    }
    
    # Validate if requested
    if args.validate:
        logging.info("Validating generated SMILES...")
        validation_results = validate_smiles(generated_smiles)
        results["validation"] = validation_results
        
        valid_count = sum(1 for r in validation_results if r.get("valid", False))
        validity_rate = valid_count / len(generated_smiles) if generated_smiles else 0
        results["validity_rate"] = validity_rate
        logging.info(f"Validation complete: {valid_count}/{len(generated_smiles)} valid SMILES ({validity_rate:.2%})")
    
    # Calculate diversity if requested
    if args.diversity:
        logging.info("Calculating molecular diversity...")
        diversity_score = calculate_diversity(generated_smiles)
        if diversity_score is not None:
            results["diversity_score"] = diversity_score
            logging.info(f"Diversity score: {diversity_score:.3f}")
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_path}")
    else:
        # Print to console
        print("\\n" + "="*50)
        print("Generated SMILES:")
        print("="*50)
        for i, smiles in enumerate(generated_smiles, 1):
            print(f"{i:3d}: {smiles}")
        
        if args.validate and "validation" in results:
            print(f"\\nValidity: {results['validity_rate']:.2%} ({results.get('validation', [])} valid molecules)")
        
        if args.diversity and "diversity_score" in results:
            print(f"Diversity: {results['diversity_score']:.3f}")
    
    if args.verbose:
        print("\\n" + "="*50)
        print("Generation Parameters:")
        print("="*50)
        for key, value in results["generation_params"].items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
