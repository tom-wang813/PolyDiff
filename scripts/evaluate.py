#!/usr/bin/env python3
"""
Evaluation Script for PolyDiffusion Models
=========================================

Evaluate trained models on various metrics including validity, diversity, 
and novelty of generated molecules.

Usage:
    python scripts/evaluate.py --checkpoint experiments/model-1/checkpoints/best-epoch=00-val_loss=0.51.ckpt --num_samples 1000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

import numpy as np
from polydiff.inference import PolymerDiffusionInference

# Configure logging
from typing import List, Dict, Any, Optional


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def calculate_metrics(generated_smiles: List[str], reference_smiles: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    metrics = {}
    
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors, Descriptors
        from rdkit import DataStructs
        import pandas as pd
    except ImportError:
        logging.error("RDKit is required for evaluation. Please install with: pip install rdkit")
        return {"error": "RDKit not available"}
    
    # Basic validity
    valid_mols = []
    valid_smiles = []
    
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_mols.append(mol)
            valid_smiles.append(smiles)
    
    validity = len(valid_mols) / len(generated_smiles) if generated_smiles else 0
    metrics["validity"] = validity
    metrics["valid_count"] = len(valid_mols)
    metrics["total_count"] = len(generated_smiles)
    
    if not valid_mols:
        logging.warning("No valid molecules generated!")
        return metrics
    
    # Uniqueness
    unique_smiles = list(set(valid_smiles))
    uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0
    metrics["uniqueness"] = uniqueness
    metrics["unique_count"] = len(unique_smiles)
    
    # Diversity (internal diversity)
    if len(valid_mols) > 1:
        fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2) for mol in valid_mols]
        similarities = []
        
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            diversity = 1.0 - avg_similarity
            metrics["diversity"] = diversity
            metrics["avg_internal_similarity"] = avg_similarity
    
    # Molecular properties
    properties: Dict[str, List[float]] = {
        "molecular_weights": [],
        "logps": [],
        "tpsas": [],
        "num_atoms": [],
        "num_bonds": [],
        "num_rings": []
    }
    
    for mol in valid_mols:
        properties["molecular_weights"].append(Descriptors.ExactMolWt(mol))
        properties["logps"].append(Descriptors.MolLogP(mol))
        properties["tpsas"].append(Descriptors.TPSA(mol))
        properties["num_atoms"].append(mol.GetNumAtoms())
        properties["num_bonds"].append(mol.GetNumBonds())
        properties["num_rings"].append(Descriptors.RingCount(mol))
    
    # Property statistics
    for prop_name, values in properties.items():
        if values:
            metrics[f"{prop_name}_mean"] = float(np.mean(values))
            metrics[f"{prop_name}_std"] = float(np.std(values))
            metrics[f"{prop_name}_min"] = float(np.min(values))
            metrics[f"{prop_name}_max"] = float(np.max(values))
    
    # Novelty (if reference dataset provided)
    if reference_smiles:
        reference_set = set(reference_smiles)
        novel_smiles = [smiles for smiles in unique_smiles if smiles not in reference_set]
        novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0
        metrics["novelty"] = novelty
        metrics["novel_count"] = len(novel_smiles)
    
    return metrics


def load_reference_smiles(data_path: str) -> List[str]:
    """Load reference SMILES from training data."""
    data_path_obj = Path(data_path)
    
    if not data_path_obj.exists():
        logging.warning(f"Reference data not found: {data_path_obj}")
        return []
    
    try:
        with open(data_path_obj, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(smiles_list)} reference SMILES from {data_path_obj}")
        return smiles_list
    except Exception as e:
        logging.error(f"Failed to load reference data: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PolyDiffusion model performance",
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
        default=1000,
        help="Number of molecules to generate for evaluation"
    )
    parser.add_argument(
        "--reference_data",
        type=str,
        default=None,
        help="Path to reference SMILES data for novelty calculation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for evaluation results (JSON)"
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
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference"
    )
    
    args = parser.parse_args()
    
    # Load model
    logging.info(f"Loading model from {args.checkpoint}")
    try:
        inference_engine = PolymerDiffusionInference(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return
    
    # Generate molecules
    logging.info(f"Generating {args.num_samples} molecules for evaluation...")
    start_time = time.time()
    
    try:
        generated_smiles = inference_engine.generate_molecules(
            num_samples=args.num_samples,
            temperature=args.temperature,
            seed=args.seed
        )
        generation_time = time.time() - start_time
        logging.info(f"Generation completed in {generation_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return
    
    # Load reference data if provided
    reference_smiles = None
    if args.reference_data:
        reference_smiles = load_reference_smiles(args.reference_data)
    
    # Calculate metrics
    logging.info("Calculating evaluation metrics...")
    metrics = calculate_metrics(generated_smiles, reference_smiles)
    
    # Add generation info to metrics
    metrics["generation_time"] = generation_time
    metrics["generation_params"] = {
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint)
    }
    
    # Print results
    print("\\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    print(f"Generated: {metrics['total_count']} molecules")
    print(f"Valid: {metrics['valid_count']} ({metrics['validity']:.2%})")
    print(f"Unique: {metrics['unique_count']} ({metrics['uniqueness']:.2%})")
    
    if "diversity" in metrics:
        print(f"Diversity: {metrics['diversity']:.3f}")
    
    if "novelty" in metrics:
        print(f"Novelty: {metrics['novel_count']}/{metrics['unique_count']} ({metrics['novelty']:.2%})")
    
    print(f"\\nMolecular Properties (mean ± std):")
    if "molecular_weights_mean" in metrics:
        print(f"  Molecular Weight: {metrics['molecular_weights_mean']:.1f} ± {metrics['molecular_weights_std']:.1f}")
        print(f"  LogP: {metrics['logps_mean']:.2f} ± {metrics['logps_std']:.2f}")
        print(f"  TPSA: {metrics['tpsas_mean']:.1f} ± {metrics['tpsas_std']:.1f}")
        print(f"  Atoms: {metrics['num_atoms_mean']:.1f} ± {metrics['num_atoms_std']:.1f}")
        print(f"  Rings: {metrics['num_rings_mean']:.1f} ± {metrics['num_rings_std']:.1f}")
    
    print(f"\\nGeneration Time: {generation_time:.2f} seconds")
    print(f"Speed: {args.num_samples/generation_time:.1f} molecules/second")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add sample molecules to output
        metrics["sample_molecules"] = generated_smiles[:20]  # First 20 as samples
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
