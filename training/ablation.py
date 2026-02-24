"""
Ablation study runner for SoftLogic ViBERT.

Systematically runs experiments with different model configurations
to measure the contribution of each component:
- Token masking
- Multi-view representations
- Soft logic inference
- Individual reasoning rules
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ..core.config import ModelConfig, TrainConfig, get_ablation_configs
from ..data_utils.utils import set_seed


# ============================================================================
# ABLATION CONFIGURATIONS
# ============================================================================

ABLATION_EXPERIMENTS = {
    # Core ablations
    "vibert_only": {
        "description": "ViBERT baseline (no enhancements)",
        "args": ["--no-mask", "--no-multiview", "--no-logic"],
    },
    "mask_only": {
        "description": "ViBERT + Token Masking",
        "args": ["--use-mask", "--no-multiview", "--no-logic"],
    },
    "multiview_no_logic": {
        "description": "ViBERT + Masking + Multi-view (no logic)",
        "args": ["--use-mask", "--use-multiview", "--no-logic"],
    },
    "full_model": {
        "description": "Full SoftLogic ViBERT",
        "args": ["--use-mask", "--use-multiview", "--use-logic"],
    },
    
    # Rule ablations
    "drop_r1": {
        "description": "Full model without r1 (sarcasm rule)",
        "args": ["--use-mask", "--use-multiview", "--use-logic", "--drop-rules", "r1"],
    },
    "drop_r2": {
        "description": "Full model without r2 (strong negative rule)",
        "args": ["--use-mask", "--use-multiview", "--use-logic", "--drop-rules", "r2"],
    },
    "drop_r3": {
        "description": "Full model without r3 (mild positive rule)",
        "args": ["--use-mask", "--use-multiview", "--use-logic", "--drop-rules", "r3"],
    },
    "drop_r4": {
        "description": "Full model without r4 (intense negative rule)",
        "args": ["--use-mask", "--use-multiview", "--use-logic", "--drop-rules", "r4"],
    },
    "drop_r5": {
        "description": "Full model without r5 (inconsistency rule)",
        "args": ["--use-mask", "--use-multiview", "--use-logic", "--drop-rules", "r5"],
    },
    
    # Loss ablations
    "focal_loss": {
        "description": "Full model with Focal Loss",
        "args": ["--use-mask", "--use-multiview", "--use-logic", "--loss-type", "focal"],
    },
    "asymmetric_loss": {
        "description": "Full model with Asymmetric Loss",
        "args": ["--use-mask", "--use-multiview", "--use-logic", "--loss-type", "asymmetric"],
    },
    
    # No masking variations
    "no_mask_with_logic": {
        "description": "Multi-view + Logic without masking",
        "args": ["--no-mask", "--use-multiview", "--use-logic"],
    },
}


# ============================================================================
# ABLATION RUNNER
# ============================================================================

def run_single_ablation(
    experiment_name: str,
    experiment_config: Dict,
    base_args: List[str],
    output_dir: str,
    dry_run: bool = False,
) -> Optional[Dict]:
    """
    Run a single ablation experiment.
    
    Args:
        experiment_name: Name of the experiment
        experiment_config: Configuration dict with args
        base_args: Base arguments for all experiments
        output_dir: Output directory for results
        dry_run: If True, only print command without running
    
    Returns:
        Result dictionary with metrics, or None if failed
    """
    exp_dir = os.path.join(output_dir, experiment_name)
    
    # Build command
    cmd = [
        sys.executable, "-m", "softlogic_vibert.training.train",
        "--output-dir", exp_dir,
        "--experiment-name", experiment_name,
    ]
    cmd.extend(base_args)
    cmd.extend(experiment_config["args"])
    
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"Description: {experiment_config['description']}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("[DRY RUN] Skipping execution")
        return None
    
    try:
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        
        if result.returncode != 0:
            print(f"FAILED with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Load results
        summary_path = os.path.join(exp_dir, experiment_name, "train_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
            return {
                "experiment": experiment_name,
                "description": experiment_config["description"],
                "best_macro_f1": summary.get("best_macro_f1", 0),
                "status": "success",
            }
        else:
            # Try parent directory
            summary_path = os.path.join(exp_dir, "train_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                return {
                    "experiment": experiment_name,
                    "description": experiment_config["description"],
                    "best_macro_f1": summary.get("best_macro_f1", 0),
                    "status": "success",
                }
            
        return {
            "experiment": experiment_name,
            "status": "no_results",
        }
        
    except Exception as e:
        print(f"Exception: {e}")
        return {
            "experiment": experiment_name,
            "status": "error",
            "error": str(e),
        }


def run_ablation_study(
    experiments: List[str],
    data_path: str,
    output_dir: str,
    model_name: str = "Fsoft-AIC/videberta-base",
    epochs: int = 5,
    batch_size: int = 16,
    seed: int = 42,
    dry_run: bool = False,
    val_path: Optional[str] = None,
) -> Dict:
    """
    Run a complete ablation study.
    
    Args:
        experiments: List of experiment names to run
        data_path: Path to training data
        output_dir: Output directory
        model_name: Model name
        epochs: Number of epochs
        batch_size: Batch size
        seed: Random seed
        dry_run: If True, only print commands
        val_path: Optional validation data path
    
    Returns:
        Dictionary with all results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = os.path.join(output_dir, f"ablation_study_{timestamp}")
    os.makedirs(study_dir, exist_ok=True)
    
    # Base arguments
    base_args = [
        "--data-path", data_path,
        "--model-name", model_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
    ]
    
    if val_path:
        base_args.extend(["--val-path", val_path])
    
    # Save study configuration
    study_config = {
        "timestamp": timestamp,
        "experiments": experiments,
        "base_args": base_args,
        "data_path": data_path,
    }
    
    config_path = os.path.join(study_dir, "study_config.json")
    with open(config_path, "w") as f:
        json.dump(study_config, f, indent=2)
    
    # Run experiments
    results = []
    
    for exp_name in experiments:
        if exp_name not in ABLATION_EXPERIMENTS:
            print(f"Warning: Unknown experiment '{exp_name}', skipping")
            continue
        
        exp_config = ABLATION_EXPERIMENTS[exp_name]
        result = run_single_ablation(
            experiment_name=exp_name,
            experiment_config=exp_config,
            base_args=base_args,
            output_dir=study_dir,
            dry_run=dry_run,
        )
        
        if result:
            results.append(result)
    
    # Create summary
    summary = {
        "timestamp": timestamp,
        "num_experiments": len(results),
        "results": results,
    }
    
    # Sort by macro F1
    successful = [r for r in results if r.get("status") == "success"]
    if successful:
        successful.sort(key=lambda x: x.get("best_macro_f1", 0), reverse=True)
        summary["ranking"] = [
            {"experiment": r["experiment"], "macro_f1": r["best_macro_f1"]}
            for r in successful
        ]
    
    # Save summary
    summary_path = os.path.join(study_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    
    if successful:
        print("\nRanking by Macro-F1:")
        for i, res in enumerate(successful, 1):
            print(f"  {i}. {res['experiment']}: {res['best_macro_f1']:.4f}")
            print(f"     {res['description']}")
    
    print(f"\nResults saved to: {study_dir}")
    
    return summary


def compare_results(study_dir: str) -> None:
    """
    Compare and visualize results from an ablation study.
    
    Args:
        study_dir: Path to ablation study directory
    """
    summary_path = os.path.join(study_dir, "ablation_summary.json")
    
    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPARISON")
    print("=" * 70)
    
    results = summary.get("results", [])
    successful = [r for r in results if r.get("status") == "success"]
    
    if not successful:
        print("No successful experiments found.")
        return
    
    # Find baseline and full model
    baseline = next((r for r in successful if r["experiment"] == "vibert_only"), None)
    full_model = next((r for r in successful if r["experiment"] == "full_model"), None)
    
    print("\nResults Table:")
    print("-" * 70)
    print(f"{'Experiment':<30} {'Macro-F1':>10} {'Δ vs Baseline':>15} {'Δ vs Full':>12}")
    print("-" * 70)
    
    baseline_f1 = baseline["best_macro_f1"] if baseline else 0
    full_f1 = full_model["best_macro_f1"] if full_model else 0
    
    for res in sorted(successful, key=lambda x: x["best_macro_f1"], reverse=True):
        f1 = res["best_macro_f1"]
        delta_base = f1 - baseline_f1
        delta_full = f1 - full_f1
        
        delta_base_str = f"+{delta_base:.4f}" if delta_base > 0 else f"{delta_base:.4f}"
        delta_full_str = f"+{delta_full:.4f}" if delta_full > 0 else f"{delta_full:.4f}"
        
        print(f"{res['experiment']:<30} {f1:>10.4f} {delta_base_str:>15} {delta_full_str:>12}")
    
    print("-" * 70)
    
    # Component contribution analysis
    print("\nComponent Contribution Analysis:")
    
    if baseline and full_model:
        total_improvement = full_f1 - baseline_f1
        print(f"\n  Total improvement over baseline: {total_improvement:+.4f}")
        
        # Estimate component contributions
        mask_only = next((r for r in successful if r["experiment"] == "mask_only"), None)
        multiview_no_logic = next((r for r in successful if r["experiment"] == "multiview_no_logic"), None)
        
        if mask_only:
            mask_contrib = mask_only["best_macro_f1"] - baseline_f1
            print(f"  Token Masking contribution: {mask_contrib:+.4f}")
        
        if multiview_no_logic and mask_only:
            multiview_contrib = multiview_no_logic["best_macro_f1"] - mask_only["best_macro_f1"]
            print(f"  Multi-view contribution: {multiview_contrib:+.4f}")
        
        if full_model and multiview_no_logic:
            logic_contrib = full_f1 - multiview_no_logic["best_macro_f1"]
            print(f"  Soft Logic contribution: {logic_contrib:+.4f}")
    
    # Rule importance analysis
    print("\nRule Importance Analysis:")
    
    rule_ablations = {
        "r1 (Sarcasm)": "drop_r1",
        "r2 (Strong Negative)": "drop_r2",
        "r3 (Mild Positive)": "drop_r3",
        "r4 (Intense Negative)": "drop_r4",
        "r5 (Inconsistency)": "drop_r5",
    }
    
    for rule_name, exp_name in rule_ablations.items():
        result = next((r for r in successful if r["experiment"] == exp_name), None)
        if result and full_model:
            importance = full_f1 - result["best_macro_f1"]
            direction = "down" if importance > 0 else "up" if importance < 0 else "same"
            print(f"  {rule_name}: {direction} {abs(importance):.4f} when removed")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run ablation study for SoftLogic ViBERT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run ablation experiments")
    run_parser.add_argument("--data-path", type=str, required=True,
                           help="Path to training data")
    run_parser.add_argument("--val-path", type=str, default=None,
                           help="Path to validation data")
    run_parser.add_argument("--output-dir", type=str, default="ablation_results",
                           help="Output directory")
    run_parser.add_argument("--experiments", type=str, default="all",
                           help="Comma-separated list of experiments, or 'all' or 'core'")
    run_parser.add_argument("--model-name", type=str, default="Fsoft-AIC/videberta-base",
                           help="Model name")
    run_parser.add_argument("--epochs", type=int, default=5,
                           help="Number of epochs")
    run_parser.add_argument("--batch-size", type=int, default=16,
                           help="Batch size")
    run_parser.add_argument("--seed", type=int, default=42,
                           help="Random seed")
    run_parser.add_argument("--dry-run", action="store_true",
                           help="Only print commands without running")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare ablation results")
    compare_parser.add_argument("--study-dir", type=str, required=True,
                               help="Path to ablation study directory")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available experiments")
    
    args = parser.parse_args()
    
    if args.command == "run":
        # Determine experiments to run
        if args.experiments == "all":
            experiments = list(ABLATION_EXPERIMENTS.keys())
        elif args.experiments == "core":
            experiments = ["vibert_only", "mask_only", "multiview_no_logic", "full_model"]
        else:
            experiments = [e.strip() for e in args.experiments.split(",")]
        
        run_ablation_study(
            experiments=experiments,
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            dry_run=args.dry_run,
            val_path=args.val_path,
        )
    
    elif args.command == "compare":
        compare_results(args.study_dir)
    
    elif args.command == "list":
        print("\nAvailable Ablation Experiments:")
        print("=" * 60)
        for name, config in ABLATION_EXPERIMENTS.items():
            print(f"\n{name}:")
            print(f"  {config['description']}")
            print(f"  Args: {' '.join(config['args'])}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
