"""
Training script for SoftLogic ViBERT Sentiment Analysis.

Features:
- Full training pipeline with validation
- Early stopping with macro-F1 monitoring
- Gradient accumulation for effective larger batch sizes
- Comprehensive logging and checkpointing
- Ablation support via command line flags
- Mixed precision training support
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from ..core.config import ModelConfig, TrainConfig, ExperimentConfig, get_ablation_configs
from ..data_utils.data import (
    SentimentDataset,
    load_data_file,
    load_hf_dataset,
    collate_batch,
    prepare_dataloaders,
    load_and_prepare_data,
    load_slang_lexicon,
)
from ..core.losses import SoftLogicLoss, compute_pos_weight
from ..core.metrics import multilabel_f1, multilabel_metrics, evaluate_model, find_optimal_threshold
from ..core.model import SoftLogicViBERT
from ..data_utils.utils import (
    build_label_map,
    ensure_list,
    build_tfidf_cache,
    set_seed,
    count_parameters,
    get_model_summary,
    stratified_split_multilabel,
)


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SoftLogic ViBERT for Vietnamese GenZ Sentiment Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data-path", type=str, default="output_data.json",
                           help="Path to training data (JSON/JSONL)")
    data_group.add_argument("--val-path", type=str, default=None,
                           help="Path to validation data (optional)")
    data_group.add_argument("--test-path", type=str, default=None,
                           help="Path to test data (optional)")
    data_group.add_argument("--hf-dataset", type=str, default=None,
                           help="HuggingFace dataset name (e.g., tridm/UIT-VSMEC)")
    data_group.add_argument("--hf-split", type=str, default="train",
                           help="HuggingFace train split")
    data_group.add_argument("--hf-val-split", type=str, default="validation",
                           help="HuggingFace validation split")
    data_group.add_argument("--val-ratio", type=float, default=0.1,
                           help="Validation split ratio if no val-path")
    
    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model-name", type=str, default="Fsoft-AIC/videberta-base",
                            help="Pretrained model name/path")
    model_group.add_argument("--max-len", type=int, default=128,
                            help="Maximum sequence length")
    model_group.add_argument("--proj-size", type=int, default=128,
                            help="Projection dimension for views")
    model_group.add_argument("--dropout", type=float, default=0.3,
                            help="Dropout rate")
    model_group.add_argument("--tfidf-cache", type=str, default="tfidf_cache.pt",
                            help="Path to TF-IDF cache file")
    model_group.add_argument("--tfidf-default-score", type=float, default=0.1,
                            help="Default TF-IDF score for special/unknown tokens")
    model_group.add_argument("--tfidf-rebuild", action="store_true",
                            help="Rebuild TF-IDF cache even if it exists")
    model_group.add_argument("--pretrain-ckpt", type=str, default=None,
                            help="Path to pretrained checkpoint (from mask-guided pretraining)")
    
    # Ablation flags
    ablation_group = parser.add_argument_group("Ablation")
    ablation_group.add_argument("--use-mask", action="store_true", dest="use_mask")
    ablation_group.add_argument("--no-mask", action="store_false", dest="use_mask")
    parser.set_defaults(use_mask=True)
    
    ablation_group.add_argument("--use-multiview", action="store_true", dest="use_multiview")
    ablation_group.add_argument("--no-multiview", action="store_false", dest="use_multiview")
    parser.set_defaults(use_multiview=True)
    
    ablation_group.add_argument("--use-logic", action="store_true", dest="use_logic")
    ablation_group.add_argument("--no-logic", action="store_false", dest="use_logic")
    parser.set_defaults(use_logic=True)
    
    ablation_group.add_argument("--drop-rules", type=str, default="",
                               help="Comma-separated rules to drop (e.g., r1,r5)")
    ablation_group.add_argument("--ablation-mode", type=str, default="full",
                               choices=["full", "vibert_only", "mask_only", "multiview_only"],
                               help="Predefined ablation configuration")
    
    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--batch-size", type=int, default=16,
                            help="Training batch size")
    train_group.add_argument("--effective-batch-size", type=int, default=32,
                            help="Effective batch size (for gradient accumulation)")
    train_group.add_argument("--epochs", type=int, default=20,
                            help="Number of training epochs")
    train_group.add_argument("--lr", type=float, default=2e-5,
                            help="Learning rate")
    train_group.add_argument("--encoder-lr", type=float, default=1e-5,
                            help="Learning rate for encoder (default: same as --lr)")
    train_group.add_argument("--head-lr", type=float, default=5e-4,
                            help="Learning rate for classification heads")
    train_group.add_argument("--weight-decay", type=float, default=0.01,
                            help="Weight decay for optimizer")
    train_group.add_argument("--warmup-ratio", type=float, default=0.1,
                            help="Warmup ratio for scheduler")
    train_group.add_argument("--grad-clip", type=float, default=1.0,
                            help="Gradient clipping value")
    train_group.add_argument("--patience", type=int, default=3,
                            help="Early stopping patience")
    train_group.add_argument("--seed", type=int, default=42,
                            help="Random seed")
    
    # Loss arguments
    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument("--loss-type", type=str, default="bce",
                           choices=["bce", "focal", "asymmetric"],
                           help="Classification loss type")
    loss_group.add_argument("--mask-lambda", type=float, default=1e-3,
                           help="Mask sparsity regularization weight")
    loss_group.add_argument("--focal-gamma", type=float, default=2.0,
                           help="Focal loss gamma parameter")
    loss_group.add_argument("--label-smoothing", type=float, default=0.0,
                           help="Label smoothing factor")
    loss_group.add_argument("--deep-supervision-lambda", type=float, default=0.5,
                           help="Weight for semantic+lexical polarity supervision")
    loss_group.add_argument("--slang-preserve-lambda", type=float, default=0.5,
                           help="Weight for slang-preservation mask loss")
    loss_group.add_argument("--mask-sparsity-lambda", type=float, default=0.1,
                           help="Weight for non-slang mask sparsity loss")
    loss_group.add_argument("--use-polarity-supervision", action="store_true", dest="use_polarity_supervision",
                           help="Enable polarity supervision on semantic/lexical predicates")
    loss_group.add_argument("--no-polarity-supervision", action="store_false", dest="use_polarity_supervision",
                           help="Disable polarity supervision")
    parser.set_defaults(use_polarity_supervision=True)
    loss_group.add_argument("--use-slang-preserve", action="store_true", dest="use_slang_preserve",
                           help="Enable slang-preservation mask loss")
    loss_group.add_argument("--no-slang-preserve", action="store_false", dest="use_slang_preserve",
                           help="Disable slang-preservation mask loss")
    parser.set_defaults(use_slang_preserve=True)
    loss_group.add_argument("--use-mask-sparsity", action="store_true", dest="use_mask_sparsity",
                           help="Enable non-slang mask sparsity loss")
    loss_group.add_argument("--no-mask-sparsity", action="store_false", dest="use_mask_sparsity",
                           help="Disable non-slang mask sparsity loss")
    parser.set_defaults(use_mask_sparsity=True)
    loss_group.add_argument("--pos-labels", type=str, default="enjoyment",
                           help="Comma-separated positive labels for polarity supervision")
    loss_group.add_argument("--neg-labels", type=str, default="anger,disgust,fear,sadness",
                           help="Comma-separated negative labels for polarity supervision")
    loss_group.add_argument("--slang-lexicon", type=str, default="dictionary.json",
                           help="Path to slang lexicon JSON for slang mask generation")
    
    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-dir", type=str, default="softlogic_outputs",
                             help="Output directory for checkpoints and logs")
    output_group.add_argument("--experiment-name", type=str, default=None,
                             help="Experiment name (default: auto-generated)")
    output_group.add_argument("--save-all-checkpoints", action="store_true",
                             help="Save checkpoint at every epoch")
    
    # Other arguments
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Classification threshold")
    parser.add_argument("--log-interval", type=int, default=50,
                       help="Logging interval (steps)")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--num-workers", type=int, default=0,
                       help="Number of data loading workers")
    
    return parser.parse_args()


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class Trainer:
    """
    Training orchestrator for SoftLogic ViBERT.
    
    Handles:
    - Training loop with gradient accumulation
    - Validation and early stopping
    - Logging and checkpointing
    - Mixed precision training
    """
    
    def __init__(
        self,
        model: SoftLogicViBERT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: SoftLogicLoss,
        config: TrainConfig,
        model_config: ModelConfig,
        label_map: Dict[str, int],
        label_list: List[str],
        tokenizer,
        output_dir: str,
        device: torch.device,
        pos_label_indices: List[int],
        neg_label_indices: List[int],
        use_fp16: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.model_config = model_config
        self.label_map = label_map
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.device = device
        self.pos_label_indices = pos_label_indices
        self.neg_label_indices = neg_label_indices
        
        # Mixed precision
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_fp16 else None
        
        # Special tokens for masking
        self.special_token_ids = torch.tensor([
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ], device=device)
        
        # Training state
        self.global_step = 0
        self.best_metric = -1.0
        self.patience_counter = 0
        self.training_history = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        mask_losses = []
        rule_stats = {f"r{i}": [] for i in range(1, 6)}
        mask_means = []
        
        accumulation_steps = self.config.accumulation_steps
        
        for step, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            labels = batch["labels"].to(self.device)
            prag = batch.get("prag_features")
            if prag is not None:
                prag = prag.to(self.device)
            slang_mask = batch.get("slang_mask")
            if slang_mask is not None:
                slang_mask = slang_mask.to(self.device)
            slang_weights = batch.get("slang_weights")
            if slang_weights is not None:
                slang_weights = slang_weights.to(self.device)
            special_tokens_mask = batch.get("special_tokens_mask")
            if special_tokens_mask is not None:
                special_tokens_mask = special_tokens_mask.to(self.device)
            
            # Forward pass
            if self.use_fp16:
                with autocast():
                    logits, extras = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        prag_features=prag,
                        special_token_ids=self.special_token_ids,
                        return_extras=True,
                    )
                    
                    polarity_targets, polarity_mask = self._build_polarity_targets(labels)
                    loss_dict = self.loss_fn(
                        logits=logits,
                        targets=labels,
                        mask_mean=extras.get("mask_mean"),
                        rules=extras.get("rules"),
                        mask_values=extras.get("mask_vals"),
                        attention_mask=attention_mask,
                        special_tokens_mask=special_tokens_mask,
                        slang_mask=slang_mask,
                        slang_weights=slang_weights,
                        polarity_targets=polarity_targets,
                        polarity_mask=polarity_mask,
                        predicates=extras.get("predicates"),
                    )
                    loss = loss_dict["loss"] / accumulation_steps
            else:
                logits, extras = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    prag_features=prag,
                    special_token_ids=self.special_token_ids,
                    return_extras=True,
                )
                
                polarity_targets, polarity_mask = self._build_polarity_targets(labels)
                loss_dict = self.loss_fn(
                    logits=logits,
                    targets=labels,
                    mask_mean=extras.get("mask_mean"),
                    rules=extras.get("rules"),
                    mask_values=extras.get("mask_vals"),
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                    slang_mask=slang_mask,
                    slang_weights=slang_weights,
                    polarity_targets=polarity_targets,
                    polarity_mask=polarity_mask,
                    predicates=extras.get("predicates"),
                )
                loss = loss_dict["loss"] / accumulation_steps
            
            # Backward pass
            if self.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            
            # Collect statistics
            if "mask_loss" in loss_dict:
                mask_losses.append(loss_dict["mask_loss"].item())
            if "mask_mean" in extras:
                mask_means.append(extras["mask_mean"].item())
            if extras.get("rules"):
                for k, v in extras["rules"].items():
                    if k in rule_stats:
                        rule_stats[k].append(v.mean().item())
            
            # Optimizer step with gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                if self.use_fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            
            # Logging
            if (step + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Step {step + 1}/{len(self.train_loader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
        
        # Compile epoch statistics
        epoch_stats = {
            "train_loss": total_loss / len(self.train_loader),
        }
        
        if mask_means:
            epoch_stats["train_mask_mean"] = sum(mask_means) / len(mask_means)
        
        for k, values in rule_stats.items():
            if values:
                epoch_stats[f"train_{k}_mean"] = sum(values) / len(values)
        
        return epoch_stats

    def _build_polarity_targets(
        self, labels: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build polarity targets and masks from multi-label targets."""
        if not self.pos_label_indices and not self.neg_label_indices:
            return None, None

        device = labels.device
        if self.pos_label_indices:
            pos_any = labels[:, self.pos_label_indices].sum(dim=1).clamp(max=1)
        else:
            pos_any = torch.zeros(labels.size(0), device=device)

        if self.neg_label_indices:
            neg_any = labels[:, self.neg_label_indices].sum(dim=1).clamp(max=1)
        else:
            neg_any = torch.zeros(labels.size(0), device=device)

        polarity_targets = torch.stack([pos_any, neg_any], dim=1)
        polarity_mask = (polarity_targets.sum(dim=1) > 0).float()

        return polarity_targets, polarity_mask
    
    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        all_logits = []
        all_labels = []
        rule_means = {f"r{i}": [] for i in range(1, 6)}
        mask_means = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                labels = batch["labels"].to(self.device)
                prag = batch.get("prag_features")
                if prag is not None:
                    prag = prag.to(self.device)
                
                logits, extras = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    prag_features=prag,
                    special_token_ids=self.special_token_ids,
                    return_extras=True,
                )
                
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                
                # Collect statistics
                if extras.get("rules"):
                    for k, v in extras["rules"].items():
                        if k in rule_means:
                            rule_means[k].append(v.mean().cpu())
                
                if "mask_mean" in extras:
                    mask_means.append(extras["mask_mean"].cpu())
        
        # Compute metrics
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        metrics = multilabel_metrics(
            logits, labels,
            threshold=self.config.threshold,
            label_names=self.label_list,
        )
        
        # Add rule and mask statistics
        if mask_means:
            metrics["val_mask_mean"] = torch.stack(mask_means).mean().item()
        
        for k, values in rule_means.items():
            if values:
                metrics[f"val_{k}_mean"] = torch.stack(values).mean().item()
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "model_config": self.model_config.__dict__,
            "label_map": self.label_map,
            "label_list": self.label_list,
            "tokenizer_name": self.model_config.model_name,
            "metrics": metrics,
            "global_step": self.global_step,
        }
        
        if is_best:
            # Save best model (state dict only for efficiency)
            path = self.output_dir / "softlogic_vibert_state.pt"
            torch.save(checkpoint, path)
            
            # Also save full model for easy loading
            full_checkpoint = checkpoint.copy()
            full_checkpoint["model"] = self.model
            full_path = self.output_dir / "softlogic_vibert_full.pt"
            torch.save(full_checkpoint, full_path)
            
            print(f"  Saved best model to {path}")
        
        if self.config.save_all_checkpoints if hasattr(self.config, 'save_all_checkpoints') else False:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)
    
    def train(self) -> Dict[str, float]:
        """
        Main training loop.
        
        Returns the best validation metrics.
        """
        print(f"\n{'='*60}")
        print(f"Starting training")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Effective batch: {self.config.effective_batch_size}")
        print(f"  Learning rate: {self.config.lr}")
        print(f"  Use mask: {self.model_config.use_mask}")
        print(f"  Use multiview: {self.model_config.use_multiview}")
        print(f"  Use logic: {self.model_config.use_logic}")
        print(f"{'='*60}\n")
        
        best_metrics = {}
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print("-" * 40)
            
            # Training
            train_stats = self._train_epoch(epoch)
            
            # Validation
            val_metrics = self._validate()
            
            # Combine stats
            epoch_stats = {**train_stats, **val_metrics}
            self.training_history.append(epoch_stats)
            
            # Print key metrics
            print(f"\n  Epoch {epoch} Summary:")
            print(f"    Train Loss: {train_stats['train_loss']:.4f}")
            print(f"    Val Macro-F1: {val_metrics['macro_f1']:.4f}")
            print(f"    Val Micro-F1: {val_metrics['micro_f1']:.4f}")
            
            if "val_mask_mean" in val_metrics:
                print(f"    Mask Mean: {val_metrics['val_mask_mean']:.4f}")
            
            # Print per-class F1 if available
            for label in self.label_list:
                key = f"{label}_f1"
                if key in val_metrics:
                    print(f"    {label} F1: {val_metrics[key]:.4f}")
            
            # Check for improvement
            current_metric = val_metrics["macro_f1"]
            
            if current_metric > self.best_metric:
                improvement = current_metric - self.best_metric
                self.best_metric = current_metric
                self.patience_counter = 0
                best_metrics = val_metrics
                
                print(f"\n  ★ New best! Macro-F1 improved by {improvement:.4f}")
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                print(f"\n  No improvement. Patience: {self.patience_counter}/{self.config.patience}")
                
                if self.patience_counter >= self.config.patience:
                    print(f"\n  Early stopping triggered after {epoch} epochs")
                    break
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save final summary
        summary = {
            "best_macro_f1": self.best_metric,
            "total_epochs": epoch,
            "model_config": self.model_config.__dict__,
            "train_config": self.config.__dict__,
            "label_list": self.label_list,
        }
        
        summary_path = self.output_dir / "train_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"  Best Macro-F1: {self.best_metric:.4f}")
        print(f"  Results saved to: {self.output_dir}")
        print(f"{'='*60}")
        
        return best_metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main() -> None:
    """Main training entry point."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load and prepare data
    print("\nLoading data...")
    train_rows, val_rows, label_map, label_list, pre_stats, post_stats = load_and_prepare_data(
        data_path=args.data_path if not args.hf_dataset else None,
        val_path=args.val_path,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_val_split=args.hf_val_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        return_stats=True,
    )
    
    print(f"  Training samples: {len(train_rows)}")
    print(f"  Validation samples: {len(val_rows)}")
    print(f"  Labels: {label_list}")
    pre_counts = ", ".join([f"{k}:{v}" for k, v in sorted(pre_stats["label_counts"].items())])
    post_counts = ", ".join([f"{k}:{v}" for k, v in sorted(post_stats["label_counts"].items())])
    print(f"  Train label counts (pre-oversample): {pre_counts}")
    print(f"  Train label counts (post-oversample): {post_counts}")
    
    # Load slang lexicon for mask supervision
    if args.slang_lexicon and Path(args.slang_lexicon).exists():
        slang_terms, slang_weight_map = load_slang_lexicon(args.slang_lexicon)
    else:
        slang_terms, slang_weight_map = [], {}
        if args.slang_lexicon:
            print(f"  Warning: slang lexicon not found at {args.slang_lexicon}. Skipping slang masks.")

    # Build TF-IDF cache for statistical gating
    if args.use_mask:
        tfidf_path = Path(args.tfidf_cache)
        if args.tfidf_rebuild or not tfidf_path.exists():
            print(f"\nBuilding TF-IDF cache at {tfidf_path}...")
            build_tfidf_cache(
                train_rows=train_rows,
                tokenizer=tokenizer,
                cache_path=str(tfidf_path),
                default_score=args.tfidf_default_score,
            )

    # Create datasets and dataloaders
    train_loader, val_loader = prepare_dataloaders(
        train_rows=train_rows,
        val_rows=val_rows,
        tokenizer=tokenizer,
        label_map=label_map,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        slang_terms=slang_terms,
        slang_weight_map=slang_weight_map,
    )
    
    # Apply ablation mode if specified
    if args.ablation_mode != "full":
        ablation_configs = get_ablation_configs()
        base_config = ablation_configs[args.ablation_mode]
        args.use_mask = base_config.use_mask
        args.use_multiview = base_config.use_multiview
        args.use_logic = base_config.use_logic
    
    # Create model config
    model_config = ModelConfig(
        model_name=args.model_name,
        max_len=args.max_len,
        num_labels=len(label_map),
        proj_size=args.proj_size,
        dropout=args.dropout,
        tfidf_cache_path=args.tfidf_cache,
        tfidf_default_score=args.tfidf_default_score,
        use_mask=args.use_mask,
        use_multiview=args.use_multiview,
        use_logic=args.use_logic,
        mask_lambda=args.mask_lambda,
        drop_rules=[r.strip() for r in args.drop_rules.split(",") if r.strip()],
    )
    
    # Create training config
    train_config = TrainConfig(
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        encoder_lr=args.encoder_lr or args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        patience=args.patience,
        seed=args.seed,
        threshold=args.threshold,
        log_interval=args.log_interval,
        save_all_checkpoints=args.save_all_checkpoints,
        deep_supervision_lambda=args.deep_supervision_lambda,
        slang_preserve_lambda=args.slang_preserve_lambda,
        mask_sparsity_lambda=args.mask_sparsity_lambda,
        use_polarity_supervision=args.use_polarity_supervision,
        use_slang_preserve=args.use_slang_preserve,
        use_mask_sparsity=args.use_mask_sparsity,
        pos_labels=args.pos_labels,
        neg_labels=args.neg_labels,
        slang_lexicon_path=args.slang_lexicon,
    )
    
    # Create model
    print("\nCreating model...")
    model = SoftLogicViBERT(model_config).to(device)
    
    # Load pretrained weights if provided
    if args.pretrain_ckpt:
        print(f"  Loading pretrained weights from: {args.pretrain_ckpt}")
        pretrain_ckpt = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=False)
        pretrain_state = pretrain_ckpt.get("state_dict", pretrain_ckpt)
        # Load only keys with matching shapes to avoid hard failures on head changes.
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in pretrain_state.items():
            if key not in model_state:
                skipped.append((key, "missing"))
                continue
            if model_state[key].shape != value.shape:
                skipped.append((key, f"shape {tuple(value.shape)} -> {tuple(model_state[key].shape)}"))
                continue
            filtered_state[key] = value
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"  Loaded pretrained weights (missing: {len(missing)}, unexpected: {len(unexpected)}, skipped: {len(skipped)})")
        if skipped:
            print("  Skipped keys (first 5):")
            for key, reason in skipped[:5]:
                print(f"    - {key}: {reason}")
    
    # Print model summary
    summary = get_model_summary(model)
    print(f"  Total parameters: {summary['total']['total']:,}")
    print(f"  Trainable parameters: {summary['total']['trainable']:,}")
    
    # Compute class weights
    pos_weight = compute_pos_weight(train_rows, label_map).to(device)
    print(f"  Class weights: {pos_weight.tolist()}")
    
    # Create loss function
    loss_fn = SoftLogicLoss(
        loss_type=args.loss_type,
        mask_lambda=args.mask_lambda,
        pos_weight=pos_weight,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        deep_supervision_lambda=args.deep_supervision_lambda,
        slang_preserve_lambda=args.slang_preserve_lambda,
        mask_sparsity_lambda=args.mask_sparsity_lambda,
        use_polarity_supervision=args.use_polarity_supervision,
        use_slang_preserve=args.use_slang_preserve,
        use_mask_sparsity=args.use_mask_sparsity,
    )
    
    # Create optimizer with layer-wise learning rates
    encoder_params = list(model.encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    
    encoder_lr = args.encoder_lr or args.lr
    
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": encoder_lr},
        {"params": other_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    
    # Create scheduler
    total_steps = len(train_loader) * args.epochs // train_config.accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Create output directory
    if args.experiment_name:
        output_dir = os.path.join(args.output_dir, args.experiment_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        config_dict = {
            "model_config": model_config.__dict__,
            "train_config": train_config.__dict__,
            "args": vars(args),
        }
        json.dump(config_dict, f, indent=2, default=str)
    
    # Resolve polarity label indices
    pos_labels = [s.strip().lower() for s in args.pos_labels.split(",") if s.strip()]
    neg_labels = [s.strip().lower() for s in args.neg_labels.split(",") if s.strip()]
    pos_label_indices = [label_map[l] for l in pos_labels if l in label_map]
    neg_label_indices = [label_map[l] for l in neg_labels if l in label_map]

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=train_config,
        model_config=model_config,
        label_map=label_map,
        label_list=label_list,
        tokenizer=tokenizer,
        output_dir=output_dir,
        device=device,
        pos_label_indices=pos_label_indices,
        neg_label_indices=neg_label_indices,
        use_fp16=args.fp16,
    )
    
    # Train
    best_metrics = trainer.train()
    
    # Print final results
    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    for key, value in best_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
