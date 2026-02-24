"""
Comprehensive metrics for multi-label sentiment classification.

Includes:
- Per-class and aggregate F1 scores
- Precision, recall, and accuracy metrics
- Label distribution analysis
- Calibration metrics
- Confusion matrix utilities
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe division with epsilon to prevent NaN."""
    return num / (den + eps)


# ============================================================================
# CORE METRICS
# ============================================================================

def multilabel_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive multi-label classification metrics.
    
    Args:
        logits: Model logits [N, C]
        labels: Ground truth labels [N, C]
        threshold: Classification threshold
        label_names: Optional list of label names for per-class metrics
    
    Returns:
        Dictionary with all metrics
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    
    # Per-class metrics
    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)
    tn = ((1 - preds) * (1 - labels)).sum(dim=0)
    
    # Precision, Recall, F1 per class
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    
    # Macro averages
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()
    
    # Micro averages
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    
    micro_precision = _safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    
    # Weighted F1 (weighted by support)
    support = labels.sum(dim=0)
    weighted_f1 = _safe_div((f1 * support).sum(), support.sum())
    
    # Subset accuracy (exact match)
    exact_match = (preds == labels).all(dim=1).float().mean()
    
    # Hamming loss
    hamming_loss = (preds != labels).float().mean()
    
    # Sample-based metrics
    sample_f1s = []
    for i in range(logits.size(0)):
        s_tp = (preds[i] * labels[i]).sum()
        s_fp = (preds[i] * (1 - labels[i])).sum()
        s_fn = ((1 - preds[i]) * labels[i]).sum()
        s_p = _safe_div(s_tp, s_tp + s_fp)
        s_r = _safe_div(s_tp, s_tp + s_fn)
        s_f1 = _safe_div(2 * s_p * s_r, s_p + s_r)
        sample_f1s.append(s_f1.item())
    sample_f1 = np.mean(sample_f1s)
    
    metrics = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1.item(),
        "weighted_f1": weighted_f1.item(),
        "sample_f1": sample_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_precision": micro_precision.item(),
        "micro_recall": micro_recall.item(),
        "exact_match_accuracy": exact_match.item(),
        "hamming_loss": hamming_loss.item(),
    }
    
    # Per-class metrics
    if label_names:
        for i, name in enumerate(label_names):
            metrics[f"{name}_f1"] = f1[i].item()
            metrics[f"{name}_precision"] = precision[i].item()
            metrics[f"{name}_recall"] = recall[i].item()
            metrics[f"{name}_support"] = support[i].item()
    
    return metrics


def multilabel_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute macro and micro F1 scores.
    
    Simplified version for training loop evaluation.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    
    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)
    
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    
    macro_f1 = f1.mean().item()
    
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    
    micro_precision = _safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1.item(),
    }


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def find_optimal_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: Optional[List[float]] = None,
    optimize_for: str = "macro_f1",
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        logits: Model logits [N, C]
        labels: Ground truth labels [N, C]
        thresholds: List of thresholds to try
        optimize_for: Metric to optimize ('macro_f1' or 'micro_f1')
    
    Returns:
        (best_threshold, best_score)
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in thresholds:
        metrics = multilabel_f1(logits, labels, thresh)
        score = metrics[optimize_for]
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def find_per_class_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: Optional[List[float]] = None,
) -> Tuple[List[float], float]:
    """
    Find optimal threshold for each class independently.
    
    Returns:
        (per_class_thresholds, combined_macro_f1)
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    num_classes = logits.size(1)
    probs = torch.sigmoid(logits)
    
    per_class_thresholds = []
    per_class_f1s = []
    
    for c in range(num_classes):
        best_thresh = 0.5
        best_f1 = 0.0
        
        for thresh in thresholds:
            preds = (probs[:, c] >= thresh).float()
            y_true = labels[:, c]
            
            tp = (preds * y_true).sum()
            fp = (preds * (1 - y_true)).sum()
            fn = ((1 - preds) * y_true).sum()
            
            p = _safe_div(tp, tp + fp)
            r = _safe_div(tp, tp + fn)
            f1 = _safe_div(2 * p * r, p + r).item()
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        per_class_thresholds.append(best_thresh)
        per_class_f1s.append(best_f1)
    
    combined_macro_f1 = np.mean(per_class_f1s)
    
    return per_class_thresholds, combined_macro_f1


# ============================================================================
# CALIBRATION METRICS
# ============================================================================

def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for multi-label classification.
    
    Measures how well predicted probabilities match actual outcomes.
    """
    probs = torch.sigmoid(logits).flatten()
    targets = labels.flatten()
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probs)
    
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        bin_size = in_bin.sum().item()
        
        if bin_size > 0:
            avg_confidence = probs[in_bin].mean().item()
            avg_accuracy = targets[in_bin].mean().item()
            ece += abs(avg_accuracy - avg_confidence) * (bin_size / total_samples)
    
    return ece


def compute_brier_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Brier score for probability calibration.
    
    Lower is better. Perfect predictions = 0.
    """
    probs = torch.sigmoid(logits)
    return ((probs - labels) ** 2).mean().item()


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def multilabel_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute confusion matrix for each label in multi-label setting.
    
    Returns dict with TP, FP, TN, FN for each class.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    
    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)
    tn = ((1 - preds) * (1 - labels)).sum(dim=0)
    
    num_classes = logits.size(1)
    
    if label_names is None:
        label_names = [f"class_{i}" for i in range(num_classes)]
    
    confusion = {}
    for i, name in enumerate(label_names):
        confusion[name] = {
            "TP": int(tp[i].item()),
            "FP": int(fp[i].item()),
            "FN": int(fn[i].item()),
            "TN": int(tn[i].item()),
        }
    
    return confusion


# ============================================================================
# LABEL DISTRIBUTION METRICS
# ============================================================================

def compute_label_correlation(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute label co-occurrence correlation matrix.
    
    Returns [C, C] correlation matrix.
    """
    # Center labels
    labels_centered = labels - labels.mean(dim=0, keepdim=True)
    
    # Compute covariance
    cov = torch.mm(labels_centered.t(), labels_centered) / labels.size(0)
    
    # Compute correlation
    std = labels.std(dim=0, keepdim=True).t()
    std_matrix = torch.mm(std, std.t())
    correlation = cov / (std_matrix + 1e-8)
    
    return correlation


def detect_label_imbalance(
    labels: torch.Tensor,
    imbalance_threshold: float = 0.2,
) -> Dict[str, Dict]:
    """
    Detect class imbalance in multi-label data.
    
    Returns statistics about label distribution.
    """
    label_freqs = labels.mean(dim=0)
    
    minority_mask = label_freqs < imbalance_threshold
    majority_mask = label_freqs > (1 - imbalance_threshold)
    
    return {
        "label_frequencies": label_freqs.tolist(),
        "minority_classes": minority_mask.nonzero(as_tuple=True)[0].tolist(),
        "majority_classes": majority_mask.nonzero(as_tuple=True)[0].tolist(),
        "imbalance_ratio": (label_freqs.max() / (label_freqs.min() + 1e-8)).item(),
    }


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_model(
    model,
    dataloader,
    device: torch.device,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
    special_token_ids: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Computes all metrics including:
    - Classification metrics
    - Calibration metrics
    - Rule activation statistics (if model has soft logic)
    - Mask statistics (if model has masking)
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    rule_activations = {f"r{i}": [] for i in range(1, 6)}
    mask_means = []
    predicate_values = {
        "p_pos_sem": [],
        "p_neg_sem": [],
        "p_pos_lex": [],
        "p_neg_lex": [],
        "p_high_int": [],
    }
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)
            prag = batch.get("prag_features")
            if prag is not None:
                prag = prag.to(device)
            
            logits, extras = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                prag_features=prag,
                special_token_ids=special_token_ids,
                return_extras=True,
            )
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
            # Collect extras
            if "rules" in extras and extras["rules"]:
                for k, v in extras["rules"].items():
                    if k in rule_activations:
                        rule_activations[k].append(v.mean().cpu())
            
            if "mask_mean" in extras:
                mask_means.append(extras["mask_mean"].cpu())
            
            if "predicates" in extras and extras["predicates"]:
                for k, v in extras["predicates"].items():
                    if k in predicate_values:
                        predicate_values[k].append(v.mean().cpu())
    
    # Concatenate all predictions
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = multilabel_metrics(logits, labels, threshold, label_names)
    
    # Add calibration metrics
    metrics["ece"] = compute_ece(logits, labels)
    metrics["brier_score"] = compute_brier_score(logits, labels)
    
    # Add rule statistics
    if any(rule_activations[k] for k in rule_activations):
        metrics["rule_means"] = {
            k: torch.stack(v).mean().item() if v else 0.0
            for k, v in rule_activations.items()
        }
    
    # Add mask statistics
    if mask_means:
        metrics["mask_mean"] = torch.stack(mask_means).mean().item()
    
    # Add predicate statistics
    if any(predicate_values[k] for k in predicate_values):
        metrics["predicate_means"] = {
            k: torch.stack(v).mean().item() if v else 0.0
            for k, v in predicate_values.items()
        }
    
    return metrics
