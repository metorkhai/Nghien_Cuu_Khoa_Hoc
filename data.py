"""
Data loading and dataset classes for SoftLogic ViBERT.

Supports:
- JSON/JSONL file formats
- HuggingFace datasets
- Custom data formats with context/comment/labels
"""

import json
from typing import Dict, List, Optional, Tuple, Callable, Iterable
from pathlib import Path
import emoji
import re
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import (
    ensure_list,
    labels_to_multi_hot,
    extract_prag_features,
    normalize_vietnamese_text,
    build_label_map,
    stratified_split_multilabel,
)


# ============================================================================
# FILE READERS
# ============================================================================

def read_jsonl(path: str) -> List[Dict]:
    """Read JSONL file (one JSON object per line)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_json(path: str) -> List[Dict]:
    """Read JSON file (array or dict of objects)."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # If dict, assume values are the data items
        return list(obj.values())
    
    raise ValueError(f"Unsupported JSON format in {path}")


def load_data_file(path: str) -> List[Dict]:
    """Load data from file (auto-detect format)."""
    path = Path(path)
    
    if path.suffix == ".jsonl":
        return read_jsonl(str(path))
    elif path.suffix == ".json":
        # For .json files, check if it's actually JSONL format
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1).strip()
        
        if first_char == "[":
            # Standard JSON array
            return read_json(str(path))
        elif first_char == "{":
            # Could be JSON object or JSONL - try JSONL first
            try:
                return read_jsonl(str(path))
            except json.JSONDecodeError:
                return read_json(str(path))
        else:
            return read_json(str(path))
    else:
        # Unknown extension - try both formats
        try:
            return read_jsonl(str(path))
        except json.JSONDecodeError:
            return read_json(str(path))


def load_slang_lexicon(path: Optional[str]) -> Tuple[List[str], Dict[str, float]]:
    """
    Load slang lexicon from JSON.

    Supports:
    - Dict: slang -> normalized form (weights default to 1.0)
    - List: list of slang terms
    """
    if not path:
        return [], {}

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        terms = [str(k).strip().lower() for k in obj.keys() if str(k).strip()]
        return terms, {}

    if isinstance(obj, list):
        terms = [str(k).strip().lower() for k in obj if str(k).strip()]
        return terms, {}

    raise ValueError(f"Unsupported slang lexicon format in {path}")


def _build_slang_char_weights(
    text: str,
    slang_terms: Iterable[str],
    slang_weight_map: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Return per-character slang weights for a text."""
    if not text:
        return []

    weights = [0.0] * len(text)
    text_lower = text.lower()
    weight_map = slang_weight_map or {}

    for term in slang_terms:
        if not term:
            continue
        term_lower = term.lower()
        term_weight = float(weight_map.get(term_lower, 1.0))
        start = 0
        while True:
            idx = text_lower.find(term_lower, start)
            if idx == -1:
                break
            end = idx + len(term_lower)
            for i in range(idx, end):
                if term_weight > weights[i]:
                    weights[i] = term_weight
            start = idx + 1

    return weights


def load_hf_dataset(dataset_name: str, split: str) -> List[Dict]:
    """
    Load dataset from HuggingFace Hub.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'tridm/UIT-VSMEC')
        split: Dataset split ('train', 'validation', 'test')
    
    Returns:
        List of data dictionaries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is required for HuggingFace datasets. "
            "Install with: pip install datasets"
        )
    
    dataset = load_dataset(dataset_name, split=split)
    return [dict(item) for item in dataset]


def clean_text_stolen_from_friend(text):
    text = emoji.demojize(text)

    text = re.sub(r'(\w)\1+', r'\1', text)

    return text

# ============================================================================
# DATASET CLASS
# ============================================================================

class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis.
    
    Handles:
    - Comment + optional context encoding
    - Multi-label target encoding
    - Pragmatic feature extraction
    
    Args:
        rows: List of data dictionaries with 'comment', 'context', 'labels' keys
        tokenizer: HuggingFace tokenizer
        label_map: Mapping from label string to index
        max_len: Maximum sequence length
        normalize_text: Whether to apply text normalization
    """
    
    def __init__(
        self,
        rows: List[Dict],
        tokenizer,
        label_map: Dict[str, int],
        max_len: int = 128,
        normalize_text: bool = True,
        slang_terms: Optional[List[str]] = None,
        slang_weight_map: Optional[Dict[str, float]] = None,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len
        self.normalize_text = normalize_text
        self.num_labels = len(label_map)
        self.slang_terms = slang_terms or []
        self.slang_weight_map = slang_weight_map or {}
    
    def __len__(self) -> int:
        return len(self.rows)
    
    def _get_text(self, row: Dict) -> Tuple[str, Optional[str]]:
        """Extract and normalize comment and context."""

        comment = str(row.get("comment", row.get("text", "")))

        comment = clean_text_stolen_from_friend(comment)

        context = row.get("context", None)
        
        if self.normalize_text:
            comment = normalize_vietnamese_text(comment)
            if context:
                context = normalize_vietnamese_text(str(context))
        elif context:
            context = str(context)
        
        return comment, context
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        
        # Get text
        comment, context = self._get_text(row)
        
        # Get labels
        labels = ensure_list(row.get("labels", []))
        y = torch.tensor(
            labels_to_multi_hot(labels, self.label_map),
            dtype=torch.float,
        )
        
        # Tokenize
        tok = self.tokenizer(
            comment,
            text_pair=context,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        
        # Extract pragmatic features
        prag = torch.tensor(extract_prag_features(comment), dtype=torch.float)
        
        # Build output dict
        output = {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "labels": y,
            "prag_features": prag,
        }
        
        # Token type IDs (might not be present for some models)
        if "token_type_ids" in tok:
            output["token_type_ids"] = tok["token_type_ids"].squeeze(0)
        else:
            output["token_type_ids"] = None

        # Special tokens mask (1 for special tokens)
        if "special_tokens_mask" in tok:
            output["special_tokens_mask"] = tok["special_tokens_mask"].squeeze(0)
        else:
            output["special_tokens_mask"] = None

        # Slang mask and weights for mask supervision
        if self.slang_terms and "offset_mapping" in tok and hasattr(tok, "sequence_ids"):
            offsets = tok["offset_mapping"].squeeze(0).tolist()
            seq_ids = tok.sequence_ids(0)

            comment_weights = _build_slang_char_weights(
                comment,
                self.slang_terms,
                self.slang_weight_map,
            )
            context_weights = _build_slang_char_weights(
                context or "",
                self.slang_terms,
                self.slang_weight_map,
            )

            slang_mask = []
            slang_weights = []
            for (start, end), seq_id in zip(offsets, seq_ids):
                if start == end or seq_id is None:
                    slang_mask.append(0.0)
                    slang_weights.append(0.0)
                    continue

                if seq_id == 0:
                    weight_src = comment_weights
                else:
                    weight_src = context_weights

                if start < 0 or end > len(weight_src):
                    slang_mask.append(0.0)
                    slang_weights.append(0.0)
                    continue

                token_weight = max(weight_src[start:end]) if end > start else 0.0
                slang_mask.append(1.0 if token_weight > 0 else 0.0)
                slang_weights.append(token_weight)

            output["slang_mask"] = torch.tensor(slang_mask, dtype=torch.float)
            output["slang_weights"] = torch.tensor(slang_weights, dtype=torch.float)
        else:
            output["slang_mask"] = None
            output["slang_weights"] = None
        
        return output


class InferenceDataset(Dataset):
    """
    Lightweight dataset for inference (no labels required).
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_len: int = 128,
        contexts: Optional[List[str]] = None,
    ):
        self.texts = texts
        self.contexts = contexts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        comment = self.texts[idx]
        context = self.contexts[idx] if self.contexts else None
        
        tok = self.tokenizer(
            comment,
            text_pair=context,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        
        prag = torch.tensor(extract_prag_features(comment), dtype=torch.float)
        
        output = {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "prag_features": prag,
        }
        
        if "token_type_ids" in tok:
            output["token_type_ids"] = tok["token_type_ids"].squeeze(0)
        else:
            output["token_type_ids"] = None

        if "special_tokens_mask" in tok:
            output["special_tokens_mask"] = tok["special_tokens_mask"].squeeze(0)
        else:
            output["special_tokens_mask"] = None
        
        return output


# ============================================================================
# COLLATION
# ============================================================================

def collate_batch(batch: List[Dict]) -> Dict[str, Optional[torch.Tensor]]:
    """
    Collate function for DataLoader.
    
    Handles optional fields like token_type_ids and stacks tensors.
    """
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    
    # Handle token_type_ids (might be None for some models)
    token_type_list = [b["token_type_ids"] for b in batch]
    if any(t is None for t in token_type_list):
        token_type_ids = None
    else:
        token_type_ids = torch.stack(token_type_list)
    
    prag = torch.stack([b["prag_features"] for b in batch])
    
    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "prag_features": prag,
    }

    slang_mask_list = [b.get("slang_mask") for b in batch]
    if any(s is None for s in slang_mask_list):
        output["slang_mask"] = None
    else:
        output["slang_mask"] = torch.stack(slang_mask_list)

    slang_weight_list = [b.get("slang_weights") for b in batch]
    if any(s is None for s in slang_weight_list):
        output["slang_weights"] = None
    else:
        output["slang_weights"] = torch.stack(slang_weight_list)

    special_tokens_list = [b.get("special_tokens_mask") for b in batch]
    if any(s is None for s in special_tokens_list):
        output["special_tokens_mask"] = None
    else:
        output["special_tokens_mask"] = torch.stack(special_tokens_list)
    
    # Labels might not be present for inference
    if "labels" in batch[0]:
        output["labels"] = torch.stack([b["labels"] for b in batch])
    
    return output


# ============================================================================
# DATA PREPARATION
# ============================================================================

def oversample_minority_rows(
    train_rows: List[Dict],
    minority_labels: Optional[List[str]] = None,
) -> List[Dict]:
    """Duplicate minority class rows once for simple random oversampling."""
    if not train_rows:
        return train_rows

    minority = {s.lower() for s in (minority_labels or ["surprise", "fear", "disgust"])}
    extra = []
    for row in train_rows:
        labels = ensure_list(row.get("labels", []))
        if any(lab.lower() in minority for lab in labels):
            extra.append(row)
    return train_rows + extra

def prepare_dataloaders(
    train_rows: List[Dict],
    val_rows: List[Dict],
    tokenizer,
    label_map: Dict[str, int],
    max_len: int = 128,
    batch_size: int = 16,
    num_workers: int = 0,
    slang_terms: Optional[List[str]] = None,
    slang_weight_map: Optional[Dict[str, float]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation DataLoaders.
    
    Args:
        train_rows: Training data
        val_rows: Validation data
        tokenizer: HuggingFace tokenizer
        label_map: Label to index mapping
        max_len: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader
    """
    train_ds = SentimentDataset(
        train_rows,
        tokenizer,
        label_map,
        max_len,
        slang_terms=slang_terms,
        slang_weight_map=slang_weight_map,
    )
    val_ds = SentimentDataset(
        val_rows,
        tokenizer,
        label_map,
        max_len,
        slang_terms=slang_terms,
        slang_weight_map=slang_weight_map,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def load_and_prepare_data(
    data_path: Optional[str] = None,
    val_path: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_split: str = "train",
    hf_val_split: Optional[str] = "validation",
    val_ratio: float = 0.1,
    seed: int = 42,
    minority_labels: Optional[List[str]] = None,
    return_stats: bool = False,
) -> Tuple[List[Dict], List[Dict], Dict[str, int], List[str]]:
    """
    Load data and prepare train/val splits with label mapping.
    
    Args:
        data_path: Path to local data file
        val_path: Path to validation data file (optional)
        hf_dataset: HuggingFace dataset name (optional)
        hf_split: HuggingFace train split name
        hf_val_split: HuggingFace validation split name
        val_ratio: Validation split ratio (if no val_path)
        seed: Random seed for splitting
    
    Returns:
        train_rows, val_rows, label_map, label_list
    """
    # Load data
    if hf_dataset:
        train_rows = load_hf_dataset(hf_dataset, hf_split)
        if hf_val_split:
            try:
                val_rows = load_hf_dataset(hf_dataset, hf_val_split)
            except Exception:
                train_rows, val_rows = stratified_split_multilabel(
                    train_rows, val_ratio, seed
                )
        else:
            train_rows, val_rows = stratified_split_multilabel(
                train_rows, val_ratio, seed
            )
    else:
        if data_path is None:
            raise ValueError("Either data_path or hf_dataset must be provided")
        
        train_rows = load_data_file(data_path)
        
        if val_path:
            val_rows = load_data_file(val_path)
        else:
            train_rows, val_rows = stratified_split_multilabel(
                train_rows, val_ratio, seed
            )
    
    # Pre-oversampling stats
    pre_labels = [ensure_list(r.get("labels", [])) for r in train_rows]
    pre_label_map, _ = build_label_map(pre_labels)
    pre_stats = analyze_label_distribution(train_rows, pre_label_map)

    # Apply simple random oversampling for minority classes
    train_rows = oversample_minority_rows(train_rows, minority_labels)

    # Build label map from training data
    all_labels = [ensure_list(r.get("labels", [])) for r in train_rows]
    label_map, label_list = build_label_map(all_labels)

    post_stats = analyze_label_distribution(train_rows, label_map)

    if return_stats:
        return train_rows, val_rows, label_map, label_list, pre_stats, post_stats

    return train_rows, val_rows, label_map, label_list


# ============================================================================
# DATA ANALYSIS
# ============================================================================

def analyze_label_distribution(
    rows: List[Dict],
    label_map: Dict[str, int],
) -> Dict[str, Dict]:
    """
    Analyze label distribution in dataset.
    
    Returns statistics about label frequencies and co-occurrences.
    """
    num_labels = len(label_map)
    label_counts = {label: 0 for label in label_map}
    co_occurrence = torch.zeros(num_labels, num_labels)
    labels_per_sample = []
    
    for row in rows:
        labels = ensure_list(row.get("labels", []))
        labels_per_sample.append(len(labels))
        
        indices = []
        for lab in labels:
            if lab.lower() in label_map:
                label_counts[lab.lower()] += 1
                indices.append(label_map[lab.lower()])
        
        # Co-occurrence
        for i in indices:
            for j in indices:
                co_occurrence[i, j] += 1
    
    # Compute statistics
    total = len(rows)
    label_freq = {
        label: count / total
        for label, count in label_counts.items()
    }
    
    avg_labels = sum(labels_per_sample) / len(labels_per_sample) if labels_per_sample else 0
    
    return {
        "total_samples": total,
        "label_counts": label_counts,
        "label_frequencies": label_freq,
        "avg_labels_per_sample": avg_labels,
        "co_occurrence_matrix": co_occurrence.tolist(),
    }
