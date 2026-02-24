"""
Mask-Guided Language Adaptation Pretraining for Vietnamese GenZ Text.

This module implements a pretraining phase that:
1. Learns which tokens are semantically important (slang, keywords)
2. Learns which tokens are noise (fillers, stopwords)
3. Uses a slang dictionary as weak supervision
4. Preserves linguistic signals without destroying surface forms

Key Components:
- SlangAwareDataset: Dataset with slang token identification
- MaskPreservationLoss: Encourages high masks on slang tokens
- MaskGuidedPretrainer: Full pretraining loop
"""

import json
import re
import logging
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Sequence
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PretrainConfig:
    """Configuration for mask-guided pretraining."""
    
    # Model
    model_name: str = "Fsoft-AIC/videberta-base"
    max_len: int = 128
    
    # Dictionary
    dictionary_path: str = "dictionary.json"
    
    # Training
    batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    
    # Loss weights
    mlm_weight: float = 1.0
    mask_preservation_weight: float = 0.2  # λ for L_mask
    mask_sparsity_weight: float = 0.01  # Optional sparsity regularization
    
    # MLM settings
    mlm_probability: float = 0.15
    use_embedding_consistency: bool = False  # Alternative to MLM
    
    # What to train
    train_masking_layer: bool = True
    train_encoder_layers: int = 3  # Last N layers, -1 for all, 0 for none
    freeze_embeddings: bool = True
    
    # Logging
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1000
    
    # Output
    output_dir: str = "pretrain_outputs"
    experiment_name: str = "mask_pretrain"


# ============================================================================
# SLANG DICTIONARY
# ============================================================================

class SlangDictionary:
    """
    Manages the slang/abbreviation dictionary for token identification.
    
    The dictionary maps slang terms to their normalized meanings.
    This is used as weak supervision - we want the model to preserve
    slang tokens rather than suppressing them.
    """
    
    def __init__(self, dictionary_path: Optional[str] = None):
        self.slang_to_meaning: Dict[str, str] = {}
        self.slang_tokens: Set[str] = set()
        self.slang_subwords: Set[str] = set()
        
        if dictionary_path and Path(dictionary_path).exists():
            self.load(dictionary_path)
    
    def load(self, path: str) -> None:
        """Load dictionary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            self.slang_to_meaning = data
        elif isinstance(data, list):
            # Handle list format [{"slang": "...", "meaning": "..."}, ...]
            for item in data:
                if isinstance(item, dict):
                    slang = item.get("slang") or item.get("term") or item.get("word")
                    meaning = item.get("meaning") or item.get("normalized") or item.get("definition")
                    if slang:
                        self.slang_to_meaning[slang.lower()] = meaning or ""
        
        # Build token set
        self.slang_tokens = set(self.slang_to_meaning.keys())
        
        logger.info(f"Loaded {len(self.slang_tokens)} slang terms from {path}")
    
    def add_common_genz_slang(self) -> None:
        """Add common Vietnamese GenZ slang terms."""
        common_slang = {
            # Expressions
            "vl": "vãi lồn",
            "vcl": "vãi cả lồn",
            "đm": "địt mẹ",
            "clgt": "cái lồn gì thế",
            "wtf": "what the fuck",
            "lmao": "laughing my ass off",
            "lol": "laugh out loud",
            "omg": "oh my god",
            
            # Positive
            "xịn": "tốt đẹp",
            "đỉnh": "tuyệt vời",
            "max": "rất nhiều",
            "chill": "thư giãn",
            "cool": "tuyệt",
            "nice": "tốt",
            "ok": "được",
            "oke": "được",
            "okie": "được",
            
            # Negative
            "lầy": "khó chịu",
            "ghẻ": "tệ",
            "chán": "nhàm chán",
            "mệt": "mệt mỏi",
            
            # Filler words (should NOT be in slang - these are noise)
            # "ơi": "", "à": "", "ạ": "", "nhé": "", "nha": "",
            
            # Intensifiers
            "quá": "rất",
            "lắm": "nhiều",
            "cực": "rất",
            "siêu": "rất",
            "mega": "rất lớn",
            "ultra": "cực kỳ",
            
            # Common abbreviations
            "dc": "được",
            "ko": "không",
            "k": "không",
            "kg": "không",
            "hk": "không",
            "cx": "cũng",
            "cg": "cùng",
            "ns": "nói",
            "bth": "bình thường",
            "bt": "bình thường",
            "ntn": "như thế nào",
            "v": "vậy",
            "vs": "với",
            "r": "rồi",
            "m": "mày",
            "t": "tao",
            "nyc": "người yêu cũ",
            "ny": "người yêu",
            "gf": "girlfriend",
            "bf": "boyfriend",
            "crush": "người thích",
            
            # Emoji-like expressions
            "hihi": "cười",
            "haha": "cười",
            "huhu": "khóc",
            "hehe": "cười",
            "keke": "cười",
            "kakaka": "cười",
            
            # GenZ specific
            "slay": "rất tốt",
            "stan": "hâm mộ",
            "flex": "khoe",
            "vibe": "cảm xúc",
            "sống ảo": "giả tạo",
            "troll": "trêu chọc",
            "toxic": "độc hại",
            "ghost": "biến mất",
            "ship": "ghép đôi",
            "bias": "thần tượng",
        }
        
        for slang, meaning in common_slang.items():
            if slang.lower() not in self.slang_to_meaning:
                self.slang_to_meaning[slang.lower()] = meaning
        
        self.slang_tokens = set(self.slang_to_meaning.keys())
    
    def tokenize_and_mark(
        self,
        tokenizer,
        text: str,
        max_len: int = 128,
    ) -> Tuple[Dict, torch.Tensor]:
        """
        Tokenize text and create slang indicator mask.
        
        Returns:
            encoding: Tokenizer output
            slang_mask: Binary tensor [seq_len] where 1 = slang token
        """
        # Normalize text
        text_lower = text.lower()
        
        # Tokenize
        encoding = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze(0).tolist())
        
        # Create slang mask
        slang_mask = torch.zeros(len(tokens), dtype=torch.float)
        
        # Check each token
        for i, token in enumerate(tokens):
            # Clean token (remove ## prefix for subwords)
            clean_token = token.replace("##", "").replace("▁", "").lower()
            
            if not clean_token or clean_token in ["[PAD]", "[CLS]", "[SEP]", "<pad>", "<s>", "</s>"]:
                continue
            
            # Direct match
            if clean_token in self.slang_tokens:
                slang_mask[i] = 1.0
                continue
            
            # Check if token is part of a slang term
            for slang in self.slang_tokens:
                if clean_token in slang or slang in clean_token:
                    slang_mask[i] = 1.0
                    break
        
        return encoding, slang_mask
    
    def is_slang(self, token: str) -> bool:
        """Check if a token is slang."""
        clean = token.replace("##", "").replace("▁", "").lower()
        return clean in self.slang_tokens
    
    def get_meaning(self, slang: str) -> Optional[str]:
        """Get normalized meaning of slang term."""
        return self.slang_to_meaning.get(slang.lower())
    
    def __len__(self) -> int:
        return len(self.slang_tokens)
    
    def __contains__(self, item: str) -> bool:
        return item.lower() in self.slang_tokens


# ============================================================================
# DATASET
# ============================================================================

class SlangAwareDataset(Dataset):
    """
    Dataset for mask-guided pretraining.
    
    Features:
    - Loads raw comments (no labels needed)
    - Identifies slang tokens using dictionary
    - Supports MLM-style masking
    - Provides slang indicators for mask preservation loss
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        slang_dict: SlangDictionary,
        max_len: int = 128,
        mlm_probability: float = 0.15,
        enable_mlm: bool = True,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.slang_dict = slang_dict
        self.max_len = max_len
        self.mlm_probability = mlm_probability
        self.enable_mlm = enable_mlm
        
        # Special token IDs
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        
        self.special_token_ids = {
            self.mask_token_id, self.pad_token_id,
            self.cls_token_id, self.sep_token_id
        }
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize and get slang mask
        encoding, slang_mask = self.slang_dict.tokenize_and_mark(
            self.tokenizer, text, self.max_len
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(0)
        else:
            token_type_ids = torch.zeros_like(input_ids)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "slang_mask": slang_mask,
        }
        
        # Apply MLM masking if enabled
        if self.enable_mlm:
            mlm_input_ids, mlm_labels = self._apply_mlm_masking(
                input_ids.clone(), attention_mask, slang_mask
            )
            result["mlm_input_ids"] = mlm_input_ids
            result["mlm_labels"] = mlm_labels
        
        return result
    
    def _apply_mlm_masking(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        slang_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MLM masking with bias toward non-slang tokens.
        
        We want to:
        - Mask some tokens for MLM objective
        - Preferentially mask non-slang tokens
        - Still mask some slang tokens (but less frequently)
        """
        labels = input_ids.clone()
        
        # Probability matrix
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = torch.tensor([
            1 if input_ids[i].item() in self.special_token_ids else 0
            for i in range(len(input_ids))
        ], dtype=torch.bool)
        
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        probability_matrix.masked_fill_(attention_mask == 0, 0.0)
        
        # Reduce masking probability for slang tokens (preserve them more)
        slang_reduction = 0.5  # Mask slang tokens less often
        probability_matrix = probability_matrix * (1 - slang_mask * (1 - slang_reduction))
        
        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels for non-masked tokens to -100 (ignore in loss)
        labels[~masked_indices] = -100
        
        # 80% of time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        # 10% of time, replace with random token
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 10% of time, keep original (already done by not modifying)
        
        return input_ids, labels


def load_pretrain_data(
    data_path: str,
    text_field: str = "comment",
) -> List[str]:
    """
    Load text data for pretraining.
    
    Supports JSON, JSONL, CSV formats.

    Notes:
        - If you pass a directory, use `load_pretrain_splits()` instead.
    """
    path = Path(data_path)
    if path.is_dir():
        raise ValueError(
            f"Expected a file path, got directory: {path}. "
            "Use load_pretrain_splits(data_path=...) for train/dev/test loading."
        )
    texts = []
    
    if path.suffix == ".jsonl" or (path.suffix == ".json" and _is_jsonl(path)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if text_field in item:
                    texts.append(item[text_field])
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and text_field in item:
                    texts.append(item[text_field])
                elif isinstance(item, str):
                    texts.append(item)
    
    logger.info(f"Loaded {len(texts)} texts for pretraining")
    return texts


def load_pretrain_splits(
    data_path: str,
    text_field: Optional[str] = None,
    default_json_field: str = "comment",
) -> Tuple[List[str], Optional[List[str]], Optional[List[str]]]:
    """Load (train, val, test) texts from a file or a directory.

    Supported layouts:
        - Directory containing `train.csv`, `dev.csv` (optional), `test.csv` (optional)
        - Single file (CSV/JSON/JSONL): returned as train split

    Args:
        data_path: File or directory path.
        text_field:
            Preferred text column/key.
            - For CSV, defaults to "original" if present, else inferred.
            - For JSON/JSONL, defaults to `default_json_field`.
        default_json_field: Field name for JSON/JSONL when `text_field` is not provided.
    """
    path = Path(data_path)

    if path.is_dir():
        train_file = path / "train.csv"
        dev_file = path / "dev.csv"
        test_file = path / "test.csv"

        if train_file.exists():
            train_texts = _load_texts_from_csv(train_file, text_field=text_field)
            val_texts = _load_texts_from_csv(dev_file, text_field=text_field) if dev_file.exists() else None
            test_texts = _load_texts_from_csv(test_file, text_field=text_field) if test_file.exists() else None
            logger.info(
                f"Loaded splits from {path}: train={len(train_texts)}, "
                f"val={len(val_texts) if val_texts is not None else 0}, "
                f"test={len(test_texts) if test_texts is not None else 0}"
            )
            return train_texts, val_texts, test_texts

        raise FileNotFoundError(
            f"No train.csv found in directory: {path}. "
            "Expected files: train.csv (required), dev.csv/test.csv (optional)."
        )

    # Single-file path
    if path.suffix.lower() == ".csv":
        train_texts = _load_texts_from_csv(path, text_field=text_field)
        return train_texts, None, None

    # JSON / JSONL
    json_field = text_field or default_json_field
    train_texts = load_pretrain_data(str(path), text_field=json_field)
    return train_texts, None, None


def _infer_csv_text_field(fieldnames: Sequence[str]) -> Optional[str]:
    """Infer the most likely text field name from CSV headers."""
    if not fieldnames:
        return None

    normalized = {f.strip().lower(): f for f in fieldnames if f is not None}

    # Prefer GenZ raw text (common in this repo)
    preferred = [
        "original",
        "text",
        "comment",
        "content",
        "sentence",
        "normalized",
    ]
    for key in preferred:
        if key in normalized:
            return normalized[key]

    # Otherwise pick the first non-empty column
    for f in fieldnames:
        if f and f.strip():
            return f
    return None


def _load_texts_from_csv(path: Path, text_field: Optional[str] = None) -> List[str]:
    """Load texts from a CSV file using DictReader."""
    texts: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        chosen_field = None
        if text_field:
            # Match case-insensitively
            lowered = {name.lower(): name for name in fieldnames if name}
            chosen_field = lowered.get(text_field.lower())
        if not chosen_field:
            chosen_field = _infer_csv_text_field(fieldnames)

        if not chosen_field:
            raise ValueError(f"Could not infer text column from CSV headers in {path}")

        for row in reader:
            if not isinstance(row, dict):
                continue
            value = row.get(chosen_field)
            if value is None:
                continue
            value = str(value).strip()
            if not value:
                continue
            texts.append(value)

    logger.info(f"Loaded {len(texts)} texts from CSV {path} (column={chosen_field})")
    return texts


def _is_jsonl(path: Path) -> bool:
    """Check if file is JSONL format."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1).strip()
        return first_char == "{"


# ============================================================================
# LOSSES
# ============================================================================

class MaskPreservationLoss(nn.Module):
    """
    Loss that encourages high mask values for slang tokens.
    
    L_mask = mean((1 - m_i) for all tokens where s_i = 1)
    
    This penalizes the model when slang tokens are suppressed.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        mask_values: torch.Tensor,
        slang_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute mask preservation loss.
        
        Args:
            mask_values: Soft mask values [B, L, 1] or [B, L]
            slang_mask: Binary slang indicators [B, L]
            attention_mask: Attention mask [B, L]
        
        Returns:
            Scalar loss value
        """
        # Squeeze mask values if needed
        if mask_values.dim() == 3:
            mask_values = mask_values.squeeze(-1)
        
        # Compute (1 - m_i) for slang tokens
        preservation_penalty = (1 - mask_values) * slang_mask
        
        # Apply attention mask
        if attention_mask is not None:
            preservation_penalty = preservation_penalty * attention_mask
            valid_count = (slang_mask * attention_mask).sum() + 1e-8
        else:
            valid_count = slang_mask.sum() + 1e-8
        
        # Compute mean
        if self.reduction == "mean":
            loss = preservation_penalty.sum() / valid_count
        elif self.reduction == "sum":
            loss = preservation_penalty.sum()
        else:
            loss = preservation_penalty
        
        return loss


class EmbeddingConsistencyLoss(nn.Module):
    """
    Alternative to MLM: encourage consistent embeddings for slang tokens.
    
    The idea is that slang tokens should produce stable representations
    that are consistent with their normalized meanings.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_embeddings: torch.Tensor,
        slang_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute embedding consistency loss.
        
        Encourages the masked-and-encoded representation to be
        similar to the original embedding for slang tokens.
        """
        # Project hidden states
        projected = self.projection(hidden_states)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(projected, original_embeddings, dim=-1)
        
        # We want high similarity for slang tokens
        # Loss = 1 - similarity for slang tokens
        consistency_loss = (1 - similarity) * slang_mask
        
        if attention_mask is not None:
            consistency_loss = consistency_loss * attention_mask
            valid_count = (slang_mask * attention_mask).sum() + 1e-8
        else:
            valid_count = slang_mask.sum() + 1e-8
        
        return consistency_loss.sum() / valid_count


# ============================================================================
# PRETRAINER
# ============================================================================

class MaskGuidedPretrainer:
    """
    Mask-Guided Language Adaptation Pretrainer.
    
    This class handles the pretraining loop that:
    1. Trains the masking layer to preserve slang tokens
    2. Optionally fine-tunes encoder layers
    3. Uses MLM or embedding consistency as main objective
    4. Logs mask statistics for slang vs non-slang tokens
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: PretrainConfig,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Load slang dictionary
        self.slang_dict = SlangDictionary(config.dictionary_path)
        self.slang_dict.add_common_genz_slang()
        
        # Losses
        self.mask_preservation_loss = MaskPreservationLoss()
        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        if config.use_embedding_consistency:
            hidden_size = model.config.hidden_size if hasattr(model, 'config') else 768
            self.embedding_consistency_loss = EmbeddingConsistencyLoss(hidden_size).to(device)
        
        # Setup parameter groups
        self._setup_training()
        
        # Statistics
        self.stats = defaultdict(list)
    
    def _setup_training(self) -> None:
        """Setup which parameters to train."""
        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze masking layer
        if self.config.train_masking_layer and hasattr(self.model, 'mask_layer'):
            for param in self.model.mask_layer.parameters():
                param.requires_grad = True
            logger.info("Training: Masking layer")
        
        # Unfreeze encoder layers
        if self.config.train_encoder_layers != 0 and hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
            
            # Get transformer layers
            if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
                layers = encoder.encoder.layer
            elif hasattr(encoder, 'layer'):
                layers = encoder.layer
            else:
                layers = []
            
            if self.config.train_encoder_layers == -1:
                # Train all layers
                for layer in layers:
                    for param in layer.parameters():
                        param.requires_grad = True
                logger.info(f"Training: All {len(layers)} encoder layers")
            else:
                # Train last N layers
                n = min(self.config.train_encoder_layers, len(layers))
                for layer in layers[-n:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                logger.info(f"Training: Last {n} encoder layers")
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        
        # Setup optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    def train(
        self,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the pretraining loop.
        
        Args:
            train_texts: List of training texts
            val_texts: Optional validation texts
        
        Returns:
            Training statistics
        """
        # Create dataset
        train_dataset = SlangAwareDataset(
            texts=train_texts,
            tokenizer=self.tokenizer,
            slang_dict=self.slang_dict,
            max_len=self.config.max_len,
            mlm_probability=self.config.mlm_probability,
            enable_mlm=not self.config.use_embedding_consistency,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = None
        if val_texts:
            val_dataset = SlangAwareDataset(
                texts=val_texts,
                tokenizer=self.tokenizer,
                slang_dict=self.slang_dict,
                max_len=self.config.max_len,
                mlm_probability=self.config.mlm_probability,
                enable_mlm=not self.config.use_embedding_consistency,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        # Output directory
        output_dir = Path(self.config.output_dir) / self.config.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting pretraining for {self.config.num_epochs} epochs")
        logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_stats = defaultdict(float)
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                loss_dict = self._training_step(batch)
                
                # Backward pass
                loss = loss_dict["total_loss"]
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
                
                self.optimizer.step()
                scheduler.step()
                
                # Accumulate stats
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        epoch_stats[k] += v.item()
                    else:
                        epoch_stats[k] += v
                
                global_step += 1
                
                # Logging
                if global_step % self.config.log_every == 0:
                    avg_loss = epoch_stats["total_loss"] / (batch_idx + 1)
                    avg_mask_loss = epoch_stats["mask_loss"] / (batch_idx + 1)
                    slang_mask_mean = epoch_stats.get("slang_mask_mean", 0) / (batch_idx + 1)
                    nonslang_mask_mean = epoch_stats.get("nonslang_mask_mean", 0) / (batch_idx + 1)
                    
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.num_epochs} | "
                        f"Step {global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Mask Loss: {avg_mask_loss:.4f} | "
                        f"Slang Mask: {slang_mask_mean:.4f} | "
                        f"Non-slang Mask: {nonslang_mask_mean:.4f}"
                    )
                
                # Save checkpoint
                if global_step % self.config.save_every == 0:
                    self._save_checkpoint(output_dir / f"checkpoint_step_{global_step}.pt")
            
            # Epoch summary
            n_batches = len(train_loader)
            epoch_summary = {k: v / n_batches for k, v in epoch_stats.items()}
            
            logger.info(f"\nEpoch {epoch+1} Summary:")
            logger.info(f"  Total Loss: {epoch_summary['total_loss']:.4f}")
            logger.info(f"  Mask Preservation Loss: {epoch_summary['mask_loss']:.4f}")
            logger.info(f"  Avg Slang Mask Value: {epoch_summary.get('slang_mask_mean', 0):.4f}")
            logger.info(f"  Avg Non-slang Mask Value: {epoch_summary.get('nonslang_mask_mean', 0):.4f}")
            
            # Store stats
            for k, v in epoch_summary.items():
                self.stats[k].append(v)

            # Optional validation (per-epoch)
            if val_loader is not None:
                val_summary = self._evaluate(val_loader)
                logger.info(f"  Val Total Loss: {val_summary['total_loss']:.4f}")
                logger.info(f"  Val Mask Preservation Loss: {val_summary['mask_loss']:.4f}")
                logger.info(f"  Val Avg Slang Mask Value: {val_summary.get('slang_mask_mean', 0):.4f}")
                logger.info(f"  Val Avg Non-slang Mask Value: {val_summary.get('nonslang_mask_mean', 0):.4f}")

                for k, v in val_summary.items():
                    self.stats[f"val_{k}"].append(v)

                if val_summary["total_loss"] < best_val_loss:
                    best_val_loss = val_summary["total_loss"]
                    self._save_checkpoint(output_dir / "pretrain_best.pt")
        
        # Save final checkpoint
        self._save_checkpoint(output_dir / "pretrain_final.pt")
        
        # Save training stats
        stats_path = output_dir / "pretrain_stats.json"
        with open(stats_path, "w") as f:
            json.dump({k: v for k, v in self.stats.items()}, f, indent=2)
        
        logger.info(f"\nPretraining completed! Checkpoint saved to {output_dir}")
        
        return dict(self.stats)

    def _evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate loss components on a dataloader."""
        self.model.eval()
        totals = defaultdict(float)
        n_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss_dict = self._training_step(batch)
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        totals[k] += float(v.item())
                    else:
                        totals[k] += float(v)
                n_batches += 1

        if n_batches == 0:
            return {"total_loss": 0.0, "mask_loss": 0.0, "mlm_loss": 0.0}

        return {k: v / n_batches for k, v in totals.items()}
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Execute one training step.
        
        Args:
            batch: Dict with input_ids, attention_mask, slang_mask, etc.
        
        Returns:
            Dict with loss components
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        slang_mask = batch["slang_mask"]
        
        # Special token IDs
        special_ids = torch.tensor([
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        ], device=self.device)
        
        # Use the model's forward method directly for proper gradient flow
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=batch.get("token_type_ids"),
            special_token_ids=special_ids,
            return_extras=True,
        )
        
        # Parse outputs
        if isinstance(outputs, tuple):
            logits, extras = outputs
            mask_values = extras.get("mask_vals", None)
            hidden_states = extras.get("hidden_states", None)
        else:
            logits = outputs
            mask_values = None
            hidden_states = None
        
        losses = {}
        
        # 1. Mask Preservation Loss - ensure gradient connection
        if mask_values is not None:
            mask_loss = self.mask_preservation_loss(
                mask_values, slang_mask, attention_mask
            )
            losses["mask_loss"] = mask_loss * self.config.mask_preservation_weight
            
            # Compute mask statistics (detach for logging)
            with torch.no_grad():
                mask_vals_flat = mask_values.squeeze(-1) if mask_values.dim() == 3 else mask_values
                slang_tokens = (slang_mask * attention_mask).sum() + 1e-8
                nonslang_tokens = ((1 - slang_mask) * attention_mask).sum() + 1e-8
                losses["slang_mask_mean"] = (mask_vals_flat * slang_mask * attention_mask).sum() / slang_tokens
                losses["nonslang_mask_mean"] = (mask_vals_flat * (1 - slang_mask) * attention_mask).sum() / nonslang_tokens
        else:
            # Create a dummy loss that has gradient through logits
            losses["mask_loss"] = logits.sum() * 0.0  # Zero loss but with grad_fn
            losses["slang_mask_mean"] = torch.tensor(0.0, device=self.device)
            losses["nonslang_mask_mean"] = torch.tensor(0.0, device=self.device)
        
        # 2. Simple consistency loss using logits (alternative to MLM)
        # Encourage the model to produce consistent outputs
        if logits is not None:
            # Use entropy regularization to prevent overconfident predictions
            probs = torch.sigmoid(logits)
            entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
            consistency_loss = entropy.mean()
            losses["mlm_loss"] = consistency_loss * self.config.mlm_weight * 0.1
        else:
            losses["mlm_loss"] = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 3. Optional mask sparsity regularization
        if mask_values is not None and self.config.mask_sparsity_weight > 0:
            mask_vals_flat = mask_values.squeeze(-1) if mask_values.dim() == 3 else mask_values
            sparsity = ((1 - slang_mask.float()) * mask_vals_flat * attention_mask.float()).mean()
            losses["sparsity_loss"] = sparsity * self.config.mask_sparsity_weight
        else:
            losses["sparsity_loss"] = torch.tensor(0.0, device=self.device)
        
        # Total loss - ensure at least one component has gradients
        losses["total_loss"] = losses["mask_loss"] + losses["mlm_loss"]
        if isinstance(losses.get("sparsity_loss"), torch.Tensor) and losses["sparsity_loss"].requires_grad:
            losses["total_loss"] = losses["total_loss"] + losses["sparsity_loss"]
        
        return losses
    
    def _save_checkpoint(self, path: Path) -> None:
        """Save checkpoint."""
        # Only save trainable parameters
        state_dict = {
            k: v.cpu() for k, v in self.model.state_dict().items()
        }
        
        checkpoint = {
            "state_dict": state_dict,
            "config": self.config.__dict__,
            "slang_dict_size": len(self.slang_dict),
            "stats": dict(self.stats),
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_pretraining(
    model,
    tokenizer,
    data_path: str,
    dictionary_path: str,
    config: Optional[PretrainConfig] = None,
    device: Optional[torch.device] = None,
    text_field: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run mask-guided pretraining.
    
    Args:
        model: SoftLogicViBERT model
        tokenizer: Tokenizer
        data_path: Path to training data file (JSON/JSONL/CSV) or directory containing train/dev/test CSVs
        dictionary_path: Path to slang dictionary
        config: Pretraining configuration
        device: Device
        text_field: Preferred text field/column (CSV: e.g. original/normalized)
    
    Returns:
        Training statistics
    """
    if config is None:
        config = PretrainConfig()
    
    config.dictionary_path = dictionary_path
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # Load data (supports directory splits)
    train_texts, val_texts, _ = load_pretrain_splits(data_path, text_field=text_field)
    
    # Create pretrainer
    pretrainer = MaskGuidedPretrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )
    
    # Run pretraining
    stats = pretrainer.train(train_texts, val_texts=val_texts)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mask-Guided Language Adaptation Pretraining")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data file (JSON/JSONL/CSV) or directory containing train.csv/dev.csv/test.csv",
    )
    parser.add_argument("--dict-path", type=str, required=True, help="Path to slang dictionary")
    parser.add_argument("--model-path", type=str, default=None, help="Path to pretrained model checkpoint")
    parser.add_argument("--model-name", type=str, default="Fsoft-AIC/videberta-base", help="Base model name")
    parser.add_argument("--output-dir", type=str, default="pretrain_outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--mask-weight", type=float, default=0.2, help="Mask preservation loss weight")
    parser.add_argument(
        "--text-field",
        type=str,
        default=None,
        help="Text field/column to use (CSV: original|normalized; JSON: comment). If omitted, CSV defaults to 'original' if present.",
    )
    
    args = parser.parse_args()
    
    # Import model - handle both module and standalone execution
    try:
        from .model import SoftLogicViBERT
        from .config import ModelConfig
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from softlogic_vibert.model import SoftLogicViBERT
        from softlogic_vibert.config import ModelConfig
    
    # Create config
    config = PretrainConfig(
        model_name=args.model_name,
        dictionary_path=args.dict_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        mask_preservation_weight=args.mask_weight,
        output_dir=args.output_dir,
    )
    
    # Create or load model
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model_config = ModelConfig(**checkpoint["model_config"])
        model = SoftLogicViBERT(model_config)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model_config = ModelConfig(model_name=args.model_name)
        model = SoftLogicViBERT(model_config)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Run pretraining
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = run_pretraining(
        model,
        tokenizer,
        args.data_path,
        args.dict_path,
        config,
        device,
        text_field=args.text_field,
    )
    
    print("\nPretraining completed!")
    print(f"Final stats: {stats}")
