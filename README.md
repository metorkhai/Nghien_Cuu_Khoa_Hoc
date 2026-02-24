# SoftLogic ViDeBERTa (SoftLogic ViDeBERTa): Soft-Logic Reasoning for Vietnamese GenZ Sentiment Analysis

A research-grade implementation of a novel sentiment analysis architecture that goes **beyond naive fine-tuning** by introducing mechanistic reasoning, conflict awareness, and inductive bias—all while remaining end-to-end differentiable.

## Core Innovation

Traditional sentiment models fail on GenZ informal language because they:
- Collapse to predicting dominant labels (e.g., "Enjoyment")
- Miss contradictions (positive surface + negative intent)
- Rely on lexical shortcuts instead of meaning

**SoftLogic ViBERT** (implemented on a ViDeBERTa backbone) addresses these through:

1. **Token-level TF-IDF Gating**: Statistical token weights that suppress low-salience tokens
2. **Multi-view Representations**: Semantic, lexical, and pragmatic perspectives
3. **Differentiable Fuzzy Logic**: Soft predicates + fuzzy rules with a learnable AND sharpness
4. **Conflict Detection**: Explicit modeling of sarcasm and contradictions

## Architecture

```
(context, comment)
       ↓
┌─────────────────────────────────┐
│   Token-level TF-IDF Gating     │  ← Precomputed scores per token id
│   s_i = tfidf(token_id_i)       │
│   E'_i = s_i * E_i              │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│    ViDeBERTa Encoder            │  ← Fsoft-AIC/videberta-base
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│   Multi-view Projection Heads   │
│   ┌───────┐ ┌───────┐ ┌───────┐ │
│   │z_sem  │ │z_lex  │ │z_prag │ │
│   │(CLS)  │ │(CNN)  │ │(MLP)  │ │
│   └───────┘ └───────┘ └───────┘ │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│   Soft Logic Inference Module   │
│                                 │
│   Predicates:                   │
│   P_pos_sem, P_neg_sem,         │
│   P_pos_lex, P_neg_lex,         │
│   P_high_int ∈ (0,1)            │
│                                 │
│   Rules:                        │
│   r1 = AND(P_pos_lex, P_neg_sem)│  ← Sarcasm
│   r2 = AND(P_neg_lex, P_neg_sem)│  ← Strong negative
│   r3 = AND(P_pos_sem, NOT(P_high_int))
│   r4 = AND(P_high_int, P_neg_sem)
│   r5 = |P_pos_lex - P_pos_sem|  │  ← Inconsistency
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│   Sentiment Inference MLP       │
│   [z_sem, z_lex, z_prag, r1-r5] │
│            ↓                    │
│   Multi-label Classification    │
└─────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone <repository>
cd softlogic_vibert

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- (Optional) CUDA for GPU acceleration

## Quick Start

### Training

```bash
# Train with local JSON/JSONL data
python -m softlogic_vibert.train \
    --data-path output_data.json \
    --output-dir softlogic_outputs \
    --epochs 10 \
    --batch-size 16

# Train with HuggingFace dataset
python -m softlogic_vibert.train \
    --hf-dataset tridm/UIT-VSMEC \
    --hf-split train \
    --output-dir softlogic_outputs
```

### Inference

```bash
# Single prediction with analysis
python -m softlogic_vibert.inference \
    --ckpt softlogic_outputs/softlogic_vibert_state.pt \
    --comment "em yêu anh quá đi 😍😍😍" \
    --verbose

# JSON output
python -m softlogic_vibert.inference \
    --ckpt softlogic_outputs/softlogic_vibert_state.pt \
    --comment "ngon thật đấy 🙄" \
    --json
```

### Python API (quick predictor):

```python
from softlogic_vibert import SentimentPredictor

path = "outputs/softlogic_vibert_state.pt"
device = "cpu"  # or "cuda"

model = SentimentPredictor.load(path, device=device)

comment = "ngon thật đấy 🙄"
context = "Đây là nhà hàng mới mở"

# Note: predict(comment, context). Using keywords avoids argument-order mistakes.
pred = model.predict(comment=comment, context=context)
print("Nhãn là:", pred)
```

### Ablation Studies

```bash
# Run core ablations
python -m softlogic_vibert.ablation run \
    --data-path output_data.json \
    --experiments core \
    --epochs 5

# Run all ablations
python -m softlogic_vibert.ablation run \
    --data-path output_data.json \
    --experiments all

# Compare results
python -m softlogic_vibert.ablation compare \
    --study-dir ablation_results/ablation_study_YYYYMMDD_HHMMSS
```

## Ablation Configurations

| Configuration | Masking | Multi-view | Logic | Description |
|--------------|---------|------------|-------|-------------|
| `vibert_only` | No | No | No | Backbone only (ViDeBERTa) |
| `mask_only` | Yes | No | No | Backbone + TF-IDF gating |
| `multiview_no_logic` | Yes | Yes | No | Multi-view without reasoning |
| `full_model` | Yes | Yes | Yes | Complete architecture |
| `drop_r1` - `drop_r5` | Yes | Yes | Yes* | Remove individual rules |

Command-line flags:
```bash
# Backbone baseline
--no-mask --no-multiview --no-logic

# Mask only
--use-mask --no-multiview --no-logic

# Multi-view (no logic)
--use-mask --use-multiview --no-logic

# Full model
--use-mask --use-multiview --use-logic

# Drop specific rules
--drop-rules r1,r5
```

## Data Format

Expected JSON/JSONL format:

```json
{
    "comment": "ngon thật đấy 🙄",
    "context": "Đây là nhà hàng mới mở",
    "labels": ["Disgust", "Sarcasm"]
}
```

- `comment`: Required. User-generated text (GenZ slang, emoji, informal)
- `context`: Optional. Background/narrative text
- `labels`: Multi-label list (e.g., Enjoyment, Anger, Disgust, Surprise, Fear, Sadness, Other)

## Interpretability

### Python API

```python
from softlogic_vibert import create_interpreter

# Create interpreter from checkpoint
interpreter = create_interpreter("outputs/softlogic_vibert_state.pt")

# Analyze a sample
result = interpreter.analyze(
    comment="ngon thật đấy 🙄",
    context="Nhà hàng này review 5 sao"
)

# Get detailed analysis
print(result.predicted_labels)       # ['Disgust']
print(result.rule_activations)       # {'r1': 0.82, 'r2': 0.21, ...}
print(result.important_tokens)       # ['ngon', 'thật', 'đấy']
print(result.reasoning_summary)      # {'sarcasm_likely': True, ...}

# Natural language explanation
explanation = interpreter.explain_prediction(
    comment="ngon thật đấy 🙄"
)
print(explanation)
```

### Accessing Token Masks

```python
from softlogic_vibert import load_model, predict_single

model, tokenizer, ckpt = load_model("outputs/softlogic_vibert_state.pt")

result = predict_single(
    model, tokenizer,
    comment="em yêu anh quá đi",
    return_details=True
)

# Token importance
for token_info in result["token_masks"]:
    print(f"{token_info['token']}: {token_info['mask_value']:.3f}")

# Rule activations
for rule, value in result["rule_activations"].items():
    print(f"{rule}: {value:.3f}")
```

## Soft Logic Module

### Differentiable Operations

```python
# Fuzzy AND (parameterized product t-norm)
# p is a learned scalar (clamped in code)
AND(a, b) = (a * b) ** p

# Fuzzy OR (probabilistic sum)
OR(a, b) = a + b - a*b

# Fuzzy NOT (complement)
NOT(a) = 1 - a

# Fuzzy implication (Reichenbach operator)
IMPLIES(a, b) = 1 - a + a*b
```

### Reasoning Rules

| Rule | Formula | Interpretation |
|------|---------|----------------|
| r1 | `AND(P_pos_lex, P_neg_sem)` | Sarcasm / contradiction (positive surface, negative meaning) |
| r2 | `AND(P_neg_lex, P_neg_sem)` | Strong negative (lexical + semantic both negative) |
| r3 | `AND(P_pos_sem, NOT(P_high_int))` | Mild positive (positive semantics without high intensity) |
| r4 | `AND(P_high_int, P_neg_sem)` | Intense negative (high intensity + negative semantics) |
| r5 | `\|P_pos_lex - P_pos_sem\|` | Inconsistency / surprise (surface vs semantic mismatch) |

## Training Tips

### Handling Class Imbalance

```bash
# Use focal loss for imbalanced data
python -m softlogic_vibert.train \
    --loss-type focal \
    --focal-gamma 2.0 \
    ...

# Use asymmetric loss (recommended for multi-label)
python -m softlogic_vibert.train \
    --loss-type asymmetric \
    ...
```

### Learning Rate Tuning

```bash
# Different learning rates for encoder vs heads
python -m softlogic_vibert.train \
    --lr 2e-5 \
    --encoder-lr 2e-5 \
    --head-lr 5e-4 \
    ...
```

### Mixed Precision Training

```bash
# Enable FP16 for faster training (requires CUDA)
python -m softlogic_vibert.train \
    --fp16 \
    ...
```

## Project Structure

```
softlogic_vibert/
├── __init__.py          # Package exports
├── config.py            # Configuration dataclasses
├── model.py             # Core SoftLogicViBERT model
├── train.py             # Training script
├── inference.py         # Inference script
├── losses.py            # Custom loss functions
├── metrics.py           # Evaluation metrics
├── data.py              # Data loading utilities
├── utils.py             # Helper functions
├── ablation.py          # Ablation study runner
├── interpretability.py  # Model analysis tools
└── README.md            # This file
```

## Saved Checkpoints

Training produces:
- `softlogic_vibert_state.pt`: State dict (lightweight, recommended)
- `softlogic_vibert_full.pt`: Full model object
- `train_summary.json`: Training metrics summary
- `training_history.json`: Per-epoch metrics
- `config.json`: Experiment configuration

## Advanced Usage

### Custom Training Loop

```python
from softlogic_vibert import (
    SoftLogicViBERT, ModelConfig, SoftLogicLoss,
    load_and_prepare_data, prepare_dataloaders
)
from transformers import AutoTokenizer
import torch

# Config
config = ModelConfig(
    model_name="Fsoft-AIC/videberta-base",
    use_mask=True,
    use_multiview=True,
    use_logic=True,
)

# Load data
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
train_rows, val_rows, label_map, label_list = load_and_prepare_data("data.json")
train_loader, val_loader = prepare_dataloaders(
    train_rows, val_rows, tokenizer, label_map
)

# Create model
config.num_labels = len(label_map)
model = SoftLogicViBERT(config).cuda()

# Custom training...
```

### Extending the Logic Module

```python
class ExtendedLogicModule(SoftLogicModule):
    def __init__(self, proj_size):
        super().__init__(proj_size)
        # Add custom predicates
        self.pred_custom = SoftPredicate(proj_size)
    
    def forward(self, z_sem, z_lex, z_prag):
        base_rules, base_details = super().forward(z_sem, z_lex, z_prag)
        
        # Add custom rule
        p_custom = self.pred_custom(z_sem)
        r_custom = self.AND(p_custom, base_details["p_high_int"])
        
        # Extend outputs...
        return extended_rules, extended_details
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{softlogic_vibert,
    title = {SoftLogic ViDeBERTa (SoftLogic ViBERT): Soft-Logic Reasoning for Vietnamese GenZ Sentiment},
  year = {2024},
  description = {A novel sentiment analysis architecture with differentiable fuzzy logic}
}
```

## License

This project is released for research purposes.

## Acknowledgments

- ViDeBERTa backbone (Fsoft-AIC/videberta-base)
- HuggingFace Transformers
- PyTorch team
