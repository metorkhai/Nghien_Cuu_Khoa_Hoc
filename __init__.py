"""
SoftLogic ViBERT: Soft-Logic Reasoning Architecture for Vietnamese GenZ Sentiment Analysis.

A research-grade implementation of a novel sentiment analysis model that combines:
- ViBERT as the semantic backbone
- Token-level soft masking for noise suppression
- Multi-view representations (semantic, lexical, pragmatic)
- Differentiable fuzzy-logic inference for reasoning
- End-to-end trainable with interpretability

Key Components:
- model.py: Core SoftLogicViBERT architecture
- config.py: Configuration dataclasses
- train.py: Training pipeline
- inference.py: Inference and prediction
- interpretability.py: Model analysis tools
- ablation.py: Systematic ablation studies
- losses.py: Custom loss functions
- metrics.py: Evaluation metrics
- data.py: Data loading utilities
- utils.py: Helper functions

Quick Start:
    # Training
    python -m softlogic_vibert.train --data-path data.json --output-dir outputs
    
    # Inference
    python -m softlogic_vibert.inference --ckpt outputs/softlogic_vibert_state.pt --comment "text"
    
    # Ablation study
    python -m softlogic_vibert.ablation run --data-path data.json --experiments core
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Core model
from .core.model import SoftLogicViBERT

# Configuration
from .core.config import (
    ModelConfig,
    TrainConfig,
    ExperimentConfig,
    get_ablation_configs,
    get_vibert_only_config,
    get_mask_only_config,
    get_multiview_no_logic_config,
    get_full_config,
)

# Data utilities
from .data_utils.data import (
    SentimentDataset,
    InferenceDataset,
    load_data_file,
    load_hf_dataset,
    prepare_dataloaders,
    load_and_prepare_data,
    collate_batch,
)

# Inference
from .analysis.inference import (
    SentimentPredictor,
    load_model,
    predict_single,
    predict_batch,
    analyze_prediction,
)

# Metrics
from .core.metrics import (
    multilabel_f1,
    multilabel_metrics,
    evaluate_model,
    find_optimal_threshold,
)

# Losses
from .core.losses import (
    FocalLoss,
    AsymmetricLoss,
    LabelSmoothingBCE,
    SoftLogicLoss,
    compute_pos_weight,
)

# Utilities
from .data_utils.utils import (
    set_seed,
    ensure_list,
    build_label_map,
    extract_prag_features,
    visualize_token_masks,
    interpret_rules,
)

# Interpretability
from .analysis.interpretability import (
    ModelInterpreter,
    InterpretabilityResult,
    create_interpreter,
)

# Pretraining
from .training.pretrain import (
    PretrainConfig,
    SlangDictionary,
    SlangAwareDataset,
    MaskPreservationLoss,
    MaskGuidedPretrainer,
    run_pretraining,
)

__all__ = [
    # Core
    "SoftLogicViBERT",
    
    # Config
    "ModelConfig",
    "TrainConfig",
    "ExperimentConfig",
    "get_ablation_configs",
    
    # Data
    "SentimentDataset",
    "InferenceDataset",
    "load_data_file",
    "prepare_dataloaders",
    
    # Inference
    "SentimentPredictor",
    "load_model",
    "predict_single",
    "predict_batch",
    
    # Metrics
    "multilabel_f1",
    "multilabel_metrics",
    "evaluate_model",
    
    # Losses
    "SoftLogicLoss",
    "FocalLoss",
    
    # Interpretability
    "ModelInterpreter",
    "create_interpreter",
    
    # Pretraining
    "PretrainConfig",
    "SlangDictionary",
    "MaskGuidedPretrainer",
    "run_pretraining",
    
    # Utils
    "set_seed",
    "extract_prag_features",
]

