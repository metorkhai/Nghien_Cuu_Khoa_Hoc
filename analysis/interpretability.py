"""
Interpretability module for SoftLogic ViBERT.

Provides tools for understanding model decisions:
- Token importance visualization
- Rule activation analysis
- Contradiction detection
- Attention-like attribution maps
- Comparison across samples
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import ModelConfig
from ..core.model import SoftLogicViBERT
from ..data_utils.utils import extract_prag_features, normalize_vietnamese_text


@dataclass
class InterpretabilityResult:
    """Container for interpretability analysis results."""
    
    # Input
    comment: str
    context: Optional[str]
    
    # Predictions
    predicted_labels: List[str]
    label_probabilities: Dict[str, float]
    
    # Token analysis
    token_importance: List[Dict[str, Any]]
    important_tokens: List[str]
    masked_tokens: List[str]
    
    # Rule analysis
    rule_activations: Dict[str, float]
    active_rules: List[str]
    rule_interpretations: List[str]
    
    # Predicate analysis
    predicate_values: Dict[str, float]
    
    # Reasoning summary
    reasoning_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "comment": self.comment,
            "context": self.context,
            "predicted_labels": self.predicted_labels,
            "label_probabilities": self.label_probabilities,
            "token_importance": self.token_importance,
            "important_tokens": self.important_tokens,
            "masked_tokens": self.masked_tokens,
            "rule_activations": self.rule_activations,
            "active_rules": self.active_rules,
            "rule_interpretations": self.rule_interpretations,
            "predicate_values": self.predicate_values,
            "reasoning_summary": self.reasoning_summary,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class ModelInterpreter:
    """
    Interpretability analyzer for SoftLogic ViBERT.
    
    Provides detailed analysis of model decisions including:
    - Token-level importance via mask values
    - Rule activation patterns
    - Predicate-level reasoning
    - Contradiction detection
    """
    
    # Rule descriptions
    RULE_DESCRIPTIONS = {
        "r1": "Sarcasm/Contradiction: Surface positive + underlying negative",
        "r2": "Strong Negative: Both lexical and semantic signals are negative",
        "r3": "Mild Positive: Genuine positive without strong intensity",
        "r4": "Intense Negative: High emotional intensity with negative semantics",
        "r5": "Inconsistency: Mismatch between lexical and semantic signals",
    }
    
    # Predicate descriptions
    PREDICATE_DESCRIPTIONS = {
        "p_pos_sem": "Positive Semantic: Deep positive meaning detected",
        "p_neg_sem": "Negative Semantic: Deep negative meaning detected",
        "p_pos_lex": "Positive Lexical: Surface positive signals detected",
        "p_neg_lex": "Negative Lexical: Surface negative signals detected",
        "p_high_int": "High Intensity: Strong emotional intensity detected",
    }
    
    def __init__(
        self,
        model: SoftLogicViBERT,
        tokenizer,
        label_list: List[str],
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.device = device or next(model.parameters()).device
        
        self.special_token_ids = torch.tensor([
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ], device=self.device)
        
        self.model.eval()
    
    def analyze(
        self,
        comment: str,
        context: Optional[str] = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        rule_threshold: float = 0.4,
    ) -> InterpretabilityResult:
        """
        Perform comprehensive interpretability analysis.
        
        Args:
            comment: Input comment text
            context: Optional context text
            threshold: Classification threshold
            mask_threshold: Threshold for important tokens
            rule_threshold: Threshold for active rules
        
        Returns:
            InterpretabilityResult with full analysis
        """
        # Normalize text
        comment = normalize_vietnamese_text(comment)
        if context:
            context = normalize_vietnamese_text(context)
        
        # Tokenize
        tok = self.tokenizer(
            comment,
            text_pair=context,
            max_length=self.model.config.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Extract features
        prag = torch.tensor([extract_prag_features(comment)], dtype=torch.float)
        
        # Forward pass
        with torch.no_grad():
            input_ids = tok["input_ids"].to(self.device)
            attention_mask = tok["attention_mask"].to(self.device)
            token_type_ids = tok.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            
            logits, extras = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                prag_features=prag.to(self.device),
                special_token_ids=self.special_token_ids,
                return_extras=True,
            )
        
        # Process predictions
        probs = torch.sigmoid(logits).squeeze(0)
        preds = (probs >= threshold).int()
        
        predicted_labels = [
            self.label_list[i] for i, v in enumerate(preds.cpu().tolist()) if v == 1
        ]
        label_probabilities = {
            self.label_list[i]: prob
            for i, prob in enumerate(probs.cpu().tolist())
        }
        
        # Analyze tokens
        token_importance, important_tokens, masked_tokens = self._analyze_tokens(
            input_ids.squeeze(0),
            attention_mask.squeeze(0),
            extras.get("mask_vals"),
            mask_threshold,
        )
        
        # Analyze rules
        rule_activations, active_rules, rule_interpretations = self._analyze_rules(
            extras.get("rules", {}),
            rule_threshold,
        )
        
        # Analyze predicates
        predicate_values = self._analyze_predicates(extras.get("predicates", {}))
        
        # Generate reasoning summary
        reasoning_summary = self._generate_reasoning_summary(
            rule_activations,
            predicate_values,
            predicted_labels,
            extras,
        )
        
        return InterpretabilityResult(
            comment=comment,
            context=context,
            predicted_labels=predicted_labels,
            label_probabilities=label_probabilities,
            token_importance=token_importance,
            important_tokens=important_tokens,
            masked_tokens=masked_tokens,
            rule_activations=rule_activations,
            active_rules=active_rules,
            rule_interpretations=rule_interpretations,
            predicate_values=predicate_values,
            reasoning_summary=reasoning_summary,
        )
    
    def _analyze_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_vals: Optional[torch.Tensor],
        threshold: float,
    ) -> Tuple[List[Dict], List[str], List[str]]:
        """Analyze token importance based on mask values."""
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.cpu().tolist())
        
        token_importance = []
        important_tokens = []
        masked_tokens = []
        
        if mask_vals is not None:
            mask_vals = mask_vals.squeeze()
            
            for i, (token, mask_val, attn) in enumerate(zip(
                tokens,
                mask_vals.cpu().tolist() if mask_vals.dim() == 1 else mask_vals.squeeze(-1).cpu().tolist(),
                attention_mask.cpu().tolist(),
            )):
                if attn == 0 or token in ["[PAD]", "<pad>"]:
                    continue
                
                # Handle multi-dimensional mask values
                if isinstance(mask_val, list):
                    mask_val = mask_val[0] if mask_val else 0.0
                
                importance_level = "high" if mask_val >= threshold else "medium" if mask_val >= 0.3 else "low"
                
                token_importance.append({
                    "token": token,
                    "mask_value": float(mask_val),
                    "importance": importance_level,
                    "position": i,
                })
                
                if mask_val >= threshold:
                    important_tokens.append(token)
                elif mask_val < 0.3:
                    masked_tokens.append(token)
        else:
            # No masking - all tokens equally important
            for i, (token, attn) in enumerate(zip(tokens, attention_mask.cpu().tolist())):
                if attn > 0 and token not in ["[PAD]", "<pad>"]:
                    token_importance.append({
                        "token": token,
                        "mask_value": 1.0,
                        "importance": "high",
                        "position": i,
                    })
                    important_tokens.append(token)
        
        return token_importance, important_tokens, masked_tokens
    
    def _analyze_rules(
        self,
        rules: Dict[str, torch.Tensor],
        threshold: float,
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """Analyze rule activations."""
        rule_activations = {}
        active_rules = []
        interpretations = []
        
        for rule_key in ["r1", "r2", "r3", "r4", "r5"]:
            if rule_key in rules:
                value = rules[rule_key]
                if isinstance(value, torch.Tensor):
                    value = value.mean().item()
                
                rule_activations[rule_key] = float(value)
                
                if value >= threshold:
                    active_rules.append(rule_key)
                    desc = self.RULE_DESCRIPTIONS.get(rule_key, f"Unknown rule {rule_key}")
                    interpretations.append(f"[{rule_key.upper()}] ({value:.2f}): {desc}")
        
        if not interpretations:
            interpretations.append("No significant reasoning patterns detected")
        
        return rule_activations, active_rules, interpretations
    
    def _analyze_predicates(
        self,
        predicates: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Analyze predicate values."""
        predicate_values = {}
        
        for pred_key in ["p_pos_sem", "p_neg_sem", "p_pos_lex", "p_neg_lex", "p_high_int"]:
            if pred_key in predicates:
                value = predicates[pred_key]
                if isinstance(value, torch.Tensor):
                    value = value.mean().item()
                predicate_values[pred_key] = float(value)
        
        return predicate_values
    
    def _generate_reasoning_summary(
        self,
        rule_activations: Dict[str, float],
        predicate_values: Dict[str, float],
        predicted_labels: List[str],
        extras: Dict,
    ) -> Dict[str, Any]:
        """Generate a high-level reasoning summary."""
        summary = {
            "dominant_pattern": None,
            "polarity": "neutral",
            "intensity": "normal",
            "contradiction_detected": False,
            "sarcasm_likely": False,
            "confidence": "low",
            "reasoning_chain": [],
        }
        
        # Determine polarity
        pos_sem = predicate_values.get("p_pos_sem", 0)
        neg_sem = predicate_values.get("p_neg_sem", 0)
        pos_lex = predicate_values.get("p_pos_lex", 0)
        
        if pos_sem > neg_sem and pos_sem > 0.5:
            summary["polarity"] = "positive"
        elif neg_sem > pos_sem and neg_sem > 0.5:
            summary["polarity"] = "negative"
        
        # Determine intensity
        high_int = predicate_values.get("p_high_int", 0)
        if high_int > 0.7:
            summary["intensity"] = "high"
        elif high_int < 0.3:
            summary["intensity"] = "low"
        
        # Check for contradiction
        r1 = rule_activations.get("r1", 0)
        r5 = rule_activations.get("r5", 0)
        
        if r1 > 0.5 or r5 > 0.5:
            summary["contradiction_detected"] = True
            summary["reasoning_chain"].append(
                "Detected mismatch between surface expression and underlying meaning"
            )
        
        if r1 > 0.5:
            summary["sarcasm_likely"] = True
            summary["reasoning_chain"].append(
                "Sarcasm pattern: positive surface form with negative semantic content"
            )
        
        # Determine dominant pattern
        max_rule = max(rule_activations.items(), key=lambda x: x[1]) if rule_activations else (None, 0)
        if max_rule[1] > 0.4:
            pattern_names = {
                "r1": "sarcasm",
                "r2": "strong_negative",
                "r3": "mild_positive",
                "r4": "intense_negative",
                "r5": "inconsistency",
            }
            summary["dominant_pattern"] = pattern_names.get(max_rule[0])
        
        # Confidence assessment
        if predicted_labels:
            max_prob = max(extras.get("label_probabilities", {}).values()) if "label_probabilities" in extras else 0
            # Estimate from logits if not available
            if max_prob == 0 and "z_sem" in extras:
                # Just check if we have strong predictions
                summary["confidence"] = "medium"
            elif max_prob > 0.8:
                summary["confidence"] = "high"
            elif max_prob > 0.5:
                summary["confidence"] = "medium"
        
        # Build reasoning chain
        if summary["polarity"] == "positive":
            summary["reasoning_chain"].append(
                f"Detected positive semantic content (strength: {pos_sem:.2f})"
            )
        elif summary["polarity"] == "negative":
            summary["reasoning_chain"].append(
                f"Detected negative semantic content (strength: {neg_sem:.2f})"
            )
        
        if summary["intensity"] == "high":
            summary["reasoning_chain"].append(
                f"High emotional intensity detected (strength: {high_int:.2f})"
            )
        
        return summary
    
    def compare_samples(
        self,
        samples: List[Tuple[str, Optional[str]]],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compare analysis across multiple samples.
        
        Args:
            samples: List of (comment, context) tuples
            threshold: Classification threshold
        
        Returns:
            Comparison dictionary
        """
        analyses = []
        
        for comment, context in samples:
            result = self.analyze(comment, context, threshold)
            analyses.append(result)
        
        # Aggregate statistics
        all_rules = {f"r{i}": [] for i in range(1, 6)}
        all_predicates = {
            "p_pos_sem": [],
            "p_neg_sem": [],
            "p_pos_lex": [],
            "p_neg_lex": [],
            "p_high_int": [],
        }
        
        for analysis in analyses:
            for rule, value in analysis.rule_activations.items():
                all_rules[rule].append(value)
            for pred, value in analysis.predicate_values.items():
                all_predicates[pred].append(value)
        
        # Compute statistics
        comparison = {
            "num_samples": len(samples),
            "rule_statistics": {
                rule: {
                    "mean": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                }
                for rule, values in all_rules.items()
            },
            "predicate_statistics": {
                pred: {
                    "mean": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                }
                for pred, values in all_predicates.items()
            },
            "individual_analyses": [a.to_dict() for a in analyses],
        }
        
        return comparison
    
    def explain_prediction(
        self,
        comment: str,
        context: Optional[str] = None,
        threshold: float = 0.5,
    ) -> str:
        """
        Generate a natural language explanation of the prediction.
        
        Args:
            comment: Input comment
            context: Optional context
            threshold: Classification threshold
        
        Returns:
            Natural language explanation string
        """
        result = self.analyze(comment, context, threshold)
        
        lines = []
        lines.append(f"Analysis for: \"{comment}\"")
        lines.append("")
        
        # Predictions
        if result.predicted_labels:
            lines.append(f"Predicted emotions: {', '.join(result.predicted_labels)}")
        else:
            lines.append("No emotions predicted above threshold")
        
        lines.append("")
        
        # Key reasoning
        lines.append("Reasoning:")
        for chain_item in result.reasoning_summary.get("reasoning_chain", []):
            lines.append(f"   - {chain_item}")
        
        if not result.reasoning_summary.get("reasoning_chain"):
            lines.append("   - No significant reasoning patterns detected")
        
        lines.append("")
        
        # Key tokens
        if result.important_tokens:
            lines.append(f"Key tokens: {', '.join(result.important_tokens[:5])}")
        
        if result.masked_tokens:
            lines.append(f"Filtered tokens: {', '.join(result.masked_tokens[:5])}")
        
        lines.append("")
        
        # Summary
        summary = result.reasoning_summary
        if summary.get("contradiction_detected"):
            lines.append("Warning: Contradiction detected between surface and deep meaning")
        if summary.get("sarcasm_likely"):
            lines.append("Sarcasm likely")
        
        return "\n".join(lines)


def create_interpreter(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> ModelInterpreter:
    """
    Create a ModelInterpreter from a checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        ModelInterpreter instance
    """
    from .inference import load_model
    
    model, tokenizer, checkpoint = load_model(checkpoint_path, device)
    label_list = checkpoint.get("label_list", [])
    
    return ModelInterpreter(
        model=model,
        tokenizer=tokenizer,
        label_list=label_list,
        device=device,
    )
