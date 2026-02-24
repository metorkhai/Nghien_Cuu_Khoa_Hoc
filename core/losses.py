"""
Custom loss functions for SoftLogic ViBERT.

Includes:
- Focal loss for handling class imbalance
- Asymmetric loss for multi-label classification
- Label smoothing
- Combined loss with mask regularization
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-label classification.
    
    Focal loss down-weights easy examples and focuses on hard negatives.
    
    FL(p) = -α * (1-p)^γ * log(p)  for positive class
    FL(p) = -(1-α) * p^γ * log(1-p)  for negative class
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class weight for positive examples
        pos_weight: Per-class positive weights [C]
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Raw model outputs [B, C]
            targets: Multi-hot targets [B, C]
        
        Returns:
            loss: Scalar loss value
        """
        probs = torch.sigmoid(logits)
        
        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weights
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        
        # Apply focal and alpha weights
        loss = alpha_t * focal_weight * bce
        
        # Apply per-class weights if provided
        if self.pos_weight is not None:
            weight = self.pos_weight.to(logits.device)
            # Adjust weight based on target
            class_weight = weight * targets + (1 - targets)
            loss = loss * class_weight
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Addresses the positive-negative imbalance inherent in multi-label tasks.
    Uses different gamma values for positive and negative examples.
    
    Reference: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    
    Args:
        gamma_neg: Focusing parameter for negative examples
        gamma_pos: Focusing parameter for positive examples
        clip: Probability margin for hard negatives
        eps: Small constant for numerical stability
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            logits: Raw model outputs [B, C]
            targets: Multi-hot targets [B, C]
        
        Returns:
            loss: Scalar loss value
        """
        probs = torch.sigmoid(logits)
        
        # Asymmetric clipping for negatives
        probs_clipped = (probs + self.clip).clamp(max=1)
        
        # Positive examples
        pos_loss = targets * torch.log(probs.clamp(min=self.eps))
        pos_loss = pos_loss * ((1 - probs) ** self.gamma_pos)
        
        # Negative examples (with clipping)
        neg_loss = (1 - targets) * torch.log((1 - probs_clipped).clamp(min=self.eps))
        neg_loss = neg_loss * (probs_clipped ** self.gamma_neg)
        
        loss = -(pos_loss + neg_loss)
        
        return loss.mean()


class LabelSmoothingBCE(nn.Module):
    """
    BCE loss with label smoothing.
    
    Smooths the targets to prevent overconfidence:
        smoothed_target = target * (1 - smoothing) + 0.5 * smoothing
    
    Args:
        smoothing: Smoothing factor (0 = no smoothing)
        pos_weight: Per-class positive weights [C]
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label-smoothed BCE loss.
        """
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets_smooth,
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
        )
        
        return loss


class SoftLogicLoss(nn.Module):
    """
    Combined loss function for SoftLogic ViBERT.
    
    Combines:
    1. Classification loss (BCE, Focal, or Asymmetric)
    2. Mask sparsity regularization
    3. Optional rule diversity regularization
    
    Args:
        loss_type: 'bce', 'focal', or 'asymmetric'
        mask_lambda: Weight for mask sparsity loss
        rule_diversity_lambda: Weight for rule diversity loss
        pos_weight: Per-class positive weights
        focal_gamma: Gamma for focal loss
        label_smoothing: Label smoothing factor
    """
    
    def __init__(
        self,
        loss_type: str = "bce",
        mask_lambda: float = 1e-3,
        rule_diversity_lambda: float = 0.0,
        pos_weight: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        deep_supervision_lambda: float = 0.5,
        slang_preserve_lambda: float = 0.1,
        mask_sparsity_lambda: float = 0.01,
        use_polarity_supervision: bool = True,
        use_slang_preserve: bool = True,
        use_mask_sparsity: bool = True,
    ):
        super().__init__()
        
        self.mask_lambda = mask_lambda
        self.rule_diversity_lambda = rule_diversity_lambda
        self.deep_supervision_lambda = deep_supervision_lambda
        self.slang_preserve_lambda = slang_preserve_lambda
        self.mask_sparsity_lambda = mask_sparsity_lambda
        self.use_polarity_supervision = use_polarity_supervision
        self.use_slang_preserve = use_slang_preserve
        self.use_mask_sparsity = use_mask_sparsity
        
        # Select classification loss
        if loss_type == "focal":
            self.cls_loss = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
        elif loss_type == "asymmetric":
            self.cls_loss = AsymmetricLoss()
        else:  # bce
            if label_smoothing > 0:
                self.cls_loss = LabelSmoothingBCE(
                    smoothing=label_smoothing,
                    pos_weight=pos_weight,
                )
            else:
                self.cls_loss = nn.BCEWithLogitsLoss(
                    pos_weight=pos_weight,
                )
        
        self.pos_weight = pos_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_mean: Optional[torch.Tensor] = None,
        rules: Optional[Dict[str, torch.Tensor]] = None,
        mask_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        slang_mask: Optional[torch.Tensor] = None,
        slang_weights: Optional[torch.Tensor] = None,
        polarity_targets: Optional[torch.Tensor] = None,
        polarity_mask: Optional[torch.Tensor] = None,
        predicates: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            logits: Model outputs [B, C]
            targets: Multi-hot targets [B, C]
            mask_mean: Average mask activation (for regularization)
            rules: Dictionary of rule activations
        
        Returns:
            Dictionary with 'loss' and component breakdowns
        """
        # Classification loss
        cls_loss = self.cls_loss(logits, targets)
        
        total_loss = cls_loss
        losses = {"cls_loss": cls_loss}
        
        # Mask sparsity regularization (legacy mean-based)
        if mask_mean is not None and self.mask_lambda > 0:
            mask_loss = self.mask_lambda * mask_mean
            total_loss = total_loss + mask_loss
            losses["mask_loss"] = mask_loss

        # Polarity supervision (semantic + lexical)
        if (
            self.use_polarity_supervision
            and self.deep_supervision_lambda > 0
            and predicates is not None
            and polarity_targets is not None
        ):
            sem_pred = torch.cat([
                predicates["p_pos_sem"],
                predicates["p_neg_sem"],
            ], dim=1)
            lex_pred = torch.cat([
                predicates["p_pos_lex"],
                predicates["p_neg_lex"],
            ], dim=1)

            sem_pred = torch.nan_to_num(sem_pred, nan=0.5, posinf=1.0, neginf=0.0)
            lex_pred = torch.nan_to_num(lex_pred, nan=0.5, posinf=1.0, neginf=0.0)
            sem_pred = sem_pred.clamp(min=1e-6, max=1 - 1e-6)
            lex_pred = lex_pred.clamp(min=1e-6, max=1 - 1e-6)

            polarity_targets = torch.nan_to_num(polarity_targets, nan=0.0, posinf=1.0, neginf=0.0)
            polarity_targets = polarity_targets.clamp(min=0.0, max=1.0)

            sem_loss = F.binary_cross_entropy(sem_pred, polarity_targets, reduction="none")
            lex_loss = F.binary_cross_entropy(lex_pred, polarity_targets, reduction="none")

            if polarity_mask is not None:
                polarity_mask = polarity_mask.unsqueeze(1)
                sem_loss = sem_loss * polarity_mask
                lex_loss = lex_loss * polarity_mask
                denom = polarity_mask.sum() * polarity_targets.size(1) + 1e-8
            else:
                denom = polarity_targets.numel() + 1e-8

            sem_loss = sem_loss.sum() / denom
            lex_loss = lex_loss.sum() / denom

            deep_loss = self.deep_supervision_lambda * (sem_loss + lex_loss)
            total_loss = total_loss + deep_loss
            losses["sem_loss"] = sem_loss
            losses["lex_loss"] = lex_loss
            losses["deep_loss"] = deep_loss

        # Slang preservation loss
        if (
            self.use_slang_preserve
            and self.slang_preserve_lambda > 0
            and mask_values is not None
            and slang_mask is not None
        ):
            token_mask = mask_values.squeeze(-1)
            valid_mask = torch.ones_like(token_mask)
            if attention_mask is not None:
                valid_mask = valid_mask * attention_mask.float()
            if special_tokens_mask is not None:
                valid_mask = valid_mask * (1 - special_tokens_mask.float())

            slang_mask = slang_mask.float() * valid_mask
            weights = slang_weights.float() if slang_weights is not None else slang_mask
            weights = weights * slang_mask

            denom = slang_mask.sum() + 1e-8
            slang_loss = (weights * (1 - token_mask)).sum() / denom
            slang_loss = self.slang_preserve_lambda * slang_loss
            total_loss = total_loss + slang_loss
            losses["slang_loss"] = slang_loss

        # Non-slang sparsity loss
        if (
            self.use_mask_sparsity
            and self.mask_sparsity_lambda > 0
            and mask_values is not None
        ):
            token_mask = mask_values.squeeze(-1)
            valid_mask = torch.ones_like(token_mask)
            if attention_mask is not None:
                valid_mask = valid_mask * attention_mask.float()
            if special_tokens_mask is not None:
                valid_mask = valid_mask * (1 - special_tokens_mask.float())

            if slang_mask is not None:
                non_slang = valid_mask * (1 - slang_mask.float())
            else:
                non_slang = valid_mask

            denom = non_slang.sum() + 1e-8
            sparsity_loss = (token_mask.abs() * non_slang).sum() / denom
            sparsity_loss = self.mask_sparsity_lambda * sparsity_loss
            total_loss = total_loss + sparsity_loss
            losses["sparsity_loss"] = sparsity_loss
        
        # Rule diversity regularization (encourage different rules to activate)
        if rules is not None and self.rule_diversity_lambda > 0:
            rule_values = torch.cat([v for v in rules.values()], dim=1)
            
            # Encourage diversity: penalize if all rules have similar activations
            rule_std = rule_values.std(dim=1).mean()
            diversity_loss = -self.rule_diversity_lambda * rule_std
            
            total_loss = total_loss + diversity_loss
            losses["diversity_loss"] = diversity_loss
        
        losses["loss"] = total_loss
        
        return losses


class ContrastivePolarityLoss(nn.Module):
    """
    Auxiliary contrastive loss for learning polarity representations.
    
    Encourages:
    - Positive predicates to be high for positive samples
    - Negative predicates to be high for negative samples
    - Separation between positive and negative representations
    
    This is optional and can help with polarity learning.
    """
    
    def __init__(self, temperature: float = 0.07, weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
    
    def forward(
        self,
        p_pos_sem: torch.Tensor,
        p_neg_sem: torch.Tensor,
        positive_mask: torch.Tensor,  # 1 if sample has positive label
        negative_mask: torch.Tensor,  # 1 if sample has negative label
    ) -> torch.Tensor:
        """
        Compute contrastive polarity loss.
        
        Encourages predicates to align with actual labels.
        """
        loss = 0.0
        
        # Positive samples should have high p_pos_sem
        if positive_mask.sum() > 0:
            pos_loss = -torch.log(p_pos_sem[positive_mask.bool()] + 1e-8).mean()
            loss = loss + pos_loss
        
        # Negative samples should have high p_neg_sem
        if negative_mask.sum() > 0:
            neg_loss = -torch.log(p_neg_sem[negative_mask.bool()] + 1e-8).mean()
            loss = loss + neg_loss
        
        return self.weight * loss


def compute_pos_weight(
    train_rows,
    label_map: Dict[str, int],
    smoothing: float = 1.0,
) -> torch.Tensor:
    """
    Compute positive class weights for imbalanced data.
    
    Uses inverse frequency weighting.
    
    Args:
        train_rows: List of training data dicts
        label_map: Label to index mapping
        smoothing: Smoothing factor to prevent extreme weights
    """
    from .utils import ensure_list, labels_to_multi_hot
    
    counts = torch.zeros(len(label_map))
    
    for row in train_rows:
        labs = ensure_list(row.get("labels", []))
        vec = torch.tensor(labels_to_multi_hot(labs, label_map), dtype=torch.float)
        counts += vec
    
    total = len(train_rows)
    neg = total - counts
    
    # Inverse frequency with smoothing
    pos_weight = (neg + smoothing) / (counts + smoothing)
    
    # Clip extreme values
    pos_weight = pos_weight.clamp(min=0.5, max=10.0)
    
    return pos_weight
