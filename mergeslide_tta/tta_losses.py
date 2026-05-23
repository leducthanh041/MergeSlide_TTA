"""
tta_losses.py - Loss functions for MergeSlide-TTA.

Sources:
  entropy_loss      : TENT (Wang et al., ICLR 2021)
  diversity_loss    : SHOT (Liang et al., TPAMI 2021)
  select_confident  : TPT  (Shu et al., NeurIPS 2022)
  l2_anchor_loss    : EATA (Niu et al., ICML 2022) simplified
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy over batch. logits: [N, C]"""
    probs = F.softmax(logits, dim=1).clamp(min=1e-8)
    return -(probs * probs.log()).sum(dim=1).mean()


def diversity_loss(logits: torch.Tensor) -> torch.Tensor:
    """Marginal entropy over batch (SHOT diversity term). logits: [N, C]"""
    mean_probs = F.softmax(logits, dim=1).mean(dim=0).clamp(min=1e-8)
    return -(mean_probs * mean_probs.log()).sum()


def dual_level_tta_loss(
    class_logits: torch.Tensor,
    task_logits:  torch.Tensor,
    alpha:        float = 0.5,
    use_diversity: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined class-level and task-level entropy objective.

    TCP mode  (use_diversity=True):
        L = (H_class - H_class_div) + alpha * (H_task - H_task_div)

    Naive mode (use_diversity=False, alpha=0):
        L = H_class
        Diversity excluded: over 13 classes it pushes mean prediction toward
        uniform distribution, which is counterproductive for 1-slide TTA.

    Args:
        class_logits  : [N, C_task] (tcp) or [N, C_total] (naive)
        task_logits   : [N, T]
        alpha         : task loss weight (set 0.0 for naive)
        use_diversity : include diversity term (set False for naive)

    Returns:
        loss (scalar), log_dict
    """
    l_class_ent = entropy_loss(class_logits)
    l_class_div = diversity_loss(class_logits) if use_diversity else torch.tensor(0.0)
    l_class     = l_class_ent - l_class_div

    l_task_ent  = entropy_loss(task_logits)
    l_task_div  = diversity_loss(task_logits) if use_diversity else torch.tensor(0.0)
    l_task      = l_task_ent - l_task_div

    total = l_class + alpha * l_task

    log = {
        "loss/class_ent":  l_class_ent.item(),
        "loss/class_div":  l_class_div.item() if use_diversity else 0.0,
        "loss/task_ent":   l_task_ent.item(),
        "loss/task_div":   l_task_div.item() if use_diversity else 0.0,
        "loss/total":      total.item(),
    }
    return total, log


def select_confident_subbags(
    logits:    torch.Tensor,
    top_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-(top_ratio * N) sub-bags with lowest entropy (most confident).
    TPT-style confidence selection.

    Args:
        logits    : [N, C]
        top_ratio : fraction to keep

    Returns:
        selected_logits : [K, C]
        selected_idx    : [K]
    """
    with torch.no_grad():
        ent = -(F.softmax(logits, dim=1).clamp(min=1e-8) *
                F.log_softmax(logits, dim=1)).sum(dim=1)
    k   = max(1, int(ent.size(0) * top_ratio))
    idx = torch.argsort(ent)[:k]
    return logits[idx], idx


def l2_anchor_loss(
    params:   List[torch.Tensor],
    params_0: List[torch.Tensor],
) -> torch.Tensor:
    """
    L2 regularization toward initial parameter values (EATA simplified).

    params   : current LN params (requires_grad=True)
    params_0 : initial LN params (detached)
    """
    return sum(((p - p0) ** 2).sum() for p, p0 in zip(params, params_0))