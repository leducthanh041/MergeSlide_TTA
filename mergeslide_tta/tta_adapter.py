"""
tta_adapter.py - MergeSlide_TTA core module.

Pipeline per slide:
  1. Quick no-grad forward -> compute entropy (WSI-level filter)
  2. If entropy < threshold (IND slide) -> skip TTA, return directly
  3. If entropy >= threshold (OOD slide) -> create M sub-bags -> forward
     -> compute dual-level loss -> update selected backbone params -> re-infer

Batch size = M = 8 sub-bags per slide, each sub-bag has K_sub = 300 patches.

Modes:
  tcp   -- TCP routing: t_hat from task_prompts -> class logits from task MLP
  naive -- use all_class_embeddings [768, C_total] directly

Param scopes:
  ln_only -- update only LayerNorm weight/bias in the backbone
  full    -- update all backbone parameters
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mergeslide_tta.constants import TITAN_PS_ARG
from mergeslide_tta.tta_losses import (
    dual_level_tta_loss,
    entropy_loss,
    l2_anchor_loss,
    select_confident_subbags,
    select_confident_subbags_intersection,
    task_margin_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_ln_params(model: nn.Module) -> Tuple[List[torch.Tensor], List[str]]:
    """Collect weight + bias of all nn.LayerNorm in model."""
    params, names = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                params.append(module.weight)
                names.append(f"{name}.weight")
            if module.bias is not None:
                params.append(module.bias)
                names.append(f"{name}.bias")
    return params, names


def collect_adaptation_params(
    model: nn.Module,
    param_scope: str = "ln_only",
) -> Tuple[List[torch.Tensor], List[str]]:
    """Collect trainable parameters according to the chosen adaptation scope."""
    if param_scope == "ln_only":
        return collect_ln_params(model)
    if param_scope == "full":
        params, names = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append(param)
                names.append(name)
        return params, names
    raise ValueError(f"Unsupported param_scope: {param_scope}")


def configure_backbone_for_tta(
    backbone: nn.Module,
    param_scope: str = "ln_only",
) -> nn.Module:
    """Configure which backbone params are trainable during TTA."""
    backbone.train()
    backbone.requires_grad_(False)
    if param_scope == "ln_only":
        for module in backbone.modules():
            if isinstance(module, nn.LayerNorm):
                module.requires_grad_(True)
    elif param_scope == "full":
        backbone.requires_grad_(True)
    else:
        raise ValueError(f"Unsupported param_scope: {param_scope}")
    return backbone


# ---------------------------------------------------------------------------
# MergeSlide_TTA
# ---------------------------------------------------------------------------

class MergeSlide_TTA(nn.Module):
    """
    Test-Time Adaptation wrapper for MergeSlide.

    Args:
        backbone             : model.backbone (TITAN vision_encoder after merging)
        task_prompts         : [T, 768] task-level prompt embeddings (frozen)
        task_weights         : list of dict{'weight', 'bias'} MLP weights per task
        num_classes          : list of int, classes per task
        device               : torch.device
        mode                 : 'tcp', 'naive', or 'task_il'
        all_class_embeddings : [768, C_total] required when mode='naive'
        M                    : sub-bags per slide = TTA batch size (default 8)
        K_sub                : patches per sub-bag (default 300)
        top_ratio            : confident sub-bag keep ratio (default 0.5)
        alpha                : task-level loss weight (default 0.5)
        l2_anchor_beta       : L2 regularizer weight toward merged source
        lr                   : Adam learning rate (default 1e-4)
        n_steps              : adapt steps per slide (default 1)
        episodic             : reset LN params after each slide (default False = continual)
        entropy_threshold    : only TTA when entropy >= threshold (default 0.4)
    """

    def __init__(
        self,
        backbone:             nn.Module,
        task_prompts:         torch.Tensor,
        task_weights:         List[Dict],
        num_classes:          List[int],
        device:               torch.device,
        mode:                 str                     = "tcp",
        all_class_embeddings: Optional[torch.Tensor] = None,
        fixed_task_id:        Optional[int]           = None,
        param_scope:          str  = "ln_only",
        M:                    int   = 8,
        K_sub:                int   = 300,
        top_ratio:            float = 0.5,
        alpha:                float = 0.5,
        l2_anchor_beta:       float = 1.0,
        lr:                   float = 1e-4,
        n_steps:              int   = 1,
        episodic:             bool  = False,
        entropy_threshold:    float = 0.4,
        use_task_diversity:   bool  = False,
        use_task_agreement:   bool  = True,
        gamma:                float = 0.5,
        select_mode:          str   = "intersection",
        use_teacher:          bool  = True,
        ema_alpha:            float = 0.999,
        adapt_task_prompts:   bool  = True,
        ema_alpha_prompt:     float = 0.999,
        delta_margin:         float = 0.10,
        tp_anchor_beta:       float = 0.3,
        gamma_margin:         float = 0.0,
        tau_task:             float = 0.70,    # TCP confidence gate
        naive_use_task_entropy: bool = True,
        use_dapc:             bool  = False,
        dapc_loss_weight:     float = 1.0,
        entropy_loss_weight:  float = 1.0,
        dapc_tau_anchor:      float = 0.92,
        dapc_beta:            float = 1.2,
    ):
        super().__init__()

        assert mode in ("tcp", "naive", "task_il"), \
            f"mode must be 'tcp', 'naive', or 'task_il', got: {mode}"
        if mode == "naive":
            assert all_class_embeddings is not None, \
                "all_class_embeddings required when mode='naive'"
        if mode == "task_il":
            assert fixed_task_id is not None, \
                "fixed_task_id required when mode='task_il'"
        if param_scope not in ("ln_only", "full"):
            raise ValueError(f"param_scope must be 'ln_only' or 'full', got: {param_scope}")
        if l2_anchor_beta < 0:
            raise ValueError(
                f"l2_anchor_beta must be non-negative, got: {l2_anchor_beta}"
            )
        if use_dapc and not use_teacher:
            raise ValueError("DaPC requires use_teacher=True.")
        if dapc_loss_weight < 0 or entropy_loss_weight < 0:
            raise ValueError("DaPC and entropy loss weights must be non-negative.")
        if not 0.0 <= dapc_tau_anchor <= 1.0:
            raise ValueError("dapc_tau_anchor must be in [0, 1].")
        if dapc_beta <= 0:
            raise ValueError("dapc_beta must be positive.")
        self.param_scope          = param_scope
        self.backbone             = configure_backbone_for_tta(backbone, param_scope)
        self.device               = device
        self.mode                 = mode
        self.task_prompts         = task_prompts.to(device)
        self.task_weights         = task_weights
        self.num_classes          = num_classes
        self.all_class_embeddings = (
            all_class_embeddings.detach().clone().to(device)
            if all_class_embeddings is not None else None
        )
        self.fixed_task_id        = fixed_task_id
        self.M                    = M
        self.K_sub                = K_sub
        self.top_ratio            = top_ratio
        self.alpha                = alpha
        self.l2_anchor_beta       = l2_anchor_beta
        self.tau_task             = tau_task
        self.naive_use_task_entropy = naive_use_task_entropy
        self.n_steps              = n_steps
        self.episodic             = episodic
        self.entropy_threshold    = entropy_threshold
        self.use_task_diversity   = use_task_diversity
        self.use_task_agreement   = use_task_agreement
        self.gamma                = gamma
        self.select_mode          = select_mode
        assert select_mode in ("union", "intersection"), \
            f"select_mode must be 'union' or 'intersection', got: {select_mode}"
        self.ps                   = torch.tensor(TITAN_PS_ARG).int().to(device)

        self.use_teacher          = use_teacher
        self.ema_alpha            = ema_alpha
        self.adapt_task_prompts   = adapt_task_prompts
        self.ema_alpha_prompt     = ema_alpha_prompt
        self.delta_margin         = delta_margin
        self.tp_anchor_beta       = tp_anchor_beta
        self.gamma_margin         = gamma_margin
        self.use_dapc            = use_dapc
        self.dapc_loss_weight    = dapc_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.dapc_tau_anchor     = dapc_tau_anchor
        self.dapc_beta           = dapc_beta

        # task_prompts becomes mutable (working copy) + frozen source anchor
        self.task_prompts_source  = self.task_prompts.detach().clone()
        # self.task_prompts (set above) is now the *working* copy, updated
        # in-place at Phase 5b if adapt_task_prompts=True.

        # Mean-teacher: EMA copy of backbone, used for routing + final
        # inference (PETAL/CoTTA-style: teacher is more stable than the
        # backbone currently receiving gradient updates).
        if self.use_teacher:
            self.teacher = deepcopy(backbone).to(device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        else:
            self.teacher = None

        # DaPC compares the current teacher with an immutable source model.
        # Create it only for DaPC to avoid baseline VRAM cost.
        if self.use_dapc:
            self.anchor = deepcopy(backbone).to(device)
            self.anchor.eval()
            for p in self.anchor.parameters():
                p.requires_grad_(False)
        else:
            self.anchor = None

        self.n_prompt_updates = {}
        self.last_tcp_conf = float("nan")
        self.last_tcp_fallback = False

        self.n_adapted = 0
        self.n_skipped = 0

        adapt_params, self.adapt_names = collect_adaptation_params(
            self.backbone, self.param_scope
        )
        self.adapt_params = adapt_params
        self.adapt_params_anchor = [
            parameter.detach().clone() for parameter in self.adapt_params
        ]

        self.optimizer = torch.optim.Adam(adapt_params, lr=lr)

        self._init_backbone = deepcopy(self.backbone.state_dict())
        self._init_optim    = deepcopy(self.optimizer.state_dict())

        mode_info = (f"mode={mode}" if mode != "task_il"
                     else f"mode=task_il(task={fixed_task_id})")
        num_ln      = len([m for m in self.backbone.modules()
                           if isinstance(m, nn.LayerNorm)])
        n_trainable = sum(p.numel() for p in adapt_params)
        n_total     = sum(p.numel() for p in self.backbone.parameters())
        self.num_ln_layers = num_ln
        self.updated_params = n_trainable
        self.total_params = n_total
        self.update_ratio = n_trainable / max(n_total, 1)
        reset_label = "episodic_per_slide" if episodic else "continual"
        print(
            f"[MergeSlide_TTA] {mode_info} | LN layers={num_ln} | "
            f"param_scope={param_scope} | trainable_params={n_trainable:,}/{n_total:,} | "
            f"M={M} sub-bags | K_sub={K_sub} | "
            f"top_ratio={top_ratio} | alpha={alpha} | "
            f"regularizer=l2_anchor | l2_anchor_beta={l2_anchor_beta} | "
            f"lr={lr} | n_steps={n_steps} | reset={reset_label} | "
            f"entropy_threshold={entropy_threshold} | "
            f"select_mode={select_mode} | use_task_diversity={use_task_diversity} | "
            f"use_task_agreement={use_task_agreement} | gamma={gamma} | "
            f"use_teacher={use_teacher} | ema_alpha={ema_alpha} | "
            f"adapt_task_prompts={adapt_task_prompts} | ema_alpha_prompt={ema_alpha_prompt} | "
            f"delta_margin={delta_margin} | tp_anchor_beta={tp_anchor_beta} | "
            f"gamma_margin={gamma_margin} | tau_task={tau_task} | "
            f"naive_use_task_entropy={naive_use_task_entropy} | "
            f"use_dapc={use_dapc} | dapc_weight={dapc_loss_weight} | "
            f"entropy_weight={entropy_loss_weight}"
        )

    # -----------------------------------------------------------------------
    # Sub-bag creation
    # -----------------------------------------------------------------------

    def _make_subbags(
        self, features: torch.Tensor, coords: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        K    = features.shape[0]
        k_sub = min(self.K_sub, K)
        feat_list, coord_list = [], []
        for _ in range(self.M):
            idx = torch.randperm(K, device=features.device)[:k_sub]
            feat_list.append(features[idx])
            coord_list.append(coords[idx])
        return feat_list, coord_list

    # -----------------------------------------------------------------------
    # Forward sub-bags
    # -----------------------------------------------------------------------

    def _forward_subbags(
        self,
        feat_list:  List[torch.Tensor],
        coord_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Returns [M, 768] keeping computation graph alive (grad through LN)."""
        embeds = [self.backbone(f, c, self.ps)
                  for f, c in zip(feat_list, coord_list)]
        return torch.cat(embeds, dim=0)

    # -----------------------------------------------------------------------
    # Class logits by mode
    # -----------------------------------------------------------------------

    def _class_logits_tcp(
        self, embeds: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        """[N, C_task] using frozen MLP weights of task_id."""
        w = self.task_weights[task_id]["weight"].detach()
        b = self.task_weights[task_id]["bias"].detach()
        return F.linear(embeds.float(), w, b)

    def _class_logits_naive(self, embeds: torch.Tensor) -> torch.Tensor:
        """[N, C_total] using all_class_embeddings."""
        return embeds.float() @ self.all_class_embeddings.detach()

    def _global_to_task_local(self, global_column: int) -> Tuple[int, int]:
        offset = 0
        for task_id, class_count in enumerate(self.num_classes):
            if offset <= global_column < offset + class_count:
                return task_id, global_column - offset
            offset += class_count
        raise ValueError(f"global class column out of range: {global_column}")

    def _l2_regularizer(self) -> torch.Tensor:
        return self.l2_anchor_beta * l2_anchor_loss(
            self.adapt_params, self.adapt_params_anchor
        )

    # -----------------------------------------------------------------------
    # Teacher EMA
    # -----------------------------------------------------------------------

    def _ema_update_teacher(self):
        """Update teacher = EMA(backbone). Called once per adapt step."""
        if not self.use_teacher:
            return
        with torch.no_grad():
            for tp, sp in zip(self.teacher.parameters(), self.backbone.parameters()):
                tp.data.mul_(self.ema_alpha).add_(sp.data, alpha=1.0 - self.ema_alpha)

    def _teacher_or_backbone_forward(self, features, coords):
        """Use teacher for routing/final inference when enabled (more stable
        than the backbone currently receiving gradient updates); fall back
        to backbone (eval mode) otherwise."""
        model = (
            self.teacher
            if self.use_teacher and self.mode == "tcp"
            else self.backbone
        )
        was_training = model.training
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z = model(features, coords, self.ps)
        if was_training:
            model.train()
        return z.float()

    @staticmethod
    def _soft_cross_entropy(
        logits: torch.Tensor, target_probs: torch.Tensor
    ) -> torch.Tensor:
        target = target_probs.detach().to(logits.device, dtype=torch.float32)
        return -(target * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()

    def _prediction_from_embedding(
        self, embeds: torch.Tensor, task_id: Optional[int]
    ) -> torch.Tensor:
        if task_id is None:
            logits = self._class_logits_naive(embeds)
        else:
            logits = self._class_logits_tcp(embeds, task_id)
        return F.softmax(logits.float(), dim=-1)

    def _dapc_context(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        feat_list: List[torch.Tensor],
        coord_list: List[torch.Tensor],
        task_id: Optional[int],
        anchor_embedding: Optional[torch.Tensor] = None,
        compute_teacher_views: bool = True,
    ) -> dict:
        """Build detached DaPC target and reliability diagnostics."""
        if self.anchor is None or self.teacher is None:
            raise RuntimeError("DaPC context requires anchor and teacher models.")

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_anchor = (
                self.anchor(features, coords, self.ps).float()
                if anchor_embedding is None
                else anchor_embedding
            )
            z_teacher = self.teacher(features, coords, self.ps).float()

        anchor_probs = self._prediction_from_embedding(z_anchor, task_id)
        teacher_original = self._prediction_from_embedding(z_teacher, task_id)
        anchor_conf = float(anchor_probs.max(dim=-1).values.item())

        # The original teacher prediction is sufficient for a confident
        # anchor. Delay the M expensive teacher-view forwards until DaPC
        # actually needs the augmented-view average.
        teacher_average = teacher_original
        use_teacher_views = compute_teacher_views and (
            anchor_conf < self.dapc_tau_anchor
        )
        if use_teacher_views:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                teacher_views = torch.cat(
                    [
                        self.teacher(f, c, self.ps).float()
                        for f, c in zip(feat_list, coord_list)
                    ],
                    dim=0,
                )
            teacher_average = self._prediction_from_embedding(
                teacher_views, task_id
            ).mean(dim=0, keepdim=True)

        if anchor_conf >= self.dapc_tau_anchor:
            teacher_target = teacher_original
            pseudo_source = "original"
        else:
            teacher_target = teacher_average
            pseudo_source = "views"

        teacher_conf = float(teacher_target.max(dim=-1).values.item())
        if self.dapc_beta * teacher_conf > anchor_conf:
            corrected = teacher_target
            blended = False
        else:
            corrected = 0.5 * (teacher_target + anchor_probs)
            blended = True

        return {
            "target": corrected.detach(),
            "anchor_probs": anchor_probs.detach(),
            "teacher_probs": teacher_original.detach(),
            "anchor_conf": anchor_conf,
            "teacher_conf": teacher_conf,
            "pseudo_source": pseudo_source,
            "blended": blended,
            "agreement": int(
                anchor_probs.argmax(dim=-1).item()
                == teacher_target.argmax(dim=-1).item()
            ),
        }

    # -----------------------------------------------------------------------
    # Task-prompt embedding-space adaptation
    # -----------------------------------------------------------------------

    def _maybe_update_task_prompt(self, t_hat: int, z_teacher: torch.Tensor) -> bool:
        """
        Update task_prompts[t_hat] toward z_teacher IF the routing confidence
        gap (top1 - top2 over task_prompts similarity) exceeds delta_margin.

        z_teacher : [1, 768] (or [N,768], will be mean-pooled) -- embedding
                    from the STABLE teacher, not the student being adapted.

        Anchor step (tp_anchor_beta): after the EMA pull, blend back toward
        task_prompts_source[t_hat] so a single task's prompt cannot drift
        arbitrarily far over a long sequential test stream. beta=0 reproduces
        the original tta_engine_v3.py behavior (no anchor).
        """
        if not self.adapt_task_prompts:
            return False

        with torch.no_grad():
            z_mean = z_teacher.mean(dim=0, keepdim=True)          # [1, 768]
            scores = F.softmax(z_mean @ self.task_prompts.T, dim=-1)  # [1, T]
            top2   = scores.topk(2, dim=-1).values.squeeze(0)
            margin = (top2[0] - top2[1]).item()

            if margin <= self.delta_margin:
                return False

            target = (
                self.ema_alpha_prompt * self.task_prompts[t_hat]
                + (1.0 - self.ema_alpha_prompt) * z_mean.squeeze(0)
            )
            # Pull the working prompt back toward its source anchor.
            new_prompt = (
                (1.0 - self.tp_anchor_beta) * target
                + self.tp_anchor_beta * self.task_prompts_source[t_hat]
            )
            self.task_prompts[t_hat] = new_prompt

        self.n_prompt_updates[t_hat] = self.n_prompt_updates.get(t_hat, 0) + 1
        return True

    def reset_task_prompts(self):
        """Reset task prompts to source to prevent cross-task drift."""
        with torch.no_grad():
            self.task_prompts.copy_(self.task_prompts_source)
        self.n_prompt_updates = {}

    # -----------------------------------------------------------------------
    # 1 adaptation step
    # -----------------------------------------------------------------------

    @torch.enable_grad()
    def _adapt_step(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        anchor_embedding: Optional[torch.Tensor] = None,
    ) -> dict:
        feat_list, coord_list = self._make_subbags(features, coords)
        embeds = self._forward_subbags(feat_list, coord_list)   # [M, 768]
        task_logits = (
            None
            if self.mode == "naive" and not self.naive_use_task_entropy
            else embeds.float() @ self.task_prompts.T.detach()
        )

        if self.mode == "tcp":
            with torch.no_grad():
                mean_z = embeds.detach().float().mean(dim=0, keepdim=True)
                route_probs = F.softmax(
                    mean_z @ self.task_prompts.T.detach(), dim=-1
                )
                route_conf, route_idx = route_probs.max(dim=-1)
                t_hat = int(route_idx.item())
                tcp_reliable = float(route_conf.item()) >= self.tau_task
            class_logits = self._class_logits_tcp(embeds, t_hat)
        elif self.mode == "task_il":
            # Task identity known: use fixed task, no routing
            t_hat        = self.fixed_task_id
            class_logits = self._class_logits_tcp(embeds, t_hat)
            tcp_reliable = True
        else:
            t_hat        = -1
            class_logits = self._class_logits_naive(embeds)
            tcp_reliable = True

        if self.mode == "naive" and not self.naive_use_task_entropy:
            _, sel_idx = select_confident_subbags(
                class_logits.detach(), self.top_ratio
            )
        elif self.select_mode == "intersection":
            sel_idx = select_confident_subbags_intersection(
                class_logits.detach(), task_logits.detach(), self.top_ratio
            )
        else:  # "union" -- v1 behavior, kept for ablation comparison
            _, idx_class = select_confident_subbags(class_logits.detach(), self.top_ratio)
            _, idx_task  = select_confident_subbags(task_logits.detach(),  self.top_ratio)
            sel_idx      = torch.unique(torch.cat([idx_class, idx_task]))

        # Loss mode:
        #   tcp     : class entropy + class diversity + alpha * (task entropy + agreement)
        #   naive   : class entropy only (alpha=0, no diversity over 13 classes)
        #   task_il : class entropy + class diversity only (alpha=0, task routing irrelevant)
        if self.mode == "naive":
            effective_alpha = 0.0
            use_task_div    = False
            use_task_agree  = False
        elif self.mode == "task_il":
            effective_alpha = 0.0    # no task loss, task is already known
            use_task_div    = False
            use_task_agree  = False
        else:  # tcp
            effective_alpha = self.alpha
            use_task_div    = self.use_task_diversity   # default False (bug fixed)
            use_task_agree  = self.use_task_agreement

        loss, log = dual_level_tta_loss(
            class_logits[sel_idx],
            None if task_logits is None else task_logits[sel_idx],
            effective_alpha,
            use_task_diversity=use_task_div,
            use_task_agreement=use_task_agree,
            gamma=self.gamma,
        )
        loss = self.entropy_loss_weight * loss
        log["loss/entropy_objective"] = loss.item()
        if self.mode == "naive" and self.naive_use_task_entropy:
            # Match the original ablation: task entropy is diagnostic only
            # (alpha remains zero), but task confidence guides view selection.
            log["loss/task_ent"] = entropy_loss(task_logits[sel_idx]).item()

        dapc_context = None
        if self.use_dapc:
            dapc_context = self._dapc_context(
                features,
                coords,
                feat_list,
                coord_list,
                None if self.mode == "naive" else t_hat,
                anchor_embedding=anchor_embedding,
                compute_teacher_views=self.use_dapc and tcp_reliable,
            )
            log["reliability/anchor_conf"] = dapc_context["anchor_conf"]
            log["reliability/teacher_conf"] = dapc_context["teacher_conf"]
            log["reliability/agreement"] = dapc_context["agreement"]
            log["reliability/dapc_blended"] = int(dapc_context["blended"])
            log["reliability/dapc_used_views"] = int(
                dapc_context["pseudo_source"] == "views"
            )

        dapc_active = bool(self.use_dapc and tcp_reliable)
        if dapc_active:
            dapc_loss = self._soft_cross_entropy(
                class_logits[sel_idx], dapc_context["target"]
            )
            loss = loss + self.dapc_loss_weight * dapc_loss
        else:
            dapc_loss = class_logits.new_zeros(())
        log["loss/dapc"] = dapc_loss.item()
        log["reliability/dapc_active"] = int(dapc_active)

        l2_reg = self._l2_regularizer()
        loss = loss + l2_reg
        log["loss/l2_anchor"] = l2_reg.item()

        # Agreement pulls
        # sub-bags to CONCUR on routing, margin pushes whichever task
        # currently leads to lead by a clear gap.
        if self.mode == "tcp" and self.gamma_margin > 0:
            l_margin = task_margin_loss(embeds, self.task_prompts, margin=0.1)
            loss = loss + self.gamma_margin * l_margin
            log["loss/task_margin"] = l_margin.item()

        log["loss/total_with_reg"] = loss.item()
        log["adapt/t_hat"]         = t_hat
        log["adapt/n_selected"]    = sel_idx.numel()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the teacher and then the selected task prompt.
        self._ema_update_teacher()
        if self.mode == "tcp":
            # Use the mean student embedding already computed this step as a
            # cheap stand-in for a fresh teacher forward (saves 1 extra
            # backbone pass per slide); teacher weights only just shifted by
            # ema_alpha ~= 0.999 so the two are numerically close.
            prompt_updated = self._maybe_update_task_prompt(
                t_hat, embeds.detach().float()
            )
            log["adapt/prompt_updated"] = prompt_updated

        return log

    # -----------------------------------------------------------------------
    # Quick inference (no grad)
    # -----------------------------------------------------------------------

    def _quick_inference(
        self, features: torch.Tensor, coords: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, int, float]:
        """
        No-grad forward with full K patches.
        Returns (pred_class, probs[1,C], pred_task, entropy_value).

        When use_teacher=True, routing and final class prediction
        use the EMA teacher (stable) instead of the backbone currently
        receiving gradient updates.
        Also routes against the CURRENT (possibly EMA-updated) task_prompts,
        not the frozen source.
        """
        model = (
            self.teacher
            if self.use_teacher and self.mode == "tcp"
            else self.backbone
        )
        was_training = model.training
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z           = model(features, coords, self.ps)
            task_logits = z.float() @ self.task_prompts.T
            task_probs = F.softmax(task_logits, dim=1)
            tcp_conf, routed_task = task_probs.max(dim=1)
            self.last_tcp_conf = float(tcp_conf.item())
            self.last_tcp_fallback = False
            pred_task = int(routed_task.item())

            if self.mode == "tcp":
                if float(tcp_conf.item()) >= self.tau_task:
                    class_logits = self._class_logits_tcp(z.float(), pred_task)
                else:
                    self.last_tcp_fallback = True
                    global_logits = self._class_logits_naive(z.float())
                    global_column = int(global_logits.argmax(dim=1).item())
                    pred_task, pred_class = self._global_to_task_local(
                        global_column
                    )
                    start = sum(self.num_classes[:pred_task])
                    end = start + self.num_classes[pred_task]
                    global_probs = F.softmax(global_logits.float(), dim=1)
                    probs = global_probs[:, start:end]
                    entropy = -(
                        global_probs
                        * global_probs.clamp(min=1e-8).log()
                    ).sum().item()
                    if was_training and model is self.backbone:
                        model.train()
                    return pred_class, probs.cpu(), pred_task, entropy
            elif self.mode == "task_il":
                # Use fixed task -- override pred_task with known identity
                pred_task    = self.fixed_task_id
                class_logits = self._class_logits_tcp(z.float(), pred_task)
            else:
                class_logits = self._class_logits_naive(z.float())

            probs      = F.softmax(class_logits.float(), dim=1)
            pred_class = int(class_logits.argmax(dim=1))
            entropy    = -(probs * probs.clamp(min=1e-8).log()).sum().item()

        if was_training and model is self.backbone:
            model.train()

        return pred_class, probs.cpu(), pred_task, entropy

    # -----------------------------------------------------------------------
    # Public: adapt + predict
    # -----------------------------------------------------------------------

    def adapt_and_predict(
        self, features: torch.Tensor, coords: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, int, dict]:
        """
        WSI-level filter + TTA + inference for 1 slide.

        Flow:
          1. Quick forward -> entropy
          2. entropy < threshold (IND) -> skip TTA, return directly
          3. entropy >= threshold (OOD) -> TTA -> re-infer

        Returns:
            pred_class : int
            probs      : [1, C] softmax probs (CPU)
            pred_task  : int
            adapt_log  : dict
        """
        if self.episodic:
            # Reset before the entropy gate so quick inference and adaptation
            # both start from the same source state for every slide.
            self._reset()

        pred_class, probs, pred_task, entropy = self._quick_inference(
            features, coords
        )
        adapt_log = {"slide/entropy": entropy, "slide/adapted": False}
        adapt_log["slide/tcp_conf"] = self.last_tcp_conf
        adapt_log["slide/tcp_fallback"] = self.last_tcp_fallback

        if entropy < self.entropy_threshold:
            self.n_skipped += 1
            self.backbone.train()
            return pred_class, probs, pred_task, adapt_log

        self.n_adapted += 1

        self.backbone.train()

        anchor_embedding = None
        if self.use_dapc:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                anchor_embedding = self.anchor(
                    features, coords, self.ps
                ).float()

        step_logs = []
        for _ in range(self.n_steps):
            step_logs.append(
                self._adapt_step(
                    features,
                    coords,
                    anchor_embedding=anchor_embedding,
                )
            )

        adapt_log = dict(step_logs[-1])
        diagnostic_keys = {
            key
            for step_log in step_logs
            for key in step_log
            if key.startswith(("loss/", "reliability/"))
        }
        for key in diagnostic_keys:
            values = [log[key] for log in step_logs if key in log]
            adapt_log[key] = sum(values) / len(values)

        pred_class, probs, pred_task, _ = self._quick_inference(features, coords)
        adapt_log["slide/entropy"] = entropy
        adapt_log["slide/adapted"] = True
        adapt_log["slide/tcp_conf"] = self.last_tcp_conf
        adapt_log["slide/tcp_fallback"] = self.last_tcp_fallback

        self.backbone.train()
        return pred_class, probs, pred_task, adapt_log

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _reset(self):
        self.backbone.load_state_dict(self._init_backbone, strict=True)
        self.optimizer.load_state_dict(self._init_optim)
        if self.use_teacher:
            self.teacher.load_state_dict(self._init_backbone, strict=True)
        if self.adapt_task_prompts:
            self.reset_task_prompts()

    def hard_reset(self):
        """Call after each fold to restore params and reset counters."""
        self._reset()
        self.n_adapted = 0
        self.n_skipped = 0


# ---------------------------------------------------------------------------
# Helper: load task MLP weights
# ---------------------------------------------------------------------------

def load_task_weights(
    task_model_paths: List[str],
    device:           torch.device,
) -> List[Dict]:
    """
    Load MLP weight + bias for each task from task_{t}.pt checkpoint.
    Takes the last 'weight' and 'bias' keys (MLP linear head).
    """
    task_weights = []
    for path in task_model_paths:
        state = torch.load(path, map_location=device)
        keys  = list(state.keys())
        w_key = next(k for k in reversed(keys) if "weight" in k)
        b_key = next(k for k in reversed(keys) if "bias"   in k)
        task_weights.append({
            "weight": state[w_key].to(device),
            "bias":   state[b_key].to(device),
        })
    return task_weights
