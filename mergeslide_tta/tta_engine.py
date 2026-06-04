"""
mergeslide_tta/tta_engine.py
============================
MergeSlide-TTA Engine — Online PETAL-WSI Test-Time Adaptation.

Architecture:
    anchor   (θ_0)  : frozen merged weights — bộ nhớ dài hạn bất biến
    student  (θ_s)  : chỉ LayerNorm trainable — gradient update mỗi slide
    teacher  (θ')   : EMA của student — dùng cho pseudo-labels và inference

Modules:
    A : WSI Bag-level Augmentation       (Frozen-to-Paraffin, MICCAI 2021)
    B : DaPC Pseudo-label Correction     (WeiContrast-DaPC, IEEE TETC 2025)
    C : PETAL Loss = Bayesian reg + CE   (PETAL, CVPR 2023)
    D : FIM-based Parameter Restoration  (PETAL, CVPR 2023)
    E : Adaptive LR via OOD score        (NOTE-inspired, NeurIPS 2022)
    P7: TCP Confidence Gate              (MergeSlide, arXiv 2025)

Usage::
    adapter = MergeSlide_TTA_Adapter(...)
    pred_local, pred_task, debug = adapter.adapt_and_predict(features, coords)
    adapter.reset_to_source()           # giữa các task nếu cần
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TTAConfig:
    # Module A — WSI Bag Augmentation (Frozen-to-Paraffin)
    K: int          = 8      # số augmented views cho teacher
    r_patch: float  = 0.75   # fraction của K_PATCHES cho mỗi aug view

    # Module B — DaPC pseudo-label correction
    tau_ap_ind: float = 0.92   # ngưỡng anchor confidence — IND
    tau_ap_ood: float = 0.70   # ngưỡng anchor confidence — OOD
    beta_ind: float   = 1.2    # DaPC sensitivity — IND
    beta_ood: float   = 1.5    # DaPC sensitivity — OOD

    # Module C — PETAL loss
    spw: float          = 1e-6  # Bayesian regularizer weight
    lam_ce: float       = 1.0   # cross-entropy weight (PETAL CE term)
    gamma_class: float  = 1.0   # L_class weight
    tau_c: float        = 0.07  # temperature cho L_class

    # Module D — FIM Restoration
    delta: float = 0.03   # quantile: reset bottom delta-% FIM params

    # Module E — Adaptive LR
    eta_base: float = 1e-4  # base learning rate
    tau_ood: float  = 0.5   # sigmoid midpoint cho OOD score

    # EMA teacher
    ema_alpha: float = 0.999  # EMA momentum

    # Phase 7 — TCP Gate
    tau_task: float = 0.70   # minimum TCP confidence để dùng routing

    # Misc
    eps: float          = 1e-6    # numerical stability
    k_patches_std: int  = 400     # K_PATCHES (standard subsample per slide)


# ──────────────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────────────

class MergeSlide_TTA_Adapter:
    """
    Online, exemplar-free TTA adapter cho MergeSlide.

    Parameters
    ----------
    base_vision_encoder : nn.Module
        TITAN vision_encoder đã load (sẽ được deepcopy → 3 instances).
    backbone_sd : dict
        State-dict từ merged_final.pth (raw vision_encoder keys).
    mean_sd : dict
        SWAG mean — param_name → Tensor.
    var_sd : dict
        SWAG variance — param_name → Tensor  (clipped ≥ eps).
    per_task_mlp_weights : dict[int, tuple[Tensor, Tensor]]
        {task_id: (weight [n_cls, 768], bias [n_cls])} đã trên device.
        Được build từ FORWARD classifier, index theo thứ tự CURRENT task.
    global_mlp_weight : Tensor  shape [13, 768]
        Class prototype weights trong CURRENT order (dùng cho fallback).
    global_mlp_bias : Tensor  shape [13]
    task_prompts : Tensor  shape [T, 768]
        Task-level prompt embeddings (đã reorder theo current order).
    task_class_ranges : dict[int, list[int]]
        {task_id: [start, end]} — current order global indices.
    cfg : TTAConfig
    device : torch.device
    """

    def __init__(
        self,
        base_vision_encoder: nn.Module,
        backbone_sd: dict,
        mean_sd: dict,
        var_sd: dict,
        per_task_mlp_weights: dict,
        global_mlp_weight: Tensor,
        global_mlp_bias: Tensor,
        task_prompts: Tensor,
        task_class_ranges: dict,
        cfg: TTAConfig,
        device: torch.device,
    ) -> None:
        self.cfg    = cfg
        self.device = device

        # ── Frozen MLP weights (prompt prototypes) ──────────────────────────
        # per_task_mlp_weights[t] = (weight [n_cls, 768], bias [n_cls])
        self.per_task_w: dict[int, Tensor] = {
            t: w.to(device) for t, (w, _) in per_task_mlp_weights.items()
        }
        self.per_task_b: dict[int, Tensor] = {
            t: b.to(device) for t, (_, b) in per_task_mlp_weights.items()
        }
        self.global_w = global_mlp_weight.to(device)   # [13, 768]
        self.global_b = global_mlp_bias.to(device)     # [13]

        # ── Task prompts (for TCP routing) ──────────────────────────────────
        self.task_prompts = task_prompts.to(device)    # [T, 768]
        self.task_class_ranges = task_class_ranges     # {t: [start, end]}

        # ── SWAG posterior statistics (Bayesian regularizer) ────────────────
        # Only need LN param names for efficient regularizer + FIM restore
        self.mean_sd = mean_sd
        self.var_sd  = var_sd

        # ── TITAN ps argument ───────────────────────────────────────────────
        from mergeslide_tta.constants import TITAN_PS_ARG
        self.ps = torch.tensor(TITAN_PS_ARG).int().to(device)

        # ── Source anchor state dict (for FIM restore) ──────────────────────
        # Store on CPU to save VRAM; move to device on demand
        self.anchor_sd = {k: v.clone().cpu() for k, v in backbone_sd.items()}

        # ── Build 3 backbone instances ───────────────────────────────────────
        self.anchor  = self._make_backbone(base_vision_encoder, backbone_sd, train=False)
        self.student = self._make_backbone(base_vision_encoder, backbone_sd, train=True)
        self.teacher = self._make_backbone(base_vision_encoder, backbone_sd, train=False)

        # ── Identify LayerNorm trainable parameters in student ───────────────
        self.ln_param_names: list[str] = [
            name
            for name, module in self.student.named_modules()
            if isinstance(module, nn.LayerNorm)
            for name_p, _ in module.named_parameters()
            for name in [f"{name}.{name_p}"]  # flatten full name
        ]
        # Simpler: iterate student.named_parameters() and check requires_grad
        self.ln_param_names = [
            n for n, p in self.student.named_parameters() if p.requires_grad
        ]

        # ── Optimizer (student LN params only) ──────────────────────────────
        ln_params = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(ln_params, lr=cfg.eta_base, weight_decay=0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @torch.enable_grad()
    def adapt_and_predict(
        self,
        features: Tensor,           # [N, 768] — full slide patches on device
        coords: Tensor,             # [N, 2]   — coords on device
        task_id: Optional[int] = None,   # None → CLASS-IL; int → TASK-IL
        use_tcp_gate: bool = True,       # False → naive global-MLP fallback
    ) -> tuple[int, int, dict]:
        """
        Thực hiện 7 phases cho một slide, trả về prediction.

        Returns
        -------
        pred_local : int
            Local class index trong không gian của pred_task.
        pred_task : int
            Task identity được predict (hoặc ground-truth với TASK-IL).
        debug : dict
            {ood_score, eta_eff, tcp_conf, t_adapt, use_aug,
             loss_petal, loss_class, loss_total}
        """
        N = features.shape[0]

        # ── Phase 1a: Standard patch subsample ──────────────────────────────
        k = min(self.cfg.k_patches_std, N)
        idx = torch.randperm(N, device=self.device)[:k]
        feat_std  = features[idx]   # [k, 768]
        coord_std = coords[idx]     # [k, 2]

        # ── Phase 1b: Anchor + Teacher std forward (no grad, bfloat16) ───────
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_anchor      = self.anchor(feat_std, coord_std, self.ps).float()   # [1, 768]
            z_teacher_std = self.teacher(feat_std, coord_std, self.ps).float()  # [1, 768]

        # ── Phase 1c: OOD score (Module E — NOTE-inspired confidence gap) ───
        # Dùng global 13-class logits để tránh dependency vào task_id
        y_anchor_g  = F.softmax(self._global_logits(z_anchor),      dim=-1)  # [1, 13]
        y_teacher_g = F.softmax(self._global_logits(z_teacher_std), dim=-1)  # [1, 13]
        max_anchor_g  = y_anchor_g.max().item()
        max_teacher_g = y_teacher_g.max().item()

        # Absolute value để handle cả hai hướng drift
        ood_score = abs(1.0 - max_anchor_g / (max_teacher_g + self.cfg.eps))

        # Adaptive learning rate: sigmoid shift theo OOD score
        eta_eff = self.cfg.eta_base * torch.sigmoid(
            torch.tensor(ood_score - self.cfg.tau_ood, dtype=torch.float32)
        ).item()

        # ── Phase 1d: Task routing cho adaptation ────────────────────────────
        if task_id is not None:
            t_adapt = task_id   # TASK-IL: ground-truth task
        else:
            t_adapt = self._tcp_route(z_anchor)   # CLASS-IL: anchor routing

        # ── Phase 1e: DaPC pseudo-label (Module B) ───────────────────────────
        # Chọn threshold theo IND/OOD regime
        if ood_score < self.cfg.tau_ood:
            tau_ap = self.cfg.tau_ap_ind
            beta   = self.cfg.beta_ind
        else:
            tau_ap = self.cfg.tau_ap_ood
            beta   = self.cfg.beta_ood

        # Anchor task-specific prediction
        y_anchor_task = F.softmax(self._task_logits(z_anchor, t_adapt), dim=-1)  # [1, n_cls]
        max_anchor_task = y_anchor_task.max().item()

        use_aug = (max_anchor_g < tau_ap)

        if use_aug:
            # OOD regime: augmentation-averaged teacher (Module A)
            y_tilde = self._aug_teacher_pred(features, coords, t_adapt, N)   # [1, n_cls]
        else:
            # IND regime: direct teacher prediction
            y_tilde = F.softmax(self._task_logits(z_teacher_std, t_adapt), dim=-1)

        # DaPC blend
        if beta * y_tilde.max().item() > max_anchor_task:
            y_corrected = y_tilde.detach()           # OOD heavy: tin teacher
        else:
            y_corrected = 0.5 * (y_tilde + y_anchor_task).detach()  # IND: blend

        # ── Phase 2: Student forward (with grad) ─────────────────────────────
        self.student.train()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_s = self.student(feat_std, coord_std, self.ps)
        z_s_f32 = z_s.float()   # [1, 768] — explicit float32 cho loss

        logits_s = self._task_logits(z_s_f32, t_adapt)   # [1, n_cls]

        # ── Phase 3: Loss computation (Module C — PETAL) ─────────────────────
        # Bayesian regularizer: log q(θ_s) ≈ -0.5 * Σ (θ-μ)²/(σ²+ε)
        log_q = self._bayesian_log_q()   # scalar Tensor with grad

        # PETAL CE term: student đồng ý với corrected pseudo-label
        loss_ce = -(
            y_corrected * F.log_softmax(logits_s, dim=-1)
        ).sum(dim=-1).mean()

        # L_PETAL = -spw * log_q + lam_ce * H_xe
        loss_petal = -self.cfg.spw * log_q + self.cfg.lam_ce * loss_ce

        # L_class: temperature-scaled CE để push Z_s align với class prototypes
        loss_class = -(
            y_corrected * F.log_softmax(logits_s / self.cfg.tau_c, dim=-1)
        ).sum(dim=-1).mean()

        loss_total = loss_petal + self.cfg.gamma_class * loss_class

        # ── Phase 4a: Backward ───────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss_total.backward()

        # ── Phase 4b: FIM computation trước optimizer step (Module D) ────────
        fisher_dict, fisher_flat = self._compute_fim()
        fim_threshold = self._find_quantile(fisher_flat, self.cfg.delta)

        # ── Phase 4c: Scale LR và step ───────────────────────────────────────
        for pg in self.optimizer.param_groups:
            pg['lr'] = eta_eff
        self.optimizer.step()
        self.optimizer.zero_grad()

        # ── Phase 5: Teacher EMA ─────────────────────────────────────────────
        self._ema_update()

        # ── Phase 6: FIM-based Parameter Restoration ─────────────────────────
        # Reset LN params với FIM thấp (ít quan trọng) về anchor θ_0
        self._fim_restore(fisher_dict, fim_threshold)

        # ── Phase 7: Inference với TCP Confidence Gate ───────────────────────
        self.teacher.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_inf = self.teacher(feat_std, coord_std, self.ps).float()  # [1, 768]

        tcp_conf = 0.0
        if task_id is not None:
            # TASK-IL: ground-truth task → direct inference
            logits_inf = self._task_logits(z_inf, task_id)
            pred_local  = int(logits_inf.argmax(-1).item())
            pred_task   = task_id
        elif use_tcp_gate:
            # CLASS-IL với TCP Gate
            tcp_scores = F.softmax(z_inf.float() @ self.task_prompts.T, dim=-1)  # [1, T]
            tcp_conf   = float(tcp_scores.max().item())

            if tcp_conf >= self.cfg.tau_task:
                # TCP routing đáng tin
                t_hat      = int(tcp_scores.argmax(-1).item())
                logits_inf = self._task_logits(z_inf, t_hat)
                pred_local = int(logits_inf.argmax(-1).item())
                pred_task  = t_hat
            else:
                # Fallback: global MLP (current order)
                logits_inf = self._global_logits(z_inf)   # [1, 13]
                g_pred     = int(logits_inf.argmax(-1).item())
                pred_task, pred_local = self._global_to_task_local(g_pred)
        else:
            # CLASS-IL naive: global MLP, no TCP gate
            logits_inf = self._global_logits(z_inf)
            g_pred     = int(logits_inf.argmax(-1).item())
            pred_task, pred_local = self._global_to_task_local(g_pred)

        debug = {
            "ood_score":   ood_score,
            "eta_eff":     eta_eff,
            "tcp_conf":    tcp_conf,
            "t_adapt":     t_adapt,
            "use_aug":     use_aug,
            "loss_petal":  float(loss_petal.detach().item()),
            "loss_class":  float(loss_class.detach().item()),
            "loss_total":  float(loss_total.detach().item()),
        }
        return pred_local, pred_task, debug

    def reset_to_source(self) -> None:
        """
        Reset student và teacher về anchor θ_0.
        Gọi sau mỗi task khi muốn bảo toàn catastrophic forgetting protection.
        """
        merged_sd = {k: v.to(self.device) for k, v in self.anchor_sd.items()}

        # Reload student weights, giữ nguyên requires_grad trên LN params
        with torch.no_grad():
            for name, param in self.student.named_parameters():
                if name in merged_sd:
                    param.data.copy_(merged_sd[name])

        # Reload teacher weights
        self.teacher.load_state_dict(merged_sd, strict=True)

        # Reset optimizer state
        ln_params = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(ln_params, lr=self.cfg.eta_base, weight_decay=0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _make_backbone(
        self, base: nn.Module, sd: dict, train: bool
    ) -> nn.Module:
        """Tạo một backbone instance từ deepcopy, load weights, set mode + grad."""
        bb = copy.deepcopy(base).to(self.device)
        bb.load_state_dict(sd, strict=True)

        if train:
            bb.train()
            # Disable all grad, then enable only LayerNorm
            for p in bb.parameters():
                p.requires_grad_(False)
            for m in bb.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad_(True)
        else:
            bb.eval()
            for p in bb.parameters():
                p.requires_grad_(False)

        return bb

    def _task_logits(self, Z: Tensor, t: int) -> Tensor:
        """logits = Z @ per_task_w[t].T + bias  →  [1, n_cls]"""
        return F.linear(Z, self.per_task_w[t], self.per_task_b[t])

    def _global_logits(self, Z: Tensor) -> Tensor:
        """logits = Z @ global_w.T + global_b  →  [1, 13]"""
        return F.linear(Z, self.global_w, self.global_b)

    def _tcp_route(self, Z: Tensor) -> int:
        """TCP routing: t_hat = argmax(Z @ task_prompts.T)"""
        with torch.no_grad():
            scores = Z.float() @ self.task_prompts.T   # [1, T]
            return int(scores.argmax(-1).item())

    def _global_to_task_local(self, g_pred: int) -> tuple[int, int]:
        """
        Map global class index (current order) → (task_id, local_class_id).
        """
        for t, (start, end) in self.task_class_ranges.items():
            if start <= g_pred <= end:
                return t, g_pred - start
        return 0, 0   # safety fallback

    def _aug_teacher_pred(
        self,
        features: Tensor,
        coords: Tensor,
        t_adapt: int,
        N: int,
    ) -> Tensor:
        """
        Module A: WSI Bag-level Augmentation (Frozen-to-Paraffin).
        K lần subsample → K teacher predictions → average.
        """
        n_aug = max(1, int(self.cfg.r_patch * self.cfg.k_patches_std))
        n_aug = min(n_aug, N)
        preds = []
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for _ in range(self.cfg.K):
                aug_idx  = torch.randperm(N, device=self.device)[:n_aug]
                z_aug    = self.teacher(features[aug_idx], coords[aug_idx], self.ps).float()
                preds.append(F.softmax(self._task_logits(z_aug, t_adapt), dim=-1))
        return torch.stack(preds).mean(0)   # [1, n_cls]

    def _bayesian_log_q(self) -> Tensor:
        """
        Module C — Gaussian log-posterior:
            log q(θ_s) = -0.5 * Σ_p (θ_s_p - μ_p)² / (σ²_p + ε)

        Chỉ tính trên LN params (non-LN không thay đổi → contribution = 0).
        """
        log_q = torch.tensor(0.0, device=self.device)
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.mean_sd:
                continue
            mu    = self.mean_sd[name].to(self.device)
            sigma2 = self.var_sd[name].to(self.device)
            log_q = log_q - 0.5 * (
                (param - mu).pow(2) / (sigma2 + self.cfg.eps)
            ).sum()
        return log_q

    def _compute_fim(self) -> tuple[dict, Tensor]:
        """
        Module D — Diagonal FIM = (∇L)²  trên LN params.
        Gọi SAU loss.backward(), TRƯỚC optimizer.step().
        """
        fisher_dict: dict[str, Tensor] = {}
        fisher_vals: list[Tensor] = []
        for name, param in self.student.named_parameters():
            if param.requires_grad and param.grad is not None:
                fim_p = param.grad.data.pow(2).clone()
                fisher_dict[name] = fim_p
                fisher_vals.append(fim_p.reshape(-1))

        if fisher_vals:
            fisher_flat = torch.cat(fisher_vals)
        else:
            fisher_flat = torch.zeros(1, device=self.device)
        return fisher_dict, fisher_flat

    @staticmethod
    def _find_quantile(arr: Tensor, perc: float) -> Tensor:
        """
        Linear-interpolation quantile (from PETAL reference implementation).
        perc=0.03 → threshold ở 3rd percentile (bottom 3% sẽ bị reset).
        """
        arr_sorted = arr.sort().values
        n = arr_sorted.numel()
        if n == 0:
            return torch.tensor(0.0, device=arr.device)
        frac_idx  = perc * (n - 1)
        low_idx   = int(frac_idx)
        high_idx  = min(low_idx + 1, n - 1)
        frac_part = frac_idx - low_idx
        return (
            arr_sorted[low_idx]
            + (arr_sorted[high_idx] - arr_sorted[low_idx]) * frac_part
        )

    def _fim_restore(self, fisher_dict: dict, threshold: Tensor) -> None:
        """
        Module D — Restore LN params với FIM < threshold về anchor θ_0.
        Tham số ít quan trọng → snap về source knowledge.
        """
        with torch.no_grad():
            for name, param in self.student.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in fisher_dict:
                    continue
                anchor_val = self.anchor_sd[name].to(self.device)
                mask = (fisher_dict[name] < threshold).float()   # 1 = restore
                param.data.copy_(
                    anchor_val * mask + param.data * (1.0 - mask)
                )

    def _ema_update(self) -> None:
        """Phase 5 — Teacher EMA: θ' ← α·θ' + (1-α)·θ_s"""
        alpha = self.cfg.ema_alpha
        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)
